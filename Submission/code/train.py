import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pennylane as qml
from pennylane import numpy as pnp
from utils import save_json, plot_confusion, plot_curves, multiclass_roc_auc

class CNN(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.c1=nn.Conv2d(1,32,3,1)
        self.c2=nn.Conv2d(32,64,3,1)
        self.p=nn.MaxPool2d(2)
        self.d1=nn.Linear(9216,128)
        self.d2=nn.Linear(128,ncls)
    def forward(self,x):
        x=F.relu(self.c1(x))
        x=F.relu(self.c2(x))
        x=self.p(x)
        x=torch.flatten(x,1)
        x=F.relu(self.d1(x))
        x=self.d2(x)
        return x

class Feat(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(1,16,3,1)
        self.c2=nn.Conv2d(16,32,3,1)
        self.p=nn.MaxPool2d(2)
        self.d=nn.Linear(4608,8)
    def forward(self,x):
        x=F.relu(self.c1(x))
        x=F.relu(self.c2(x))
        x=self.p(x)
        x=torch.flatten(x,1)
        x=self.d(x)
        return x

def vqc_layer(n_wires, n_layers):
    dev=qml.device('default.qubit',wires=n_wires)
    weight_shapes={'w':(n_layers,n_wires,3)}
    @qml.qnode(dev, interface='torch')
    def circuit(inputs,w):
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation='Y')
        for i in range(n_layers):
            for j in range(n_wires):
                qml.Rot(w[i,j,0],w[i,j,1],w[i,j,2], wires=j)
            for j in range(n_wires-1):
                qml.CZ(wires=[j,j+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]
    return qml.qnn.TorchLayer(circuit, weight_shapes)

class VQA(nn.Module):
    def __init__(self,ncls):
        super().__init__()
        self.f=Feat()
        self.q=vqc_layer(8,2)
        self.h=nn.Linear(8,ncls)
    def forward(self,x):
        x=self.f(x)
        x=self.q(x)
        x=self.h(x)
        return x

def get_loaders(batch, subset_train, subset_test, workers):
    tfm=transforms.Compose([transforms.ToTensor()])
    tr=datasets.MNIST(root='data',train=True,download=True,transform=tfm)
    te=datasets.MNIST(root='data',train=False,download=True,transform=tfm)
    if subset_train>0:
        tr=torch.utils.data.Subset(tr, list(range(subset_train)))
    if subset_test>0:
        te=torch.utils.data.Subset(te, list(range(subset_test)))
    tl=DataLoader(tr,batch_size=batch,shuffle=True,num_workers=workers)
    vl=DataLoader(te,batch_size=batch,shuffle=False,num_workers=workers)
    return tl,vl

def train_epoch(model, opt, crit, dl, device):
    model.train()
    tot=0.0
    accn=0
    den=0
    for x,y in dl:
        x=x.to(device)
        y=y.to(device)
        opt.zero_grad()
        o=model(x)
        loss=crit(o,y)
        loss.backward()
        opt.step()
        tot+=float(loss.detach().cpu())*x.size(0)
        p=o.argmax(1)
        accn+=int((p==y).sum().item())
        den+=x.size(0)
    return tot/den, accn/den

def eval_epoch(model, crit, dl, device, ncls):
    model.eval()
    tot=0.0
    accn=0
    den=0
    ys=[]
    ps=[]
    with torch.no_grad():
        for x,y in dl:
            x=x.to(device)
            y=y.to(device)
            o=model(x)
            loss=crit(o,y)
            tot+=float(loss.detach().cpu())*x.size(0)
            p=o.argmax(1)
            accn+=int((p==y).sum().item())
            den+=x.size(0)
            ys.append(y.detach().cpu().numpy())
            ps.append(torch.softmax(o,1).detach().cpu().numpy())
    y_true=np.concatenate(ys)
    y_proba=np.concatenate(ps)
    auc=multiclass_roc_auc(y_true,y_proba)
    cm=confusion_matrix(y_true, y_proba.argmax(1), labels=list(range(ncls)))
    return tot/den, accn/den, auc, cm

def run_cnn(args):
    tl,vl=get_loaders(args.batch_size,args.subset_train,args.subset_test,args.workers)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ncls=10
    model=CNN(ncls).to(device)
    opt=optim.Adam(model.parameters(), lr=args.lr)
    crit=nn.CrossEntropyLoss()
    hist={'epoch':[],'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
    for e in range(args.epochs):
        tloss,tacc=train_epoch(model,opt,crit,tl,device)
        vloss,vacc,auc,cm=eval_epoch(model,crit,vl,device,ncls)
        hist['epoch'].append(e+1)
        hist['train_loss'].append(tloss)
        hist['train_acc'].append(tacc)
        hist['val_loss'].append(vloss)
        hist['val_acc'].append(vacc)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out,'model.pt'))
    plot_curves(hist, os.path.join(args.out,'curves.png'))
    plot_confusion(cm,[str(i) for i in range(ncls)], os.path.join(args.out,'confusion.png'))
    save_json({'val_loss':vloss,'val_acc':vacc,'val_auc':auc,'history':hist}, os.path.join(args.out,'metrics.json'))


def run_vqa(args):
    tl,vl=get_loaders(args.batch_size,args.subset_train,args.subset_test,args.workers)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ncls=10
    model=VQA(ncls).to(device)
    opt=optim.Adam(model.parameters(), lr=args.lr)
    crit=nn.CrossEntropyLoss()
    hist={'epoch':[],'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
    for e in range(args.epochs):
        tloss,tacc=train_epoch(model,opt,crit,tl,device)
        vloss,vacc,auc,cm=eval_epoch(model,crit,vl,device,ncls)
        hist['epoch'].append(e+1)
        hist['train_loss'].append(tloss)
        hist['train_acc'].append(tacc)
        hist['val_loss'].append(vloss)
        hist['val_acc'].append(vacc)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out,'model.pt'))
    plot_curves(hist, os.path.join(args.out,'curves.png'))
    plot_confusion(cm,[str(i) for i in range(ncls)], os.path.join(args.out,'confusion.png'))
    save_json({'val_loss':vloss,'val_acc':vacc,'val_auc':auc,'history':hist}, os.path.join(args.out,'metrics.json'))


def run_svm(args):
    tfm=transforms.Compose([transforms.ToTensor()])
    tr=datasets.MNIST(root='data',train=True,download=True,transform=tfm)
    te=datasets.MNIST(root='data',train=False,download=True,transform=tfm)
    if args.subset_train>0:
        tr=torch.utils.data.Subset(tr, list(range(args.subset_train)))
    if args.subset_test>0:
        te=torch.utils.data.Subset(te, list(range(args.subset_test)))
    xtr=torch.stack([tr[i][0] for i in range(len(tr))]).numpy().reshape(len(tr),-1)
    ytr=np.array([tr[i][1] for i in range(len(tr))])
    xte=torch.stack([te[i][0] for i in range(len(te))]).numpy().reshape(len(te),-1)
    yte=np.array([te[i][1] for i in range(len(te))])
    pipe=Pipeline([
        ('scaler',StandardScaler()),
        ('pca',PCA(n_components=64, random_state=0)),
        ('svc',SVC(kernel='rbf',probability=True,random_state=0))
    ])
    pipe.fit(xtr,ytr)
    pro=pipe.predict_proba(xte)
    pred=pro.argmax(1)
    acc=float(accuracy_score(yte,pred))
    from joblib import dump
    Path(args.out).mkdir(parents=True, exist_ok=True)
    dump(pipe, os.path.join(args.out,'svm.joblib'))
    cm=confusion_matrix(yte,pred,labels=list(range(10)))
    auc=multiclass_roc_auc(yte, pro)
    plot_confusion(cm,[str(i) for i in range(10)], os.path.join(args.out,'confusion.png'))
    save_json({'val_acc':acc,'val_auc':auc}, os.path.join(args.out,'metrics.json'))


def build_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['mnist'],default='mnist')
    p.add_argument('--model',choices=['cnn','svm','vqa'],required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--out',type=str,required=True)
    p.add_argument('--subset-train',type=int,default=10000)
    p.add_argument('--subset-test',type=int,default=2000)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cpu',action='store_true')
    return p


def main(argv=None):
    p=build_parser()
    if argv is not None:
        a=p.parse_args(argv)
    else:
        a=p.parse_args()
    if a.model=='cnn':
        run_cnn(a)
    elif a.model=='svm':
        run_svm(a)
    else:
        run_vqa(a)

if __name__=='__main__':
    main()
