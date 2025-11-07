import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2)

def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)

def multiclass_roc_auc(y_true, y_proba):
    y_true_ohe=np.eye(int(np.max(y_true))+1)[y_true]
    try:
        return float(roc_auc_score(y_true_ohe, y_proba, average='macro', multi_class='ovo'))
    except Exception:
        return float('nan')

def plot_confusion(cm, labels, out):
    fig,ax=plt.subplots(figsize=(6,6))
    im=ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='black')
    fig.colorbar(im, ax=ax)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)

def plot_curves(history, out):
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(history['epoch'],history['train_loss'],label='train')
    if 'val_loss' in history:
        ax[0].plot(history['epoch'],history['val_loss'],label='val')
    ax[0].set_title('loss')
    ax[0].legend()
    ax[1].plot(history['epoch'],history['train_acc'],label='train')
    if 'val_acc' in history:
        ax[1].plot(history['epoch'],history['val_acc'],label='val')
    ax[1].set_title('acc')
    ax[1].legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)

