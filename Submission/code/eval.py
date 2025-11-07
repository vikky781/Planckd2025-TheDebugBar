import argparse
import os
from pathlib import Path
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from utils import save_json, load_json

def load_probs(run_dir):
    p=os.path.join(run_dir,'probs.npy')
    if os.path.exists(p):
        return np.load(p)
    return None

def gather(run_dir):
    mpath=os.path.join(run_dir,'metrics.json')
    if not os.path.exists(mpath):
        return None
    m=load_json(mpath)
    return {'path':run_dir,**m}

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--out',type=str,required=True)
    a=ap.parse_args(argv)
    rows=[]
    for r in a.runs:
        g=gather(r)
        if g is not None:
            rows.append(g)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    save_json({'runs':rows}, os.path.join(a.out,'metrics_summary.json'))

if __name__=='__main__':
    main()
