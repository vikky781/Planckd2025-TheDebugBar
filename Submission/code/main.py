import argparse
import json
import os
from pathlib import Path
from train import main as train_main
from eval import main as eval_main

def run():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['train','eval'],required=True)
    p.add_argument('--args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.mode=='train':
        train_main(a.args)
    else:
        eval_main(a.args)

if __name__=='__main__':
    run()
