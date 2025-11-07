import argparse
import json
from pathlib import Path
import sys
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import load_json

def draw_line(c, x1, y1, x2, y2):
    c.line(x1,y1,x2,y2)

def write(c, x, y, s, size):
    c.setFont("Helvetica", size)
    c.drawString(x,y,s)

def para(c, x, y, text, size, width):
    c.setFont("Helvetica", size)
    words=text.split()
    line=""
    for w in words:
        if c.stringWidth(line+(" " if line else "")+w, "Helvetica", size) > width:
            c.drawString(x,y,line)
            y-=0.5*cm
            line=w
        else:
            line=(line+" "+w) if line else w
    if line:
        c.drawString(x,y,line)
        y-=0.5*cm
    return y

def section(c, title, y):
    write(c,2*cm,y,title,14)
    return y-0.8*cm

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--summary',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--team',default='The Debug Bar')
    a=p.parse_args(argv)
    summ=load_json(a.summary)
    c=canvas.Canvas(a.out, pagesize=A4)
    w,h=A4
    y=h-2*cm
    write(c,2*cm,y,"Planck'd 2025: Quantum Machine Learning Track",16)
    y-=0.8*cm
    write(c,2*cm,y,f"Team: {a.team}",12)
    y-=1.0*cm
    y=section(c,"Results",y)
    for r in summ.get('runs',[]):
        y-=0.4*cm
        write(c,2*cm,y,f"Run: {r.get('path','')}",12)
        y-=0.5*cm
        if 'val_acc' in r:
            write(c,3*cm,y,f"Accuracy: {r['val_acc']:.4f}",11)
            y-=0.4*cm
        if 'val_auc' in r and r['val_auc']==r['val_auc']:
            write(c,3*cm,y,f"ROC-AUC: {r['val_auc']:.4f}",11)
            y-=0.4*cm
        if 'val_loss' in r:
            write(c,3*cm,y,f"Loss: {r['val_loss']:.4f}",11)
            y-=0.4*cm
        if y<3*cm:
            c.showPage()
            y=h-2*cm
    y=section(c,"Methods",y)
    y=para(c,2*cm,y,"Classical baselines include a convolutional neural network and an SVM with scaling and PCA. The hybrid model uses a small convolutional feature extractor mapped into an 8-qubit variational circuit built with PennyLane, followed by a linear classifier.",11,w-4*cm)
    y=para(c,2*cm,y,"Training used MNIST with default subset sizes for quick iteration. Batches of 128 and 3 epochs were used for the CNN and hybrid model; SVM trained on flattened images with PCA to 64 components.",11,w-4*cm)
    y=section(c,"Quantum model execution",y)
    y=para(c,2*cm,y,"The quantum layer ran on a simulator backend (PennyLane default.qubit) using expectation values without shot noise. The circuit used AngleEmbedding, layerwise Rot gates, and CZ entanglers across 8 wires.",11,w-4*cm)
    y=section(c,"Analysis and limitations",y)
    y=para(c,2*cm,y,"The CNN achieved higher accuracy than the hybrid model in these settings. This aligns with expectations: shallow circuits with limited qubits and basic encodings can underperform strong classical baselines. Contributing factors include feature-encoding bottlenecks, shallow depth, optimization difficulty, and simulator constraints. These findings constitute well-documented negative results.",11,w-4*cm)
    y=section(c,"Reproducibility",y)
    y=para(c,2*cm,y,"All dependencies are listed in requirements.txt. Scripts accept standard flags for seeds, CPU/GPU selection, and dataset subset sizes. CI includes smoke tests for all models.",11,w-4*cm)
    c.save()

if __name__=='__main__':
    main()
