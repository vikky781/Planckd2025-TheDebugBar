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

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--summary',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    summ=load_json(a.summary)
    c=canvas.Canvas(a.out, pagesize=A4)
    w,h=A4
    y=h-2*cm
    write(c,2*cm,y,"Planck'd 2025: QML Track Report",16)
    y-=1*cm
    for r in summ.get('runs',[]):
        y-=0.8*cm
        write(c,2*cm,y,f"Run: {r.get('path','')}",12)
        y-=0.6*cm
        if 'val_acc' in r:
            write(c,3*cm,y,f"Accuracy: {r['val_acc']:.4f}",11)
            y-=0.5*cm
        if 'val_auc' in r and r['val_auc']==r['val_auc']:
            write(c,3*cm,y,f"ROC-AUC: {r['val_auc']:.4f}",11)
            y-=0.5*cm
        if 'val_loss' in r:
            write(c,3*cm,y,f"Loss: {r['val_loss']:.4f}",11)
            y-=0.5*cm
        if y<3*cm:
            c.showPage()
            y=h-2*cm
    c.save()

if __name__=='__main__':
    main()
