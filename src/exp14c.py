import sys, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from exp14_keywords import info, text_lines, generate, R
from parse import parse
from collections import defaultdict
import numpy as np
NCH,CS=36,150
def take(ws,off=0): 
    ws=ws[off:off+NCH*CS]; return [ws[i:i+CS] for i in range(0,len(ws),CS)] if len(ws)>=NCH*CS else None
pg=defaultdict(list); meta={}
for l in parse():
    if l['loc'] in ('P0','P1','Pb') and l['words'] and l['clean']:
        pg[l['folio']]+=l['words']; meta[l['folio']]=(l.get('I'),l.get('L'),l.get('H'))
rows=[]
def add(name,ws,offs=(0,)):
    vals=[info(take(ws,o),minf=8)[:2] for o in offs if take(ws,o)]
    if vals: rows.append((name,np.mean([v[0] for v in vals]),np.mean([v[1] for v in vals]),len(vals)))
for name,sel in [('VOY Herbal A hand1',lambda m:m==('H','A','1')),('VOY Balneological',lambda m:m[0]=='B'),('VOY Stars/recipes',lambda m:m[0]=='S')]:
    ws=[x for f,w in pg.items() if sel(meta[f]) for x in w]; add(name,ws)
for n in ['latin.txt','italian.txt','czech.txt','catalan.txt','occitan.txt']:
    ws=[w for l in text_lines(n,'wrap') for w in l]; add(n,ws,(0,6000,12000))
for lab,cfg in [('GEN drift+copy',dict(kind='jpos',drift=0.2,copy_p=0.3)),('GEN no copy',dict(kind='jpos',drift=0.2))]:
    for s in [1,2]:
        R.seed(s); G=generate(**cfg); ws=[w for p in G for l in p for w in l]; add(f'{lab} s{s}',ws)
for r in rows: print(f'{r[0]:24s} I={r[1]:.4f}  keywords={100*r[2]:5.1f}%  (samples {r[3]})')
