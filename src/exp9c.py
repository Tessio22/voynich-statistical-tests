from common import *; from parse import parse
import random, numpy as np
from collections import defaultdict
R=random.Random(9)
pg=defaultdict(list)
for l in parse():
    if l['loc'] in ('P0','P1','Pb') and len(l['words'])>=4 and l['clean']:
        pg[l['folio']].append((l['pos'],l['words']))
P=[v for v in pg.values() if len(v)>=6]
def period(P,k,rot=False):
    pairs=[]
    for p in P:
        off=R.randrange(len(p)) if rot else 0
        for i,(pos,ws) in enumerate(p):
            if pos!='+' : continue          # skip paragraph-initial lines
            for w in ws[1:]: pairs.append(((i+off)%k, tok(w)[0]))
    return mi(pairs)
for k in [2,3,4,5,6]:
    o=period(P,k); n=[period(P,k,True) for _ in range(60)]
    print(f'period {k}: MI={o:.4f} null={np.mean(n):.4f}±{np.std(n):.4f} z={(o-np.mean(n))/np.std(n):.1f}')
