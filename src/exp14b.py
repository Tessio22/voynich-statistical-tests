from exp14_keywords import info
from parse import parse
from collections import defaultdict
import numpy as np
pg=defaultdict(list); meta={}
for l in parse():
    if l['loc'] in ('P0','P1','Pb') and l['words'] and l['clean']:
        pg[l['folio']]+=l['words']; meta[l['folio']]=(l.get('I'),l.get('L'),l.get('H'))
for name,sel in [('Herbal, lang A, hand 1',lambda m:m==('H','A','1')),('Herbal, lang B',lambda m:m[0]=='H' and m[1]=='B'),('Balneological (B, hand 2)',lambda m:m[0]=='B'),('Stars/recipes (S, B)',lambda m:m[0]=='S')]:
    parts=[w for f,w in pg.items() if sel(meta[f]) and len(w)>=40]
    # equalize part sizes: cut into chunks of ~150 words
    ws=[x for p in parts for x in p]; P=[ws[i:i+150] for i in range(0,len(ws)-150,150)]
    r=info(P,minf=8); print(f'{name:28s} pages={len(parts):3d} chunks={len(P):3d} I={r[0]:.4f} kw={100*r[1]:.1f}% n={r[2]} top: {" ".join(r[3][:8])}')
