# Davis & Layfield-style check: LSA similarity of neighbouring pages; where do neighbours NOT fit?
import numpy as np, re
from common import *; from parse import parse
from collections import defaultdict
pg=defaultdict(list); meta={}
for l in parse():
    if l['loc'] in ('P0','P1','Pb') and l['words']:
        pg[l['folio']]+=l['words']; meta[l['folio']]=(l.get('I'),l.get('L'),l.get('H'),l.get('Q'))
def key(f):
    m=re.match(r'f(\d+)([rv])(\d?)',f); return (int(m.group(1)), m.group(2)=='v', int(m.group(3) or 0))
pages=sorted([f for f in pg if len(pg[f])>=40],key=key)
vocab=[w for w,n in Counter(w for f in pages for w in pg[f]).items() if n>=3]; vi={w:i for i,w in enumerate(vocab)}
X=np.zeros((len(pages),len(vocab)))
for i,f in enumerate(pages):
    for w in pg[f]:
        if w in vi: X[i,vi[w]]+=1
df=(X>0).sum(0); X=np.log1p(X)*np.log(len(pages)/df)
U,S,Vt=np.linalg.svd(X,full_matrices=False); E=U[:,:40]*S[:40]; E/=np.linalg.norm(E,axis=1,keepdims=True)
sim=E@E.T
adj=[sim[i,i+1] for i in range(len(pages)-1)]
rnd=[sim[i,j] for i in range(len(pages)) for j in range(len(pages)) if abs(i-j)>5]
print(f'neighbouring pages mean similarity {np.mean(adj):.3f}  vs pages >5 apart {np.mean(rnd):.3f}')
same=[sim[i,i+1] for i in range(len(pages)-1) if meta[pages[i]][2]==meta[pages[i+1]][2] and meta[pages[i]][0]==meta[pages[i+1]][0]]
diff=[sim[i,i+1] for i in range(len(pages)-1) if not (meta[pages[i]][2]==meta[pages[i+1]][2] and meta[pages[i]][0]==meta[pages[i+1]][0])]
print(f'neighbours with same hand+section {np.mean(same):.3f} (n={len(same)})   different hand or section {np.mean(diff):.3f} (n={len(diff)})')
# within same hand+section: neighbours that fit worst, and best-fitting non-neighbour partner
z=[]
for i in range(len(pages)-1):
    a,b=pages[i],pages[i+1]
    if meta[a][2]==meta[b][2] and meta[a][0]==meta[b][0]:
        cand=[j for j in range(len(pages)) if j not in (i,i+1) and meta[pages[j]][2]==meta[a][2] and meta[pages[j]][0]==meta[a][0]]
        bj=max(cand,key=lambda j:sim[i,j]) if cand else None
        z.append((sim[i,i+1],a,b,pages[bj] if bj is not None else '-',sim[i,bj] if bj is not None else 0))
z.sort()
print('weakest neighbour pairs (same hand+section) and the page that would fit better:')
for s,a,b,c,sc in z[:10]: print(f'  {a:6s}->{b:6s} sim {s:.2f}   best partner for {a}: {c} ({sc:.2f})')
