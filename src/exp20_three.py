# Three "known" plants: f9v=viola, f2r=centaurea, f56r=borago.
# Is there ONE glyph->letter mapping that decodes a word near the start of each page to its plant name,
# better than for random triples of other plant names (decoys from Macer's index)?
import random, re, numpy as np, sys
from common import *; from parse import parse
L=parse()
GAL={'k','t','p','f','cth','ckh','cph','cfh'}
def cands(folio, k=6):
    ls=[l for l in L if l['folio']==folio and l['loc'].startswith('P')]
    ws=[]
    ws+=ls[0]['words'][:k]                      # start of page
    ws+=[l['words'][0] for l in ls if l['pos'] in '*@' and l['words']]  # paragraph starts
    if folio=='f2r': ws+=['ytoail','ios']        # the two labels
    out=[]
    for w in dict.fromkeys(ws):
        g=tok(w); out.append(g)
        if g and g[0] in GAL and len(g)>2: out.append(g[1:])
    return out
PAGES=['f9v','f2r','f56r']
C={p:cands(p) for p in PAGES}
def lev(a,b):
    d=list(range(len(b)+1))
    for i in range(1,len(a)+1):
        p=d[0]; d[0]=i
        for j in range(1,len(b)+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,p+(a[i-1]!=b[j-1])); p=t
    return d[-1]
def sim(dec,name):
    if len(dec)<3: return 0.0
    pre=name[:len(dec)]
    return 1-lev(dec,pre)/max(len(dec),len(pre))
UNITS=sorted({u for p in PAGES for g in C[p] for u in g})
ALPHA='abcdefghilmnopqrstuxyz'
def score(m,names):
    tot=0
    for p,nm in zip(PAGES,names):
        best=0
        for g in C[p]:
            dec=''.join(m[u] for u in g)
            for n in nm: best=max(best,sim(dec,n))
        tot+=best
    return tot/3
def optimize(names,iters=4000,restarts=3,seed=0):
    r=random.Random(seed); best=None; opts=list(ALPHA)+['']
    for _ in range(restarts):
        m={u:r.choice(ALPHA) for u in UNITS}; cur=score(m,names)
        for it in range(iters):
            u=r.choice(UNITS); old=m[u]; m[u]=r.choice(opts); s=score(m,names)
            T=0.03*(1-it/iters)+1e-4
            if s>=cur or r.random()<math.exp((s-cur)/T): cur=s
            else: m[u]=old
        if best is None or cur>best[0]: best=(cur,dict(m))
    return best
t=open(D+'latin.txt',encoding='utf-8').read(); idx=t[t.find('Index'):t.find('Index')+2000]
MACER=[norm(x).replace('j','i').replace('v','u').replace('ë','e').split()[0] for x in re.findall(r'\d+ [IVXL]+\.\s+([A-Za-zë ]+)\.',idx)]
MACER=[x for x in MACER if x not in ('uiolae','centaurea','buglossa')]
TRUE=[['uiola'],['centaurea'],['borago']]
TRUE2=[['uiola'],['cyanus'],['buglossa']]
N=int(sys.argv[1]) if len(sys.argv)>1 else 200
o1,m1=optimize(TRUE,seed=1); o2,m2=optimize(TRUE2,seed=2)
R=random.Random(9); null=[]
for k in range(N):
    trip=[[x] for x in R.sample(MACER,3)]; null.append(optimize(trip,seed=100+k)[0])
null=np.array(null)
for lab,o in [('viola/centaurea/borago',o1),('viola/cyanus/buglossa',o2)]:
    print(f'{lab:24s} score {o:.3f}   random triples {null.mean():.3f}±{null.std():.3f}  rank: {np.mean(null>=o)*100:.1f}% of random triples score >= ')
for lab,m,names in [('A',m1,TRUE),('B',m2,TRUE2)]:
    for p,nm in zip(PAGES,names):
        best=max(((sim(''.join(m[u] for u in g),n),''.join(g),''.join(m[u] for u in g),n) for g in C[p] for n in nm))
        print(f'  {lab} {p:5s} {best[1]:14s} -> {best[2]:10s} ~ {best[3]} ({best[0]:.2f})')
