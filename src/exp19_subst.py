# Data-driven discovery of interchangeable glyphs: for glyph pair (a,b), compare context distributions of
# word w and w[a->b] (both attested). High similarity relative to random same-frequency word pairs => interchangeable.
import io, contextlib, random, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from exp10_generators import *
L=[l for p in VPAGES for l in p]
c=Counter(w for l in L for w in l)
ctxL=defaultdict(Counter); ctxR=defaultdict(Counter)
for l in L:
    for i,w in enumerate(l):
        if i>0: ctxL[w][l[i-1]]+=1
        if i<len(l)-1: ctxR[w][l[i+1]]+=1
def cos(a,b):
    keys=set(a)|set(b)
    va=np.array([a[k] for k in keys],float); vb=np.array([b[k] for k in keys],float)
    return va@vb/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-9)
def csim(w,v): return 0.5*(cos(ctxL[w],ctxL[v])+cos(ctxR[w],ctxR[v]))
words=[w for w,n in c.items() if n>=8]
units=sorted({u for w in words for u in tok(w)}, key=lambda u:-sum(tok(w).count(u)*c[w] for w in words))[:22]
R=random.Random(0)
# baseline: random pairs of words with frequency >= 8
base=np.mean([csim(*R.sample(words,2)) for _ in range(3000)])
# baseline 2: pairs differing by ONE arbitrary glyph (any substitution) -> form-neighbour baseline
res=[]
for a in units:
    for b in units:
        if a>=b: continue
        sims=[]
        for w in words:
            g=tok(w)
            for i,x in enumerate(g):
                if x==a:
                    v=''.join(g[:i]+[b]+g[i+1:])
                    if c.get(v,0)>=8: sims.append(csim(w,v))
        if len(sims)>=8: res.append((np.mean(sims),len(sims),a,b))
allsub=np.mean([r[0] for r in res])
res.sort(reverse=True)
print(f'random word pairs: {base:.3f}   mean over all one-glyph substitutions: {allsub:.3f}')
for s,n,a,b in res[:18]: print(f'  {a:>4s} <-> {b:<4s}  context similarity {s:.3f}  (n={n})')
print('least interchangeable:');
for s,n,a,b in res[-6:]: print(f'  {a:>4s} <-> {b:<4s}  {s:.3f} (n={n})')
