# Null-word hypothesis: are some frequent words fillers? Greedily remove word types that most increase
# word-to-word dependency of the remaining text. Positive control: Latin with inserted fake null words.
import io, contextlib, random, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from exp10_generators import *
    from exp19b import dnorm
def dep(lines, N=30000, top=300, reps=3, seed=1):
    toks=[w for l in lines for w in l][:N]; c=Counter(toks); keep={w for w,_ in c.most_common(top)}
    L=[];k=0
    for l in lines:
        if k>=N: break
        if len(l)>=2: L.append([w if w in keep else '#' for w in l]); k+=len(l)
    def cond(Ls):
        pairs=[(a,b) for l in Ls for a,b in zip(l,l[1:])]
        j=Counter(pairs); a=Counter(p[0] for p in pairs); b=Counter(p[1] for p in pairs)
        return (H(j)-H(a))/H(b)
    o=cond(L); r=random.Random(seed); ns=[]
    for _ in range(reps):
        fl=[w for l in L for w in l]; r.shuffle(fl); i=0; S=[]
        for l in L: S.append(fl[i:i+len(l)]); i+=len(l)
        ns.append(cond(S))
    return np.mean(ns)-o
def greedy(lines, steps=6, cand=40, tag=''):
    removed=[]; cur=lines; base=dep(cur); hist=[round(base,4)]
    for st in range(steps):
        c=Counter(w for l in cur for w in l); best=None
        for w,_ in c.most_common(cand):
            trial=[[x for x in l if x!=w] for l in cur]
            d=dep(trial)
            if best is None or d>best[0]: best=(d,w)
        if best[0]<=hist[-1]+0.0005: break
        removed.append(best[1]); cur=[[x for x in l if x!=best[1]] for l in cur]; hist.append(round(best[0],4))
    print(f'{tag:30s} start {hist[0]:.4f} -> {hist[-1]:.4f}  removed: {removed}',flush=True)
    return removed,hist
# positive control: Latin (Pliny) with 5 fake null words inserted at 15% of positions
LAT=[l for l in text_lines('pliny.txt','wrap')]
NULLS=['zork','quix','blam','frub','tesk']; r=random.Random(3); LN=[]
for l in LAT:
    o=[]
    for w in l:
        if r.random()<0.15: o.append(r.choice(NULLS))
        o.append(w)
    LN.append(o)
greedy(LAT,steps=4,tag='Latin (no nulls)')
greedy(LN,steps=7,tag='Latin + 5 fake nulls (control)')
V=[[dnorm(w) for w in l] for p in VPAGES for l in p]
greedy(V,steps=8,tag='Voynich (variants merged)')
V0=[l for p in VPAGES for l in p]
greedy(V0,steps=8,tag='Voynich (raw words)')
R.seed(1); G=[[dnorm(w) for w in l] for p in generate(kind='jpos',drift=0.2,copy_p=0.3) for l in p]
greedy(G,steps=8,tag='Generator (merged)')
