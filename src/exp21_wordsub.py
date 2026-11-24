# Word-level substitution decipherment: collapsed Voynich word type -> Latin word (homophonic allowed).
# Positive control: Latin held-out text encrypted with a homophonic word cipher of similar shape.
import sys, random, math, numpy as np, re, io, contextlib
from common import *; from parse import parse
with contextlib.redirect_stdout(io.StringIO()):
    from exp19b import dnorm
rng=np.random.default_rng(1)

def latin_lines():
    out=[]
    for name,mode in [('latin.txt','orig'),('pliny.txt','wrap')]:
        for l in text_lines(name,mode):
            ws=[w.replace('j','i').replace('v','u') for w in l if not re.fullmatch(r'[ivxlcdm]+',w)]
            if len(ws)>=3: out.append(ws)
    return out
LAT=latin_lines(); random.Random(3).shuffle(LAT)
cut=int(len(LAT)*0.8); TRAIN=LAT[:cut]; HELD=LAT[cut:]
V=3000
lc=Counter(w for l in TRAIN for w in l); vocab=[w for w,_ in lc.most_common(V)]; vix={w:i for i,w in enumerate(vocab)}; UNK=len(vocab)
A=len(vocab)+1
# bigram LM with interpolation
uni=np.ones(A); big={}
for l in TRAIN:
    ids=[vix.get(w,UNK) for w in l]
    for x in ids: uni[x]+=1
    for a,b in zip(ids,ids[1:]): big[(a,b)]=big.get((a,b),0)+1
uni/=uni.sum(); lu=np.log2(uni)
rowtot=Counter()
for (a,b),n in big.items(): rowtot[a]+=n
def lp(a,b):
    n=big.get((a,b),0); t=rowtot.get(a,0); w=t/(t+3.0) if t else 0
    return math.log2(w*(n/t if t else 0)+(1-w)*uni[b])
LPcache={}
def LP(a,b):
    k=(a,b)
    if k not in LPcache: LPcache[k]=lp(a,b)
    return LPcache[k]

def solve(seq_lines, K, iters, seed=0, lam=1.0, init=None):
    """seq_lines: list of lists of cipher symbol ids (-1 = unknown/break). K symbols."""
    r=random.Random(seed)
    occ=[[] for _ in range(K)]; flat=[]
    for li,l in enumerate(seq_lines):
        for i,s in enumerate(l):
            if s>=0: occ[s].append((li,i))
    freq=np.array([len(o) for o in occ],float)
    # init by frequency rank
    order=np.argsort(-freq); latord=np.argsort(-uni[:UNK])
    key=np.zeros(K,int)
    if init is None:
        for rnk,s in enumerate(order): key[s]=latord[rnk % len(latord)] if rnk<len(latord) else latord[r.randrange(200)]
    else: key=init.copy()
    def local(s):
        tot=0.0
        for li,i in occ[s]:
            l=seq_lines[li]
            if i>0 and l[i-1]>=0: tot+=LP(key[l[i-1]],key[s])
            if i+1<len(l) and l[i+1]>=0 and l[i+1]!=s: tot+=LP(key[s],key[l[i+1]])
        return tot
    def total():
        t=0
        for l in seq_lines:
            for a,b in zip(l,l[1:]):
                if a>=0 and b>=0: t+=LP(key[a],key[b])
        return t
    cand=list(latord[:600])
    picks=r.choices(range(K),weights=freq,k=iters)
    for it in range(iters):
        T=2.0*(1-it/iters)+0.01
        s=picks[it]; old=key[s]; before=local(s)
        key[s]=cand[r.randrange(len(cand))] if r.random()<0.7 else latord[r.randrange(60)]
        d=local(s)-before
        if not (d>=0 or r.random()<math.exp(d/T)): key[s]=old
    n=sum(1 for l in seq_lines for a,b in zip(l,l[1:]) if a>=0 and b>=0)
    return total()/n, key

def encode_latin(lines, K):
    # homophonic word cipher: frequent words get 1-3 cipher variants; keep top-K symbols, rest -> -1
    c=Counter(w for l in lines for w in l); sym={}; nxt=0; truth={}
    for w,n in c.most_common():
        nv=3 if n>200 else (2 if n>40 else 1)
        sym[w]=list(range(nxt,nxt+nv));
        for s in sym[w]: truth[s]=w
        nxt+=nv
    seq=[[r_.choice(sym[w]) for w in l] for l,r_ in [(l,random.Random(i)) for i,l in enumerate(lines)]]
    sc=Counter(s for l in seq for s in l); keep={s:i for i,(s,_) in enumerate(sc.most_common(K))}
    out=[[keep.get(s,-1) for s in l] for l in seq]; tr={keep[s]:truth[s] for s in keep}
    return out,tr

K=int(sys.argv[1]) if len(sys.argv)>1 else 300; IT=int(sys.argv[2]) if len(sys.argv)>2 else 300000
# ---- positive control ----
ctrl,truth=encode_latin(HELD+HELD[:0],K)
sc,key=solve(ctrl,K,IT,seed=1)
tok_acc=np.mean([vocab[key[s]]==truth[s] if key[s]<UNK else False for l in ctrl for s in l if s>=0])
plain=[[vix.get(w,UNK) for w in l] for l in HELD]; ref=np.mean([LP(a,b) for l in plain for a,b in zip(l,l[1:])])
print(f'CONTROL (Latin held-out, homophonic word cipher, K={K}): score {sc:.3f}  plain {ref:.3f}  token accuracy {tok_acc:.3f}',flush=True)
# ---- Voynich ----
VL=[[dnorm(w) for w in l['words']] for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words']]
vc=Counter(w for l in VL for w in l); vk={w:i for i,(w,_) in enumerate(vc.most_common(K))}
vseq=[[vk.get(w,-1) for w in l] for l in VL]
# match size to control (number of tokens)
sc_v,key_v=solve(vseq,K,IT,seed=2)
sh=[l[:] for l in vseq]; flat=[s for l in sh for s in l]; random.Random(4).shuffle(flat); k=0
for l in sh:
    for i in range(len(l)): l[i]=flat[k]; k+=1
sc_s,_=solve(sh,K,IT,seed=3)
print(f'VOYNICH collapsed words K={K}: score {sc_v:.3f}   shuffled Voynich {sc_s:.3f}',flush=True)
inv={i:w for w,i in vk.items()}
for l in VL[:8]:
    print('  ',' '.join(l),'\n   =>',' '.join(vocab[key_v[vk[w]]] if w in vk and key_v[vk[w]]<UNK else '_' for w in l))
