# Improved word-substitution decipherment with frequency-band candidates and a larger Latin corpus.
import sys, random, math, re, numpy as np, io, contextlib
from common import *; from parse import parse
with contextlib.redirect_stdout(io.StringIO()):
    from exp19b import dnorm
K=int(sys.argv[1]) if len(sys.argv)>1 else 500; IT=int(sys.argv[2]) if len(sys.argv)>2 else 1000000
FILES=sys.argv[3].split(',') if len(sys.argv)>3 else ['latin.txt','pliny.txt','latin_big.txt']
def clean_lines(f, mode):
    out=[]
    for l in text_lines(f,mode):
        ws=[w.replace('j','i').replace('v','u') for w in l if not re.fullmatch(r'[ivxlcdm]+',w)]
        if len(ws)>=3: out.append(ws)
    return out
LAT=[]
for f in FILES:
    try: LAT+=clean_lines(f,'orig' if f=='latin.txt' else 'wrap')
    except FileNotFoundError: print('missing',f)
R0=random.Random(3); R0.shuffle(LAT)
cut=int(len(LAT)*0.85); TRAIN=LAT[:cut]; HELD=LAT[cut:]
print('Latin training words',sum(map(len,TRAIN)),flush=True)
V=6000
lc=Counter(w for l in TRAIN for w in l); vocab=[w for w,_ in lc.most_common(V)]; vix={w:i for i,w in enumerate(vocab)}; UNK=V
uni=np.array([lc[w] for w in vocab]+[sum(lc.values())-sum(lc[w] for w in vocab)],float); uni/=uni.sum()
big=Counter(); rowtot=Counter(); rowtypes=Counter()
for l in TRAIN:
    ids=[vix.get(w,UNK) for w in l]
    for a,b in zip(ids,ids[1:]): big[(a,b)]+=1
for (a,b),n in big.items(): rowtot[a]+=n; rowtypes[a]+=1
Dd=0.75
CACHE={}
def LP(a,b):
    k=(a,b)
    v=CACHE.get(k)
    if v is None:
        t=rowtot.get(a,0)
        if t: p=max(big.get(k,0)-Dd,0)/t + Dd*rowtypes[a]/t*uni[b]
        else: p=uni[b]
        v=math.log2(p); CACHE[k]=v
    return v
def solve(lines, K, iters, seed):
    r=random.Random(seed)
    occ=[[] for _ in range(K)]
    for li,l in enumerate(lines):
        for i,s in enumerate(l):
            if s>=0: occ[s].append((li,i))
    freq=np.array([len(o) for o in occ],float)
    srank=np.argsort(np.argsort(-freq))
    cands=[]
    for s in range(K):
        rk=srank[s]; lo=int(rk/4); hi=min(V-1,int(rk*4)+60)
        cands.append(list(range(lo,hi+1)))
    key=np.array([min(srank[s],V-1) for s in range(K)])
    def local(s):
        tot=0.0; ks=key[s]
        for li,i in occ[s]:
            l=lines[li]
            if i>0 and l[i-1]>=0: tot+=LP(key[l[i-1]],ks)
            if i+1<len(l) and l[i+1]>=0 and l[i+1]!=s: tot+=LP(ks,key[l[i+1]])
        return tot
    w=np.sqrt(freq); w/=w.sum(); picks=np.random.default_rng(seed).choice(K,size=iters,p=w)
    for it in range(iters):
        T=1.5*(1-it/iters)+0.01; s=picks[it]
        if r.random()<0.3:
            s2=picks[(it*7919)%iters]
            if s2==s: continue
            b=local(s)+local(s2); key[s],key[s2]=key[s2],key[s]; d=local(s)+local(s2)-b
            if not (d>=0 or r.random()<math.exp(d/T)): key[s],key[s2]=key[s2],key[s]
        else:
            old=key[s]; b=local(s); key[s]=r.choice(cands[s]); d=local(s)-b
            if not (d>=0 or r.random()<math.exp(d/T)): key[s]=old
    tot=0;n=0
    for l in lines:
        for a,b in zip(l,l[1:]):
            if a>=0 and b>=0: tot+=LP(key[a],key[b]); n+=1
    return tot/n,key
def encode(lines,K):
    c=Counter(w for l in lines for w in l); sym={}; nxt=0; truth={}
    for w_,n in c.most_common():
        nv=3 if n>300 else (2 if n>60 else 1); sym[w_]=list(range(nxt,nxt+nv))
        for s in sym[w_]: truth[s]=w_
        nxt+=nv
    rr=random.Random(1); seq=[[rr.choice(sym[w_]) for w_ in l] for l in lines]
    sc=Counter(s for l in seq for s in l); keep={s:i for i,(s,_) in enumerate(sc.most_common(K))}
    return [[keep.get(s,-1) for s in l] for l in seq],{keep[s]:truth[s] for s in keep}
VL=[[dnorm(w) for w in l['words']] for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words']]
ntok=sum(map(len,VL))
held=[];k=0
for l in HELD:
    held.append(l); k+=len(l)
    if k>=ntok: break
ctrl,truth=encode(held,K)
sc,key=solve(ctrl,K,IT,1)
acc=np.mean([key[s]<UNK and vocab[key[s]]==truth[s] for l in ctrl for s in l if s>=0])
# score of the true key on control
tk=np.array([vix.get(truth[s],UNK) for s in range(K)])
tt=[LP(tk[a],tk[b]) for l in ctrl for a,b in zip(l,l[1:]) if a>=0 and b>=0]
shc=[l[:] for l in ctrl]; fl=[s for l in shc for s in l]; random.Random(9).shuffle(fl); i=0
for l in shc:
    for j in range(len(l)): l[j]=fl[i]; i+=1
scs,_=solve(shc,K,IT,4)
print(f'CONTROL: solved score {sc:.3f}, true-key score {np.mean(tt):.3f}, shuffled-control {scs:.3f}, token accuracy {acc:.3f}',flush=True)
vc=Counter(w for l in VL for w in l); vk={w:i for i,(w,_) in enumerate(vc.most_common(K))}
vseq=[[vk.get(w,-1) for w in l] for l in VL]
scv,keyv=solve(vseq,K,IT,2)
sh=[l[:] for l in vseq]; fl=[s for l in sh for s in l]; random.Random(4).shuffle(fl); i=0
for l in sh:
    for j in range(len(l)): l[j]=fl[i]; i+=1
scvs,_=solve(sh,K,IT,3)
print(f'VOYNICH: score {scv:.3f}   shuffled Voynich {scvs:.3f}',flush=True)
for l in VL[:10]:
    print('  ',' '.join(l),'\n   =>',' '.join(vocab[keyv[vk[w]]] if w in vk and keyv[vk[w]]<UNK else '_' for w in l),flush=True)
