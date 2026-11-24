# EM (Baum-Welch with fixed plaintext bigram LM) word-level decipherment, Knight-style.
# Hidden: plaintext word (top V + UNK). Observed: cipher word (top K + UNK). Transitions fixed from LM; emissions learned.
import sys, re, random, numpy as np, io, contextlib
from common import *; from parse import parse
with contextlib.redirect_stdout(io.StringIO()):
    from exp19b import dnorm
from solver import arabic_words
LANG=sys.argv[1] if len(sys.argv)>1 else 'latin'
V=int(sys.argv[2]) if len(sys.argv)>2 else 400
K=int(sys.argv[3]) if len(sys.argv)>3 else 400
ITER=int(sys.argv[4]) if len(sys.argv)>4 else 40
rng=np.random.default_rng(0)

def latin_lines():
    out=[]
    for f in ['latin.txt','pliny.txt','latin_big.txt']:
        for l in text_lines(f,'orig' if f=='latin.txt' else 'wrap'):
            ws=[re.sub(r'[^a-z]','',w.replace('j','i').replace('v','u')) for w in l]
            ws=[w for w in ws if w and not re.fullmatch(r'[ivxlcdm]+',w)]
            if len(ws)>=3: out.append(ws)
    return out
def arabic_lines():
    w=arabic_words('arabic.txt'); return [w[i:i+9] for i in range(0,len(w)-9,9)]
L=latin_lines() if LANG=='latin' else arabic_lines()
random.Random(3).shuffle(L); cut=int(len(L)*0.85); TRAIN=L[:cut]; HELD=L[cut:]
c=Counter(w for l in TRAIN for w in l); vocab=[w for w,_ in c.most_common(V-1)]; vix={w:i for i,w in enumerate(vocab)}; UNK=V-1
def ids(l): return [vix.get(w,UNK) for w in l]
A=np.full((V,V),0.1); P0=np.full(V,0.1)
for l in TRAIN:
    x=ids(l); P0[x[0]]+=1
    for a,b in zip(x,x[1:]): A[a,b]+=1
uni=A.sum(0); uni/=uni.sum()
A=A/A.sum(1,keepdims=True); A=0.85*A+0.15*uni[None,:]; P0/=P0.sum()
print(f'{LANG}: training tokens {sum(map(len,TRAIN))}, V={V}, K={K}',flush=True)

def em(seqs, K, iters, seed):
    r=np.random.default_rng(seed)
    # init emissions: frequency-rank prior + noise
    cf=np.bincount(np.concatenate([np.array(s) for s in seqs]),minlength=K+1).astype(float)
    # rank-based prior: frequent plaintext words emit frequent cipher symbols
    pr=np.argsort(np.argsort(-uni)); cr=np.argsort(np.argsort(-cf))
    B=np.exp(-np.abs(np.log1p(pr)[:,None]-np.log1p(cr)[None,:]))*(0.5+r.random((V,K+1)))
    B[:,K]=0.02*B[:,:K].mean(1)
    B[UNK,:]=0.02/K; B[UNK,K]=0.98
    B/=B.sum(1,keepdims=True)
    for it in range(iters):
        cnt=np.full((V,K+1),1e-3); ll=0; n=0
        for s in seqs:
            T=len(s)
            if T==0: continue
            al=np.zeros((T,V)); sc=np.zeros(T)
            a=P0*B[:,s[0]]; sc[0]=a.sum(); al[0]=a/sc[0]
            for t in range(1,T):
                a=(al[t-1]@A)*B[:,s[t]]; sc[t]=a.sum(); al[t]=a/sc[t]
            be=np.ones(V)
            g=al[T-1]*be; cnt[:,s[T-1]]+=g/g.sum()
            for t in range(T-2,-1,-1):
                be=A@(B[:,s[t+1]]*be); be/=sc[t+1]
                g=al[t]*be; cnt[:,s[t]]+=g/g.sum()
            ll+=np.log(sc).sum(); n+=T
        B=cnt/cnt.sum(1,keepdims=True)
        B[UNK,:]=0.02/K; B[UNK,K]=0.98
    return B, ll/n/np.log(2)
def decode(s,B):
    # posterior-argmax per position
    T=len(s); al=np.zeros((T,V)); sc=np.zeros(T)
    a=P0*B[:,s[0]]; sc[0]=a.sum(); al[0]=a/sc[0]
    for t in range(1,T): a=(al[t-1]@A)*B[:,s[t]]; sc[t]=a.sum(); al[t]=a/sc[t]
    be=np.ones(V); out=[0]*T; out[T-1]=int(np.argmax(al[T-1]))
    for t in range(T-2,-1,-1):
        be=A@(B[:,s[t+1]]*be); be/=sc[t+1]; out[t]=int(np.argmax(al[t]*be))
    return out
def to_symbols(word_lines, K):
    cc=Counter(w for l in word_lines for w in l); keep={w:i for i,(w,_) in enumerate(cc.most_common(K))}
    return [[keep.get(w,K) for w in l] for l in word_lines], keep
def shuffled(seqs, seed):
    fl=[x for s in seqs for x in s]; random.Random(seed).shuffle(fl); out=[];i=0
    for s in seqs: out.append(fl[i:i+len(s)]); i+=len(s)
    return out
# ---- target size = Voynich token count
VL=[[dnorm(w) for w in l['words']] for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words']]
NT=sum(map(len,VL))
held=[];k=0
for l in HELD:
    held.append(l); k+=len(l)
    if k>=NT: break
# ---- positive control: homophonic word cipher of held-out text
cw=Counter(w for l in held for w in l); sym={}; nxt=0
for w,n in cw.most_common():
    nv=3 if n>300 else (2 if n>60 else 1); sym[w]=[f'{w}#{i}' for i in range(nv)]
rr=random.Random(1); enc=[[rr.choice(sym[w]) for w in l] for l in held]
cs,keep=to_symbols(enc,K); inv={i:w for w,i in keep.items()}
B,llc=em(cs,K,ITER,1)
cor=tot=0
for s,l in zip(cs,held):
    d=decode(s,B)
    for x,w,orig in zip(d,s,l):
        if w<K: tot+=1; cor+= (x<UNK and vocab[x]==orig)
_,llcs=em(shuffled(cs,5),K,ITER,1)
print(f'CONTROL: accuracy {cor/tot:.3f}  loglik/token {llc:.3f}  shuffled {llcs:.3f}  gain {llc-llcs:.3f}',flush=True)
# ---- Voynich
vs,vkeep=to_symbols(VL,K)
Bv,llv=em(vs,K,ITER,2)
_,llvs=em(shuffled(vs,6),K,ITER,2)
print(f'VOYNICH: loglik/token {llv:.3f}  shuffled {llvs:.3f}  gain {llv-llvs:.3f}',flush=True)
for l,s in list(zip(VL,vs))[:8]:
    d=decode(s,Bv); print('  ',' '.join(l),'\n   =>',' '.join(vocab[x] if x<UNK else '_' for x in d),flush=True)
# ---- extra controls: mechanical generator, and a WRONG language (Italian or Czech) encrypted
with contextlib.redirect_stdout(io.StringIO()):
    from exp10_generators import generate, R as GR
GR.seed(1); G=[[dnorm(w) for w in l] for p in generate(kind='jpos',drift=0.2,copy_p=0.3) for l in p]
gs,_=to_symbols(G,K); Bg,llg=em(gs,K,ITER,3); _,llgs=em(shuffled(gs,7),K,ITER,3)
print(f'GENERATOR: loglik/token {llg:.3f}  shuffled {llgs:.3f}  gain {llg-llgs:.3f}',flush=True)
for wl in ['italian.txt','czech.txt']:
    W=[l for l in text_lines(wl,'wrap')]; out=[];k=0
    for l in W:
        out.append(l); k+=len(l)
        if k>=NT: break
    cw=Counter(w for l in out for w in l); sym={}
    for w,n in cw.most_common():
        nv=3 if n>300 else (2 if n>60 else 1); sym[w]=[f'{w}#{i}' for i in range(nv)]
    rr=random.Random(2); enc=[[rr.choice(sym[w]) for w in l] for l in out]
    ws,_=to_symbols(enc,K); Bw,llw=em(ws,K,ITER,4); _,llws=em(shuffled(ws,8),K,ITER,4)
    print(f'WRONG LANGUAGE {wl}: loglik/token {llw:.3f}  shuffled {llws:.3f}  gain {llw-llws:.3f}',flush=True)
