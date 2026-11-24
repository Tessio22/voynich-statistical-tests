# Distributional neighbours: do words that share contexts also share FORM (form-driven system)
# or are they often formally unrelated (meaning-driven, e.g. 'aqua' ~ 'vinum')?
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from exp10_generators import *
import numpy as np
def lev(a,b):
    d=list(range(len(b)+1))
    for i in range(1,len(a)+1):
        p=d[0]; d[0]=i
        for j in range(1,len(b)+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,p+(a[i-1]!=b[j-1])); p=t
    return d[-1]
def neigh_form(lines, unit, top=300, minf=15, k=5, dim=40, win=2):
    c=Counter(w for l in lines for w in l)
    V=[w for w,n in c.most_common() if n>=minf][:top]; ix={w:i for i,w in enumerate(V)}
    ctx=[w for w,n in c.most_common(1000)]; cx={w:i for i,w in enumerate(ctx)}
    M=np.zeros((len(V),len(ctx)))
    for l in lines:
        for i,w in enumerate(l):
            if w not in ix: continue
            for j in range(max(0,i-win),min(len(l),i+win+1)):
                if j!=i and l[j] in cx: M[ix[w],cx[l[j]]]+=1
    tot=M.sum(); pw=M.sum(1,keepdims=True)/tot; pc=M.sum(0,keepdims=True)/tot
    with np.errstate(divide='ignore',invalid='ignore'):
        P=np.log((M/tot)/(pw*pc)); P[~np.isfinite(P)]=0; P=np.maximum(P,0)
    U,S,_=np.linalg.svd(P,full_matrices=False); E=U[:,:dim]*S[:dim]
    E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
    sim=E@E.T; np.fill_diagonal(sim,-1)
    nd=[];rd=[];ex=[]
    rng=np.random.default_rng(0)
    for i,w in enumerate(V):
        nb=np.argsort(-sim[i])[:k]
        uw=unit(w)
        for j in nb: nd.append(lev(uw,unit(V[j]))/max(len(uw),len(unit(V[j]))))
        for j in rng.choice(len(V),k,replace=False):
            if j!=i: rd.append(lev(uw,unit(V[j]))/max(len(uw),len(unit(V[j]))))
        if i<8: ex.append(w+'→'+','.join(V[j] for j in nb[:3]))
    nd=np.array(nd); rd=np.array(rd)
    return nd.mean(), rd.mean(), (nd<=0.34).mean(), (rd<=0.34).mean(), ex
rows=[]
VL=[l for p in VPAGES for l in p]
rows.append(('VOYNICH',)+neigh_form(VL,tok))
for n in ['latin.txt','italian.txt','czech.txt','catalan.txt']:
    rows.append((n,)+neigh_form(text_lines(n,'wrap')[:4000],list))
for s in [1,2]:
    R.seed(s); G=generate(kind='jpos',drift=0.2,copy_p=0.3); rows.append((f'GEN s{s}',)+neigh_form([l for p in G for l in p],tok))
print(f"{'':14s} {'nbrFormDist':>11s} {'randDist':>9s} {'ratio':>6s} {'%nbrSimilar':>11s} {'%randSimilar':>12s}")
for r in rows:
    print(f'{r[0]:14s} {r[1]:11.3f} {r[2]:9.3f} {r[1]/r[2]:6.2f} {100*r[3]:10.1f}% {100*r[4]:11.1f}%   {" | ".join(r[5][:4])}')
