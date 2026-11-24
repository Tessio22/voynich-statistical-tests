import io,contextlib,numpy as np,random
with contextlib.redirect_stdout(io.StringIO()):
    from exp10_generators import *
    from exp19b import dnorm
def net(lines, N=34000, top=300, reps=10):
    toks=[w for l in lines for w in l][:N]; c=Counter(toks); keep={w for w,_ in c.most_common(top)}
    L=[];k=0
    for l in lines:
        if k>=N: break
        L.append([w if w in keep else '#' for w in l]); k+=len(l)
    def cond(Ls):
        pairs=[(a,b) for l in Ls for a,b in zip(l,l[1:])]
        j=Counter(pairs); a=Counter(p[0] for p in pairs); b=Counter(p[1] for p in pairs)
        Hb=H(b); return (H(j)-H(a))/Hb, Hb
    o,Hb=cond(L)
    r=random.Random(1); ns=[]
    for _ in range(reps):
        fl=[w for l in L for w in l]; r.shuffle(fl); i=0; S=[]
        for l in L: S.append(fl[i:i+len(l)]); i+=len(l)
        ns.append(cond(S)[0])
    # successor entropy profile: share of mass of the 10 words with the most diverse successors
    return round(o,4), round(np.mean(ns),4), round(np.mean(ns)-o,4)
V=[l for p in VPAGES for l in p]
rows=[('Voynich',net(V)),('Voynich, variants merged',net([[dnorm(w) for w in l] for l in V]))]
R.seed(1); G=[l for p in generate(kind='jpos',drift=0.2,copy_p=0.3) for l in p]
rows+= [('Generator',net(G)),('Generator, merged',net([[dnorm(w) for w in l] for l in G]))]
for n in ['latin.txt','pliny.txt','italian.txt','czech.txt','german.txt','catalan.txt','turkish.txt']:
    mode='orig' if n=='latin.txt' else 'wrap'
    rows.append((n,net(text_lines(n,mode))))
print('H(next|prev)/H(next): lower = previous word predicts more;  reduction vs shuffled order')
for n,(o,s,d) in rows: print(f'  {n:28s} {o:.3f}  shuffled {s:.3f}  reduction {d:.3f}')
