# Montemurro & Zanette-style keyword information, compared against natural texts AND mechanical generators.
from exp10_generators import *
import numpy as np, random
RR=random.Random(5)

def info(parts, minf=10, reps=20):
    toks=[(pi,w) for pi,p in enumerate(parts) for w in p]
    N=len(toks); P=len(parts)
    c=Counter(w for _,w in toks)
    words=[w for w,n in c.items() if n>=minf]
    def ent(dist):
        cnt=Counter(dist); n=sum(cnt.values()); return -sum(v/n*math.log2(v/n) for v in cnt.values())
    occ=defaultdict(list)
    for pi,w in toks: occ[w].append(pi)
    H={w:ent(occ[w]) for w in words}
    # shuffled baseline: random reassignment of part labels preserving part sizes
    labels=[pi for pi,_ in toks]; Hs=defaultdict(list)
    for _ in range(reps):
        RR.shuffle(labels); o=defaultdict(list)
        for (_,w),l in zip(toks,labels):
            if c[w]>=minf: o[w].append(l)
        for w in words: Hs[w].append(ent(o[w]))
    I=0; sig=0; per=[]
    for w in words:
        m=np.mean(Hs[w]); s=np.std(Hs[w])+1e-9
        I+= c[w]/N*(m-H[w]); z=(m-H[w])/s; per.append((z,w))
        if z>3: sig+=1
    per.sort(reverse=True)
    return I, sig/len(words), len(words), [w for _,w in per[:10]]

def chunks(lines,size=180):
    ws=[w for l in lines for w in l]; return [ws[i:i+size] for i in range(0,min(len(ws),36000)-size,size)]

rows=[]
vparts=[[w for l in p for w in l] for p in VPAGES]
rows.append(('VOYNICH (pages)',)+info(vparts))
for name,mode,lim in [('latin.txt','wrap',None),('italian.txt','wrap',None),('czech.txt','wrap',None),('catalan.txt','wrap',None)]:
    L=text_lines(name,mode); P=chunks(L,int(np.mean([len(p) for p in vparts])))
    rows.append((name,)+info(P))
for cfg,lab in [(dict(kind='jpos'),'GEN no drift, no copy'),(dict(kind='jpos',drift=0.2),'GEN drift'),(dict(kind='jpos',drift=0.2,copy_p=0.3),'GEN drift+copy (best fit)'),(dict(kind='jpos',drift=0.35,copy_p=0.3),'GEN strong drift+copy')]:
    R.seed(3); G=generate(**cfg); rows.append((lab,)+info([[w for l in p for w in l] for p in G]))
print(f"{'':28s} {'I bits/word':>12s} {'%keywords':>10s} {'nwords':>7s}  top")
for r in rows: print(f'{r[0]:28s} {r[1]:12.4f} {100*r[2]:9.1f}% {r[3]:7d}  {" ".join(r[4][:6])}')
