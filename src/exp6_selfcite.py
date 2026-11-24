# Self-citation test: are words more similar to recent words (same page, previous K) than to words from elsewhere?
from common import *; from parse import parse
import random
def lev(a,b):
    d=list(range(len(b)+1))
    for i in range(1,len(a)+1):
        p=d[0]; d[0]=i
        for j in range(1,len(b)+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,p+(a[i-1]!=b[j-1])); p=t
    return d[-1]
def run(pages, unit, name, K=30, N=3000):
    r=random.Random(2)
    allw=[(pi,w) for pi,p in enumerate(pages) for w in p]
    near=[];far=[];exact_near=0;exact_far=0;cnt=0
    for _ in range(N):
        pi=r.randrange(len(pages)); p=pages[pi]
        if len(p)<K+5: continue
        i=r.randrange(K,len(p)); w=unit(p[i])
        prev=[unit(x) for x in p[i-K:i]]
        other=[]
        while len(other)<K:
            pj,x=allw[r.randrange(len(allw))]
            if pj!=pi: other.append(unit(x))
        dn=min(lev(w,x) for x in prev); df=min(lev(w,x) for x in other)
        near.append(dn); far.append(df); cnt+=1
        exact_near+= dn==0; exact_far+= df==0
    m=lambda a: sum(a)/len(a)
    print(f'{name:22s} minLev near={m(near):.3f} far={m(far):.3f} diff={m(far)-m(near):.3f} | exact repeat within {K}: near {100*exact_near/cnt:.1f}% far {100*exact_far/cnt:.1f}%')
from collections import defaultdict
pg=defaultdict(list)
for l in parse():
    if l['loc'] in ('P0','P1','Pb'): pg[l['folio']]+=l['words']
run(list(pg.values()),tok,'Voynich (glyphs)')
def chunks(lines,size=300):
    ws=[w for l in lines for w in l]; return [ws[i:i+size] for i in range(0,len(ws)-size,size)]
run(chunks(text_lines('latin.txt','orig')[:1800]),list,'Latin verse')
run(chunks(text_lines('latin.txt','wrap')[1400:]),list,'Latin prose')
run(chunks(text_lines('italian.txt','wrap')),list,'Italian')
run(chunks(text_lines('german.txt','orig')),list,'German')
