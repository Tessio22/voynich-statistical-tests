# Does the ending of the SAME stem change depending on the next word? (modification vs selection)
from common import *; from parse import parse
import random
def cond_mi(triples):
    # I(E;N|S) = sum_s p(s) I(E;N | S=s)
    by={}
    for s,e,n in triples: by.setdefault(s,[]).append((e,n))
    tot=len(triples); v=0
    for s,pairs in by.items():
        if len(pairs)<2: continue
        v+=len(pairs)/tot*mi(pairs)
    return v
def test(lines, unit, name, reps=20):
    tr=[]
    for ws in lines:
        for i in range(len(ws)-1):
            u=unit(ws[i])
            if len(u)<2: continue
            tr.append((tuple(u[:-1]), u[-1], unit(ws[i+1])[0]))
    # keep stems that occur with >=2 different endings and >=10 times
    c=Counter(t[0] for t in tr); ends={}
    for s,e,n in tr: ends.setdefault(s,set()).add(e)
    tr=[t for t in tr if c[t[0]]>=10 and len(ends[t[0]])>=2]
    obs=cond_mi(tr); r=random.Random(1); nulls=[]
    for _ in range(reps):
        by={}
        for i,t in enumerate(tr): by.setdefault(t[0],[]).append(i)
        nn=[t[2] for t in tr]
        for s,ix in by.items():
            vals=[nn[i] for i in ix]; r.shuffle(vals)
            for i,v in zip(ix,vals): nn[i]=v
        nulls.append(cond_mi([(t[0],t[1],x) for t,x in zip(tr,nn)]))
    m=sum(nulls)/len(nulls)
    print(f'{name:28s} n={len(tr):6d} I(end;next|stem)={obs:.4f} null={m:.4f} excess={obs-m:.4f}')
L=[l['words'] for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words']]
test(L,tok,'Voynich (glyphs)')
test(text_lines('latin.txt','orig')[:1800],list,'Latin verse')
test(text_lines('latin.txt','wrap'),list,'Latin prose')
test(text_lines('italian.txt','wrap'),list,'Italian prose')
test(text_lines('german.txt','orig'),list,'German')
