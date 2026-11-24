import io,contextlib,numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from exp10_generators import *
def lev1_density(types):
    S=set(types); n=0
    alph=set(''.join(types))
    for w in types:
        cnt=0
        for i in range(len(w)+1):
            for a in alph:
                if w[:i]+a+w[i:] in S: cnt+=1
            if i<len(w):
                if w[:i]+w[i+1:] in S: cnt+=1
                for a in alph:
                    if a!=w[i] and w[:i]+a+w[i+1:] in S: cnt+=1
        n+=cnt
    return n/len(types)
def prof(types, unit=list):
    seq=[]
    for w in types: seq+=['^']+unit(w)+['$']
    c1=Counter(seq); c2=Counter(zip(seq,seq[1:])); h1=H(c1)
    ends=Counter(''.join(unit(w)[-2:]) if len(unit(w))>=2 else w for w in types); st=Counter(''.join(unit(w)[:2]) for w in types)
    return dict(n=len(types), alph=len(c1)-2, wlen=round(np.mean([len(unit(w)) for w in types]),2), h2=round(H(c2)-h1,2),
                top5ends=round(100*sum(v for _,v in ends.most_common(5))/len(types),1), top5starts=round(100*sum(v for _,v in st.most_common(5))/len(types),1),
                neigh=round(lev1_density(types),2), ends=' '.join(k for k,_ in ends.most_common(5)))
LI=[w.strip() for w in open(D+'lingua_ignota.txt') if w.strip()]
N=len(LI)
def top_types(words,n=N): return [w for w,_ in Counter(words).most_common(n)]
rows=[('Lingua Ignota (Hildegarda)',prof(LI))]
VW=[w for p in VPAGES for l in p for w in l]
rows.append(('Voynich (EVA písmená)',prof(top_types(VW))))
R.seed(1); GW=[w for p in generate(kind='jpos',drift=0.2,copy_p=0.3) for l in p for w in l]
rows.append(('Generátor',prof(top_types(GW))))
for n in ['latin.txt','german.txt','italian.txt','czech.txt','catalan.txt']:
    ws=[w for l in text_lines(n,'wrap') for w in l]
    rows.append((n,prof(top_types(ws))))
# Latin dictionary-like list (rarer content words, like a glossary): types ranked 200-1258
ws=[w for l in text_lines('pliny.txt','wrap') for w in l]
rows.append(('Plínius, menej časté slová (ako slovník)',prof([w for w,_ in Counter(ws).most_common(200+N)][200:])))
for name,p in rows: print(f'{name:36s}',p)
