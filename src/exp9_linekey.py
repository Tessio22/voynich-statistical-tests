# Point 1: is there a per-line "key"? Tests:
#  (a) line heterogeneity of glyph distribution beyond page level (words permuted among lines of same page)
#  (b) periodicity with line number (mod k)
#  (c) word-position template inside lines (positions 2..n-1)
#  (d) paired-glyph ratios (k/t, ch/sh) varying per line more than chance
from common import *; from parse import parse
import random, numpy as np
from collections import defaultdict
R=random.Random(4)

def pages_voy():
    pg=defaultdict(list)
    for l in parse():
        if l['loc'] in ('P0','P1','Pb') and len(l['words'])>=4 and l['clean']:
            pg[l['folio']].append(l['words'])
    return [v for v in pg.values() if len(v)>=4], tok
def pages_txt(name,mode,size=25,lim=None):
    L=text_lines(name,mode); L=[x for x in L if len(x)>=4]
    if lim: L=L[:lim]
    return [L[i:i+size] for i in range(0,len(L)-size,size)], list

def G_het(pages, unit):
    # sum over lines of G statistic of line glyph counts vs its page distribution
    g=0
    for P in pages:
        pc=Counter(u for ws in P for w in ws for u in unit(w)); pn=sum(pc.values())
        for ws in P:
            lc=Counter(u for w in ws for u in unit(w)); ln=sum(lc.values())
            for u,c in lc.items(): g+=2*c*math.log(c/(ln*pc[u]/pn))
    return g
def shuffle_pages(pages):
    out=[]
    for P in pages:
        ws=[w for l in P for w in l]; R.shuffle(ws); k=0; NP=[]
        for l in P: NP.append(ws[k:k+len(l)]); k+=len(l)
        out.append(NP)
    return out
def ratio_var(pages, unit, a, b):
    # variance of per-line fraction a/(a+b) over lines with >=3 occurrences
    v=[]
    for P in pages:
        for ws in P:
            us=[u for w in ws for u in unit(w)]; na=us.count(a); nb=us.count(b)
            if na+nb>=3: v.append(na/(na+nb))
    return np.var(v), len(v)
def pos_template(pages, unit):
    # MI between word position (2..5, excluding first and last) and word-initial unit
    pairs=[]
    for P in pages:
        for ws in P:
            for i in range(1,min(len(ws)-1,6)): pairs.append((i,unit(ws[i])[0]))
    return mi(pairs)
def pos_shuf(pages):
    out=[]
    for P in pages:
        NP=[]
        for ws in P:
            mid=ws[1:-1]; R.shuffle(mid); NP.append([ws[0]]+mid+[ws[-1]])
        out.append(NP)
    return out
def period(pages, unit, k):
    pairs=[(i%k, unit(w)[0]) for P in pages for i,ws in enumerate(P) for w in ws]
    return mi(pairs)
def line_label_shuf(pages):
    out=[]
    for P in pages:
        Q=P[:]; R.shuffle(Q); out.append(Q)
    return out

def report(name, pages, unit, ratios):
    print('==',name,'pages',len(pages),'lines',sum(len(p) for p in pages))
    g=G_het(pages,unit); gn=[G_het(shuffle_pages(pages),unit) for _ in range(8)]
    print(f'  (a) line heterogeneity G={g:.0f} null={np.mean(gn):.0f}±{np.std(gn):.0f}  ratio={g/np.mean(gn):.3f}  z={(g-np.mean(gn))/np.std(gn):.1f}')
    for a,b in ratios:
        v,n=ratio_var(pages,unit,a,b); vn=[ratio_var(shuffle_pages(pages),unit,a,b)[0] for _ in range(20)]
        print(f'  (d) {a}/{b} per-line variance {v:.4f} null {np.mean(vn):.4f}±{np.std(vn):.4f} z={(v-np.mean(vn))/np.std(vn):.1f} (n={n})')
    t=pos_template(pages,unit); tn=[pos_template(pos_shuf(pages),unit) for _ in range(20)]
    print(f'  (c) position template MI={t:.4f} null={np.mean(tn):.4f}  z={(t-np.mean(tn))/np.std(tn):.1f}')
    for k in [2,3,4,5]:
        p=period(pages,unit,k); pn=[period(line_label_shuf(pages),unit,k) for _ in range(20)]
        print(f'  (b) period {k}: MI={p:.4f} null={np.mean(pn):.4f} z={(p-np.mean(pn))/np.std(pn):.1f}')

P,u=pages_voy(); report('Voynich',P,u,[('k','t'),('ch','sh'),('iin','in'),('d','s')])
P,u=pages_txt('latin.txt','orig',lim=1800); report('Latin verse',P,u,[('e','i'),('s','t'),('m','n')])
P,u=pages_txt('latin.txt','wrap'); report('Latin prose',P,u,[('e','i'),('s','t'),('m','n')])
P,u=pages_txt('italian.txt','wrap',lim=4000); report('Italian prose',P,u,[('e','i'),('s','t'),('o','a')])
P,u=pages_txt('german.txt','orig'); report('German MS lines',P,u,[('e','i'),('s','t'),('n','r')])
