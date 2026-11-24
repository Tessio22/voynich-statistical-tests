# Point 3: can simple mechanisms (doable by a 15th-c. scribe) reproduce Voynich statistics?
# All generators build words with a glyph Markov chain fitted to Voynich word tokens.
# We then measure properties that were NOT fitted directly.
from common import *; from parse import parse
import random, numpy as np, json, sys
from collections import defaultdict
R=random.Random(1)

# ---------- data ----------
pg=defaultdict(list)
for l in parse():
    if l['loc'] in ('P0','P1','Pb') and l['words'] and l['clean']:
        pg[l['folio']].append(l['words'])
VPAGES=[v for v in pg.values() if len(v)>=3]
LINE_LENS=[len(l) for p in VPAGES for l in p]; PAGE_LENS=[len(p) for p in VPAGES]

# ---------- glyph chain with configurable start context ----------
END='$'
def train_chain(start_ctx_fn):
    cnt=defaultdict(Counter)
    for p in VPAGES:
        for li,ws in enumerate(p):
            for wi,w in enumerate(ws):
                g=tok(w)+[END]; c=start_ctx_fn(ws,wi)
                for x in g:
                    cnt[c][x]+=1; c=(c[1],x)
    return {c:(list(v.keys()),np.array(list(v.values()),float)) for c,v in cnt.items()}
def sample_word(chain, c, bias=None):
    out=[]
    for _ in range(20):
        if c not in chain: break
        ks,ws=chain[c]
        if bias is not None:
            ws=ws*np.array([bias.get(k,1.0) for k in ks])
        x=ks[R.choices(range(len(ks)),weights=ws)[0]] if True else None
        if x==END: break
        out.append(x); c=(c[1],x)
    return ''.join(out) if out else 'o'

ctx_indep=lambda ws,wi: ('^','-')
def ctx_junction(ws,wi):
    if wi==0: return ('^','|')
    return ('^',tok(ws[wi-1])[-1])
def ctx_junction_pos(ws,wi):
    if wi==0: return ('^','|')
    b='a' if wi<=2 else ('z' if wi>=len(ws)-1 else 'm')
    return ('^'+b,tok(ws[wi-1])[-1])
CH={'indep':train_chain(ctx_indep),'junc':train_chain(ctx_junction),'jpos':train_chain(ctx_junction_pos)}

# ---------- self-citation mutations (Timm & Schinner style) ----------
SUB={'k':'t','t':'k','ch':'sh','sh':'ch','o':'a','a':'o','r':'l','l':'r','iin':'in','in':'iin','d':'s','s':'d','e':'ee','ee':'e','y':'o'}
def mutate(w):
    g=tok(w)
    for _ in range(R.choice([0,1,1,2])):
        r=R.random()
        if r<0.6 and g:
            i=R.randrange(len(g));
            if g[i] in SUB: g[i]=SUB[g[i]]
            elif g[i]=='e': g[i]='ee'
        elif r<0.8:
            if g and g[0]=='q': g=g[1:]
            elif g and g[0]=='o': g=['q']+g
            else: g=['o']+g
        else:
            if len(g)>2: g.pop(R.randrange(1,len(g)))
    return ''.join(g) if g else 'o'

def generate(kind, copy_p=0.0, drift=0.0):
    pages=[]
    for npl in PAGE_LENS:
        page=[]; recent=[]
        b1=R.gauss(0,drift); b2=R.gauss(0,drift)
        for li in range(npl):
            n=R.choice(LINE_LENS); ws=[]
            b1=0.7*b1+R.gauss(0,drift); b2=0.7*b2+R.gauss(0,drift)
            bias={'k':math.exp(b1),'t':math.exp(-b1),'ch':math.exp(b2),'sh':math.exp(-b2)} if drift else None
            for wi in range(n):
                if copy_p and recent and R.random()<copy_p:
                    cands=[mutate(x) for x in R.sample(recent[-30:],min(8,len(recent[-30:])))]
                    if wi>0 and kind!='indep':
                        last=tok(ws[-1])[-1]; bkt='a' if wi<=2 else ('z' if wi>=n-1 else 'm')
                        ks,wt=CH[kind].get(('^'+bkt,last) if kind=='jpos' else ('^',last),([],[]))
                        d=dict(zip(ks,wt)) if len(ks) else {}
                        wts=[d.get(tok(x)[0],0.01) for x in cands]
                        w=R.choices(cands,weights=wts)[0]
                    else: w=cands[0]
                else:
                    if kind=='indep': c=('^','-')
                    else:
                        tmp=ws+['x']*(n-len(ws))
                        c=(ctx_junction if kind=='junc' else ctx_junction_pos)(tmp,wi) if wi==0 else None
                        if wi>0:
                            last=tok(ws[-1])[-1]
                            if kind=='junc': c=('^',last)
                            else:
                                bkt='a' if wi<=2 else ('z' if wi>=n-1 else 'm'); c=('^'+bkt,last)
                    w=sample_word(CH[kind],c,bias)
                ws.append(w); recent.append(w)
            page.append(ws)
        pages.append(page)
    return pages

# ---------- metrics ----------
def metrics(pages, unit=tok):
    lines=[l for p in pages for l in p]; toks=[w for l in lines for w in l]
    seq=[]
    for w in toks: seq+=unit(w)+['_']
    c1=Counter(seq); c2=Counter(zip(seq,seq[1:])); h1=H(c1); h2=H(c2)-h1
    wc=Counter(toks); T=toks[:30000]
    ranks=np.arange(1,101); fr=np.array(sorted(wc.values(),reverse=True)[:100])
    zipf=np.polyfit(np.log(ranks),np.log(fr),1)[0]
    within=[(unit(l[i])[-1],unit(l[i+1])[0]) for l in lines for i in range(len(l)-1)]
    across=[]
    for p in pages:
        for k in range(len(p)-1): across.append((unit(p[k][-1])[-1],unit(p[k+1][0])[0]))
    rep=sum(l[i]==l[i+1] for l in lines for i in range(len(l)-1))/max(1,len(within))
    # locality: exact repeat within previous 30 words on same page vs random other page
    near=far=n=0
    for _ in range(3000):
        p=R.choice(pages); ws=[w for l in p for w in l]
        if len(ws)<40: continue
        i=R.randrange(30,len(ws)); w=ws[i]
        q=R.choice(pages); qs=[x for l in q for x in l]
        if q is p or len(qs)<31: continue
        j=R.randrange(len(qs)-30)
        near+= w in ws[i-30:i]; far+= w in qs[j:j+30]; n+=1
    # position gradient: share of sh- at word pos 2 vs last; first-word special
    L7=[l for l in lines if len(l)>=7]
    sh2=np.mean([unit(l[1])[0]=='sh' for l in L7]); shL=np.mean([unit(l[-1])[0]=='sh' for l in L7])
    # line heterogeneity (k vs t per-line variance relative to page-shuffle)
    def kt(pages_):
        v=[]
        for p in pages_:
            for l in p:
                us=[u for w in l for u in unit(w)]; a=us.count('k'); b=us.count('t')
                if a+b>=3: v.append(a/(a+b))
        return np.var(v)
    def shuf(pages_):
        out=[]
        for p in pages_:
            ws=[w for l in p for w in l]; R.shuffle(ws); k=0; NP=[]
            for l in p: NP.append(ws[k:k+len(l)]); k+=len(l)
            out.append(NP)
        return out
    het=kt(pages)/np.mean([kt(shuf(pages)) for _ in range(5)])
    wl=Counter(len(unit(w)) for w in toks)
    return dict(h2=round(h2,2), wlen=round(np.mean([len(unit(w)) for w in toks]),2),
                ttr30k=round(len(set(T))/len(T),3), hapax=round(sum(1 for v in wc.values() if v==1)/len(wc),3),
                zipf=round(zipf,2), MIin=round(mi_excess(within,5)[2],3), MIacross=round(mi_excess(across,5)[2],3),
                rep=round(100*rep,2), local=round(near/max(far,1),2), sh2=round(100*sh2,1), shLast=round(100*shL,1),
                ktHet=round(het,3))

if __name__=='__main__':
    rows={'VOYNICH':metrics(VPAGES)}
    cfg={'A indep words':dict(kind='indep'),
         'B +junction rule':dict(kind='junc'),
         'C +junction+position':dict(kind='jpos'),
         'D +junc+pos+drift':dict(kind='jpos',drift=0.35),
         'E self-citation only':dict(kind='indep',copy_p=0.6),
         'F jpos+drift+selfcite':dict(kind='jpos',drift=0.35,copy_p=0.4)}
    for k,c in cfg.items(): rows[k]=metrics(generate(**c))
    keys=list(rows['VOYNICH'].keys())
    print(f"{'':24s}"+''.join(f'{k:>9s}' for k in keys))
    for n,r in rows.items(): print(f'{n:24s}'+''.join(f'{r[k]:>9}' for k in keys))
    json.dump(rows,open(RESULTS+'res_exp10.json','w'),indent=1)
