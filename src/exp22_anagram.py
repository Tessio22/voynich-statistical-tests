# Hauer & Kondrak-style test: substitution + ANAGRAMMED words. Order inside words ignored.
# Find glyph->letter mapping maximising share of word tokens whose decoded letter-multiset is a real word (any order).
import sys, random, math, numpy as np, io, contextlib
from common import *; from parse import parse
from solver import hebrew_words, lm_text, arabic_words
with contextlib.redirect_stdout(io.StringIO()):
    from exp10_generators import generate, R as GR
def words_of_lang(name):
    if name=='hebrew.txt': return hebrew_words()
    if name in ('arabic.txt','persian.txt'): return arabic_words(name)
    nv=name.endswith(':nv'); base=name[:-3] if nv else name
    ws=[]
    files=[base] if base!='latinbig' else ['latin.txt','pliny.txt']
    for f in files:
        for l in text_lines(f,'wrap'):
            for w in l:
                w=w.replace('j','i').replace('v','u').replace('ß','ss')
                w=re.sub(r'[^a-z]','',w)
                if nv: w=re.sub(r'[aeiouy]','',w)
                if w and not re.fullmatch(r'[ivxlcdm]+',w): ws.append(w)
    return ws
def build_dict(words, alpha):
    ix={a:i for i,a in enumerate(alpha)}; d=Counter()
    for w in words:
        v=np.zeros(len(alpha),np.int8)
        for ch in w: v[ix[ch]]+=1
        d[v.tobytes()]+=1
    return d
def anagram_solve(types, freqs, S, alpha, D, iters=20000, seed=0, mind=2):
    """types: list of glyph-id arrays; returns best coverage (token share) and mapping"""
    r=random.Random(seed); A=len(alpha)
    C=np.zeros((len(types),S),np.int8)
    for i,g in enumerate(types):
        for s in g: C[i,s]+=1
    f=np.array(freqs,float); tot=f.sum()
    rows_with=[np.where(C[:,s]>0)[0] for s in range(S)]
    # init: map symbols by frequency rank to letters by frequency rank
    lf=Counter();
    for k,n in D.items():
        v=np.frombuffer(k,np.int8)
        for a in range(A): lf[a]+=int(v[a])*n
    lord=[a for a,_ in lf.most_common()]+[a for a in range(A) if a not in lf]
    sf=C.T@f; sord=np.argsort(-sf)
    m=np.zeros(S,int)
    for k,s in enumerate(sord): m[s]=lord[k%len(lord)]
    M=np.zeros((len(types),A),np.int8)
    for s in range(S): M[:,m[s]]+=C[:,s]
    ok=np.array([D.get(M[i].tobytes(),0)>=mind for i in range(len(types))])
    cur=(f*ok).sum()/tot; best=(cur,m.copy())
    for it in range(iters):
        T=0.02*(1-it/iters)+1e-4
        s=r.randrange(S); a_old=m[s]; a_new=r.randrange(A)
        if a_new==a_old: continue
        rows=rows_with[s]
        M[rows,a_old]-=C[rows,s]; M[rows,a_new]+=C[rows,s]
        new_ok=np.array([D.get(M[i].tobytes(),0)>=mind for i in rows])
        d=(f[rows]*(new_ok.astype(float)-ok[rows])).sum()/tot
        if d>=0 or r.random()<math.exp(d/T):
            ok[rows]=new_ok; m[s]=a_new; cur+=d
            if cur>best[0]: best=(cur,m.copy())
        else:
            M[rows,a_new]-=C[rows,s]; M[rows,a_old]+=C[rows,s]
    return best
def prep(word_list, unit):
    c=Counter(word_list); types=[w for w,n in c.items() if n>=2]
    units=sorted({u for w in types for u in unit(w)}); ui={u:i for i,u in enumerate(units)}
    return [np.array([ui[u] for u in unit(w)]) for w in types],[c[w] for w in types],len(units),types,units
VW=[w for l in parse() if l['loc'] in ('P0','P1','Pb') for w in l['words']]
GR.seed(1); GW=[w for p in generate(kind='jpos',drift=0.2,copy_p=0.3) for l in p for w in l]
# Voynich with glyphs shuffled across the corpus (keeps word lengths and glyph frequencies)
allg=[u for w in VW for u in tok(w)]; random.Random(2).shuffle(allg); k=0; SW=[]
for w in VW:
    n=len(tok(w)); SW.append(tuple(allg[k:k+n])); k+=n
SW=['|'.join(x) for x in SW]
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 20000
for lang in sys.argv[1].split(','):
    words=words_of_lang(lang); alpha=sorted(set(''.join(words)))
    cut=int(len(words)*0.8); D=build_dict(words[:cut],alpha)
    # positive control: held-out words, anagrammed + homophonic substitution into 37 symbols
    held=words[cut:cut+len(VW)]; rr=random.Random(5)
    syms=list(range(37)); code={a:[syms[i%37]] for i,a in enumerate(alpha)}
    extra=37-len(alpha)
    for i in range(max(0,extra)): code[alpha[i%len(alpha)]].append(len(alpha)+i)
    ctrl=[]
    for w in held:
        g=[rr.choice(code[ch]) for ch in w]; rr.shuffle(g); ctrl.append('|'.join(map(str,g)))
    res={}
    for name,wl,unit in [('CONTROL anagram+subst',ctrl,lambda w:w.split('|')),('VOYNICH',VW,tok),('Voynich glyph-shuffled',SW,lambda w:w.split('|')),('GENERATOR',GW,tok)]:
        types,fr,S,tl,units=prep(wl,unit)
        sc,m=anagram_solve(types,fr,S,alpha,D,iters=ITERS,seed=1)
        res[name]=sc
        ex=''
        if name=='VOYNICH':
            mp={u:alpha[m[i]] for i,u in enumerate(units)}
            ex=' e.g. '+', '.join(f"{w}->{''.join(sorted(mp[u] for u in tok(w)))}" for w in ['daiin','chedy','ol','shedy','qokeedy','chol'] if w in tl)
        print(f'{lang:14s} {name:24s} token coverage {100*sc:5.1f}%{ex}',flush=True)
