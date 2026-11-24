# Linear-B-like syllabary hypothesis: Voynich units (BPE 60) = reduced syllables (onset+vowel, codas dropped).
import sys, re, numpy as np, io, contextlib, random
from common import *; from parse import parse
from solver import LM, solve, freq_init, make_valid
from exp23_syll import bpe
def red_syll(w):
    # reduced syllables: (first consonant of onset or '') + first vowel ; codas dropped
    out=[]
    for m in re.finditer(r'([^aeiouy]*)([aeiouy]+)',w):
        on=m.group(1)[:1]; out.append(on+m.group(2)[0])
    return out or [w[:1]]
def lang_stream(name):
    ws=[w for l in text_lines(name,'wrap') for w in l]
    return [red_syll(w.replace('j','i').replace('v','u')) for w in ws]
LANG=sys.argv[1] if len(sys.argv)>1 else 'italian.txt'; IT=int(sys.argv[2]) if len(sys.argv)>2 else 40000
SW=lang_stream(LANG)
inv=Counter(s for w in SW for s in w); syl=[s for s,_ in inv.most_common()]
CH={s:chr(0x100+i) for i,s in enumerate(syl)}; CH['_']=' '
def to_text(words): return ''.join(''.join(CH[s] for s in w)+' ' for w in words)
cut=int(len(SW)*0.8)
lm=LM(to_text(SW[:cut]),order=3); lm.lam=10.0
print(LANG,'reduced-syllable inventory',len(syl),'LM alphabet',lm.A,flush=True)
sp=lm.ix[' ']
def stream(words_units, uid):
    x=[];seg=[];s=0
    for i,w in enumerate(words_units):
        if i%9==0: s+=1
        for u in w: x.append(uid[u]); seg.append(s)
        x.append(0); seg.append(s)
    return np.array(x),np.array(seg)
# Voynich
VW=[tok(w) for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words'] for w in l['words']]
VU=bpe(VW,60); units=sorted({u for w in VU for u in w}); uid={u:i+1 for i,u in enumerate(units)}
xv,segv=stream(VU,uid)
def run(x,seg,tag):
    best=None
    for sd in range(3):
        r=solve(x,seg,lm,iters=IT,restarts=1,seed=sd,T0=2.0,fixed={0:sp},init_key=None)
        if best is None or r[0]>best[0]: best=r
    sc,key,_=best; v=make_valid(seg,3); lam=lm.lam; lm.lam=0; raw=lm.score(key[x],v)[0]/v.sum(); lm.lam=lam
    return raw,key
raw_v,key_v=run(xv,segv,'voy')
# shuffled Voynich units (within corpus, keep spaces)
xs=xv.copy(); nz=np.where(xs>0)[0]; vals=xs[nz]; np.random.default_rng(3).shuffle(vals); xs[nz]=vals
raw_s,_=run(xs,segv,'shuf')
# positive control: held-out reduced syllables, homophonic into len(units) symbols
held=SW[cut:cut+len(VU)]; f=Counter(s for w in held for s in w); S=len(units); syms={}; k=1; rr=random.Random(1)
for s,n in f.most_common():
    nv=2 if n>300 and k+2<=S else 1
    syms[s]=[1+((k+i-1)%S) for i in range(nv)]; k+=nv
xc=[];segc=[];truth=[];s_=0
for i,w in enumerate(held):
    if i%9==0: s_+=1
    for sy in w: c=rr.choice(syms[sy]); xc.append(c); segc.append(s_); truth.append(sy)
    xc.append(0); segc.append(s_); truth.append('_')
xc=np.array(xc); segc=np.array(segc)
raw_c,key_c=run(xc,segc,'ctrl')
inv_ch={v:k for k,v in CH.items()}
acc=np.mean([inv_ch.get(lm.alpha[key_c[c]],'?')==t for c,t in zip(xc,truth) if c>0])
ref_x=np.array([lm.ix[CH[t]] for t in truth if CH[t] in lm.ix]); v=make_valid(segc[:len(ref_x)],3); lam=lm.lam; lm.lam=0; ref=lm.score(ref_x,v)[0]/v.sum(); lm.lam=lam
print(f'CONTROL acc {acc:.3f} score {raw_c:.3f} | plain {ref:.3f} | VOYNICH {raw_v:.3f} | shuffled {raw_s:.3f}',flush=True)
dec=[]
for w in VU[:40]: dec.append(''.join(inv_ch.get(lm.alpha[key_v[uid[u]]],'?') for u in w))
print('Voynich decoded (first 40 words):',' '.join(dec))
