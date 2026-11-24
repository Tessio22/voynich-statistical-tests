import numpy as np, re, random, math
from common import *

HEB={c:a for c,a in zip('אבגדהוזחטיכלמנסעפצקרשת','abgdhwzxtyklmnsepcqrjv')}
HEB.update({'ך':HEB['כ'],'ם':HEB['מ'],'ן':HEB['נ'],'ף':HEB['פ'],'ץ':HEB['צ']})
def hebrew_words(name='hebrew.txt'):
    import unicodedata as ud
    t=open(D+name,encoding='utf-8').read()
    t=''.join(ch for ch in ud.normalize('NFD',t) if not ud.combining(ch))
    ws=re.findall(r'[\u05D0-\u05EA]+',t)
    return [''.join(HEB.get(c,'') for c in w) for w in ws]
ARAB_NORM={'آ':'ا','أ':'ا','إ':'ا','ٱ':'ا','ى':'ي','ی':'ي','ك':'ک','ة':'ه','ؤ':'و','ئ':'ي'}
def arabic_words(name):
    import unicodedata as ud
    t=open(D+name,encoding='utf-8').read()
    t=''.join(ARAB_NORM.get(ch,ch) for ch in t)
    ws=re.findall(r'[\u0621-\u064A\u067E\u0686\u0698\u06A9\u06AF]+',t)
    letters=sorted(set(''.join(ws)))
    sym='abcdefghijklmnopqrstuvwxyzABCDEFGHIJ'
    mp={c:sym[i] for i,c in enumerate(letters)}
    return [''.join(mp[c] for c in w) for w in ws]
WHOLE={'et':'7','est':'S','quod':'q','quia':'qa','non':'n','per':'P','pro':'O','prae':'E','enim':'eN','autem':'aT','sed':'sD','esse':'ee','cum':'c'}
def abbr_word(w):
    if w in WHOLE: return WHOLE[w]
    for pre,sig in (('con','C'),('com','C'),('per','P'),('prae','E'),('pro','O')):
        if w.startswith(pre) and len(w)>len(pre)+2: w=sig+w[len(pre):]; break
    for suf,sig in (('orum','R'),('arum','A'),('ibus','B'),('que','Q'),('tur','Z'),('us','9'),('um','U'),('ur','Z'),('is','I')):
        if w.endswith(suf) and len(w)>len(suf)+1: w=w[:-len(suf)]+sig; break
    w=re.sub(r'([aeiou])[mn](?=[bcdfgpqrstuvx])',r'\1~',w)
    return w
def abbr_words(files=('latin.txt','pliny.txt')):
    out=[]
    for f in files:
        for l in text_lines(f,'orig' if f=='latin.txt' else 'wrap'):
            for w in l:
                w=re.sub(r'[^a-z]','',w.replace('j','i').replace('v','u'))
                if w and not re.fullmatch(r'[ivxlcdm]+',w): out.append(abbr_word(w))
    return out
def lm_text(name):
    if name=='latinabbr': return ''.join(abbr_words())
    if name in ('arabic.txt','persian.txt'): return ''.join(arabic_words(name))
    if name=='hebrew.txt': return ''.join(hebrew_words())
    if name.endswith(':nv'):
        return re.sub(r'[aeiouy]','',lm_text(name[:-3]))
    return lm_text_base(name)
def lm_text_base(name):
    t=norm(load_text(name))
    ws=re.findall(r'[a-zäöüß]+',t)
    ws=[w for w in ws if not re.fullmatch(r'[ivxlcdm]+',w)]
    t=''.join(ws).replace('j','i').replace('v','u').replace('ß','ss')
    return re.sub(r'[^a-z]','',t)

class LM:
    def __init__(self, text, order=4, k=0.1):
        self.alpha=sorted(set(text)); A=len(self.alpha); self.A=A
        self.ix={c:i for i,c in enumerate(self.alpha)}
        x=np.array([self.ix[c] for c in text])
        self.order=order
        # interpolated n-gram model (Witten-Bell-ish fixed weights), full table over A^order
        P=None
        lams={1:0.05,2:0.15,3:0.3,4:0.5}
        for n in range(1,order+1):
            idx=np.zeros(len(x)-n+1,dtype=np.int64)
            for j in range(n): idx=idx*A+x[j:len(x)-n+1+j]
            cnt=np.bincount(idx,minlength=A**n).astype(float).reshape(-1,A)
            rs=cnt.sum(1,keepdims=True)
            pn=np.where(rs>0,cnt/np.maximum(rs,1),0.0)
            if P is None:
                P=(cnt+0.5)/(cnt+0.5).sum()   # unigram, shape (1,A)
                continue
            # broadcast lower-order P to this order: lower context = last n-2 symbols
            low=np.tile(P.reshape(-1,A),(A,1))   # shape (A**(n-1), A)
            w=np.where(rs>0, rs/(rs+5.0), 0.0)   # confidence in this order
            P=w*pn+(1-w)*low
        self.lp=np.log2(P).ravel()
        self.unif=np.bincount(x,minlength=A)/len(x)
        self.text=x
    lam=0.0
    def score(self, letters, valid):
        # letters: int array; valid: bool array (window entirely within a segment)
        o=self.order; A=self.A
        idx=np.zeros(len(letters)-o+1,dtype=np.int64)
        for j in range(o): idx=idx*A+letters[j:len(letters)-o+1+j]
        s=self.lp[idx]
        tot=s[valid].sum()
        if self.lam:
            p=np.bincount(letters,minlength=A)/len(letters)
            q=self.unif
            kl=np.sum(np.where(p>0,p*np.log2(np.maximum(p,1e-12)/q),0))
            tot-=self.lam*valid.sum()*kl
        return tot, valid.sum()

def make_valid(seg, o):
    # seg: int array of segment ids per position; window valid if all same segment
    n=len(seg); v=np.ones(n-o+1,bool)
    for j in range(1,o): v&= seg[j:n-o+1+j]==seg[:n-o+1]
    return v

def solve(cipher, seg, lm, iters=150000, restarts=3, seed=0, T0=2.0, init_freq=True, fixed=None, init_key=None):
    """cipher: int array of symbol ids (0..S-1); seg: segment ids. homophonic: key maps symbol->letter freely
    fixed: dict symbol->letter index that must not change; those letters are not assignable to others"""
    rng=np.random.default_rng(seed); S=cipher.max()+1; A=lm.A; o=lm.order
    valid=make_valid(seg,o); best=None
    freq=np.bincount(cipher,minlength=S).astype(float)
    fixed=fixed or {}
    free_letters=np.array([a for a in range(A) if a not in fixed.values()])
    up=lm.unif[free_letters]/lm.unif[free_letters].sum()
    for r in range(restarts):
        key=free_letters[rng.choice(len(free_letters),size=S,p=up)] if init_key is None else init_key.copy()
        for s_,a_ in fixed.items(): key[s_]=a_
        cur,n=lm.score(key[cipher],valid)
        fr=freq.copy()
        for s_ in fixed: fr[s_]=0
        p=fr/fr.sum(); movable=np.array([s_ for s_ in range(S) if s_ not in fixed])
        for it in range(iters):
            T=T0*(1-it/iters)+1e-3
            s=rng.choice(S,p=p) if rng.random()<0.7 else movable[rng.integers(len(movable))]
            if rng.random()<0.5:
                s2=movable[rng.integers(len(movable))]
                if s2==s: continue
                key[s],key[s2]=key[s2],key[s]
                new,_=lm.score(key[cipher],valid); d=new-cur
                if d>=0 or rng.random()<math.exp(d/T): cur=new
                else: key[s],key[s2]=key[s2],key[s]
                continue
            old=key[s]; key[s]=free_letters[rng.integers(len(free_letters))]
            new,_=lm.score(key[cipher],valid)
            d=new-cur
            if d>=0 or rng.random()<math.exp(d/T):
                cur=new
            else: key[s]=old
        if best is None or cur>best[0]: best=(cur,key.copy())
    return best[0]/n, best[1], n


def solve_fast(cipher, seg, lm, iters=1000000, restarts=2, seed=0, T0=1.0, fixed=None, init_key=None):
    """incremental-delta homophonic solver"""
    rng=np.random.default_rng(seed); S=int(cipher.max())+1; A=lm.A; o=lm.order; N=len(cipher)
    valid=make_valid(seg,o); W=len(valid)
    fixed=fixed or {}
    # windows touched by each symbol
    occ=[[] for _ in range(S)]
    for p,s in enumerate(cipher): occ[s].append(p)
    touch=[]
    for s in range(S):
        ws=set()
        for p in occ[s]:
            for w in range(max(0,p-o+1),min(W,p+1)):
                if valid[w]: ws.add(w)
        touch.append(np.array(sorted(ws),dtype=np.int64))
    mult=A**np.arange(o-1,-1,-1)
    free=np.array([a for a in range(A) if a not in fixed.values()])
    up=lm.unif[free]/lm.unif[free].sum()
    freq=np.array([len(x) for x in occ],float)
    for s_ in fixed: freq[s_]=0
    pick=freq/freq.sum()
    best=None
    for r in range(restarts):
        key=free[rng.choice(len(free),size=S,p=up)] if init_key is None else init_key.copy()
        for s_,a_ in fixed.items(): key[s_]=a_
        L=key[cipher]
        def wscore(ws):
            if len(ws)==0: return 0.0
            idx=np.zeros(len(ws),dtype=np.int64)
            for j in range(o): idx=idx*A+L[ws+j]
            return lm.lp[idx].sum()
        cur=wscore(np.where(valid)[0])
        syms=rng.choice(S,size=iters,p=pick)
        for it in range(iters):
            T=T0*(1-it/iters)+1e-3
            s=syms[it]; ws=touch[s]
            before=wscore(ws); old=key[s]; new=free[rng.integers(len(free))]
            if new==old: continue
            key[s]=new; L[occ[s]]=new
            after=wscore(ws); d=after-before
            if d>=0 or rng.random()<math.exp(d/T): cur+=d
            else: key[s]=old; L[occ[s]]=old
        if best is None or cur>best[0]: best=(cur,key.copy())
    return best[0]/valid.sum(), best[1], valid.sum()


def freq_init(cipher, lm):
    S=int(cipher.max())+1; f=np.bincount(cipher,minlength=S)
    lo=np.argsort(-lm.unif); key=np.zeros(S,int)
    for r,s in enumerate(np.argsort(-f)): key[s]=lo[min(r,len(lo)-1)]
    return key
