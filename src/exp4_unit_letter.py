import os
import sys, json
from solver import *; from parse import parse
ITERS=int(sys.argv[1]) if len(sys.argv)>1 else 100000
MERGES=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [0,20]

def lm_text_sp(name):
    t=norm(load_text(name)).replace('j','i').replace('v','u').replace('ß','ss')
    t=re.sub(r'[^a-z]+',' ',t); return re.sub(' +',' ',t)

L=[l for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words'] and l['clean']]
base=[[tuple(tok(w)) for w in l['words']] for l in L]

def bpe(lines,k):
    ws=[w for l in lines for w in l]
    merges=[]
    for _ in range(k):
        pc=Counter()
        for w in ws:
            for x,y in zip(w,w[1:]): pc[(x,y)]+=1
        a,b=pc.most_common(1)[0][0]; merges.append(a+b)
        def m(w):
            o=[];i=0
            while i<len(w):
                if i+1<len(w) and w[i]==a and w[i+1]==b: o.append(a+b); i+=2
                else: o.append(w[i]); i+=1
            return tuple(o)
        ws=[m(w) for w in ws]
        lines=[[m(w) for w in l] for l in lines]
    return lines

res={}
for k in MERGES:
    lines=bpe(base,k)
    units=sorted({u for l in lines for w in l for u in w}); uid={u:i+1 for i,u in enumerate(units)}  # 0 = space
    cipher=[];seg=[]
    for si,l in enumerate(lines):
        for wi,w in enumerate(l):
            if wi: cipher.append(0); seg.append(si)
            for u in w: cipher.append(uid[u]); seg.append(si)
    cipher=np.array(cipher); seg=np.array(seg)
    rng=np.random.default_rng(7); shuf=cipher.copy(); nz=np.where(shuf>0)[0]; vals=shuf[nz]; rng.shuffle(vals); shuf[nz]=vals
    for lang in os.environ.get('LANGS','latin.txt,italian.txt,german.txt').split(','):
        full=lm_text_sp(lang); cut=int(len(full)*0.8); lm=LM(full[:cut]); held=full[cut:]
        sp=lm.ix[' ']
        sc,key,n=solve(cipher,seg,lm,iters=ITERS,restarts=2,seed=1,fixed={0:sp})
        sc2,_,_=solve(shuf,seg,lm,iters=ITERS,restarts=2,seed=2,fixed={0:sp})
        # synthetic: encrypt held-out text homophonically with len(units) symbols, spaces kept
        held=''.join(c for c in held if c in lm.ix); letters=[c for c in sorted(set(held)) if c!=' ']; f=Counter(c for c in held if c!=' '); tot=sum(f.values())
        alloc={a:max(1,round(len(units)*f[a]/tot)) for a in letters}; syms={};kk=1
        for a in letters: syms[a]=list(range(kk,kk+alloc[a])); kk+=alloc[a]
        x=[];truth={};segx=[];s_=0;cnt=0
        for ch in held:
            if len(x)>=len(cipher): break
            if ch==' ':
                x.append(0); cnt+=1
                if cnt%9==0: s_+=1
            else:
                hs=syms[ch]; w=np.array([1/(i+1) for i in range(len(hs))]); w/=w.sum()
                sy=hs[rng.choice(len(hs),p=w)]; x.append(sy); truth[sy]=ch
            segx.append(s_)
        x=np.array(x); segx=np.array(segx)
        sc3,key3,_=solve(x,segx,lm,iters=ITERS,restarts=2,seed=3,fixed={0:sp})
        nzx=x[x>0]; acc=float(np.mean([lm.alpha[key3[s]]==truth[s] for s in nzx]))
        hx=np.array([lm.ix[c] for c in held[:len(cipher)] if c in lm.ix])[:len(segx)]; vv=make_valid(segx[:len(hx)],4); ref=lm.score(hx,vv)[0]/vv.sum()
        dec=''.join(lm.alpha[key[s]] for s in cipher[:300])
        r=dict(merges=k,units=len(units),voynich=round(sc,3),shuffled=round(sc2,3),synthetic=round(sc3,3),synth_acc=round(acc,3),plain_ref=round(ref,3),sample=dec)
        res[f'{k}_{lang}']=r; print(r,flush=True)
json.dump(res,open(RESULTS+f'res_exp4.json','w'),indent=1)
