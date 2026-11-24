import os
# Hypothesis: each Voynich word (or glyph unit) = one plaintext letter (homophonic, Naibbe-like).
# Pipeline: exchange clustering of symbols -> K classes -> substitution solve with n-gram LM + unigram KL penalty.
import sys, json
from solver import *; from parse import parse; from cluster import exchange_cluster
MODE=sys.argv[1]      # 'word' or 'glyph'
K=int(sys.argv[2]); LAM=float(sys.argv[3])
L=[l for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words']]
_T=os.environ.get('TRIM')
if _T:
    from trim import trim
    for l in L: l['words']=[trim(w,_T) for w in l['words']]
rng=np.random.default_rng(11)
if MODE=='word':
    c=Counter(w for l in L for w in l['words'])
    keep=sorted(w for w,n in c.items() if n>=5); sid={w:i for i,w in enumerate(keep)}
    cipher=[];seg=[];s=0
    for l in L:
        s+=1
        for w in l['words']:
            if w in sid: cipher.append(sid[w]); seg.append(s)
            else: s+=1
    names=keep
else:
    units=sorted({u for l in L for w in l['words'] for u in tok(w)}); uid={u:i for i,u in enumerate(units)}
    cipher=[];seg=[]
    for si,l in enumerate(L):
        for w in l['words']:
            for u in tok(w): cipher.append(uid[u]); seg.append(si)
    names=units
cipher=np.array(cipher); seg=np.array(seg)
print(MODE,'tokens',len(cipher),'symbols',cipher.max()+1,flush=True)

def run(x, segx, lm, tag):
    z,_=exchange_cluster(x,segx,K,seed=1)
    cl=z[x]; best=None
    for sd in range(4):
        r=solve(cl,segx,lm,iters=60000,restarts=1,seed=sd,T0=2.0,init_key=(freq_init(cl,lm) if sd%2==0 else None))
        if best is None or r[0]>best[0]: best=r
    sc,key,_=best
    v=make_valid(segx,4); lam=lm.lam; lm.lam=0; raw=lm.score(key[cl],v)[0]/v.sum(); lm.lam=lam
    return sc, raw, z, key

res={}
for lang in os.environ.get('LANGS','latin.txt,italian.txt,german.txt').split(','):
    full=lm_text(lang); cut=int(len(full)*0.8); lm=LM(full[:cut]); lm.lam=LAM; held=''.join(ch for ch in full[cut:] if ch in lm.ix)
    sc,raw,z,key=run(cipher,seg,lm,'voy')
    dec=''.join(lm.alpha[key[z[s_]]] for s_ in cipher[:300])
    # show segmentation by line for first lines
    lines=[]; cur=seg[0]; buf=''
    for s_,g in zip(cipher[:400],seg[:400]):
        if g!=cur: lines.append(buf); buf=''; cur=g
        buf+=lm.alpha[key[z[s_]]]
    sh=cipher.copy(); rng.shuffle(sh)
    sc2,raw2,_,_=run(sh,seg,lm,'shuf')
    # synthetic control: same #symbols, same segment structure
    S=int(cipher.max())+1; f=Counter(held); tot=sum(f.values())
    alloc={a:max(1,round(S*f[a]/tot)) for a in lm.alpha}; syms={};kk=0
    for a in lm.alpha: syms[a]=list(range(kk,kk+alloc[a])); kk+=alloc[a]
    x=[];truth={}
    for ch in held[:len(cipher)]:
        hs=syms[ch]; w=np.array([1/(i+1) for i in range(len(hs))]); w/=w.sum(); sy=hs[rng.choice(len(hs),p=w)]; x.append(sy); truth[sy]=ch
    x=np.array(x); segx=seg[:len(x)]
    sc3,raw3,z3,key3=run(x,segx,lm,'syn')
    acc=float(np.mean([lm.alpha[key3[z3[s_]]]==truth[s_] for s_ in x]))
    px=np.array([lm.ix[truth[s_]] for s_ in x]); v=make_valid(segx,4); lam=lm.lam; lm.lam=0; ref=lm.score(px,v)[0]/v.sum(); lm.lam=lam
    r=dict(voy_raw=round(raw,3),shuf_raw=round(raw2,3),syn_raw=round(raw3,3),syn_acc=round(acc,3),plain_ref=round(ref,3),lines=lines[:12])
    res[lang]=r; print(lang,json.dumps(r,ensure_ascii=False),flush=True)
json.dump(res,open(RESULTS+f'res_exp8_{MODE}_{K}.json','w'),indent=1,ensure_ascii=False)
