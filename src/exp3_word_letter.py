import sys, json
from solver import *; from parse import parse
ITERS=int(sys.argv[1]) if len(sys.argv)>1 else 100000
L=[l for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words']]
c=Counter(w for l in L for w in l['words'])
keep=sorted(w for w,n in c.items() if n>=5); sid={w:i for i,w in enumerate(keep)}
cipher=[];seg=[];s=0
for l in L:
    s+=1
    for w in l['words']:
        if w in sid: cipher.append(sid[w]); seg.append(s)
        else: s+=1
cipher=np.array(cipher); seg=np.array(seg)
print('tokens',len(cipher),'symbols',len(keep),flush=True)
rng=np.random.default_rng(5)
shuf=cipher.copy(); rng.shuffle(shuf)

def synth(lm_train_text, heldout, S, n):
    # homophonic encryption of real text with S symbols, homophones ~ letter freq, zipf within letter
    A=sorted(set(lm_train_text)); f=Counter(heldout); tot=sum(f.values())
    alloc={a:max(1,round(S*f[a]/tot)) for a in A}
    syms={};k=0
    for a in A:
        syms[a]=list(range(k,k+alloc[a])); k+=alloc[a]
    x=[];truth={}
    for ch in heldout[:n]:
        hs=syms[ch]; w=np.array([1/(i+1) for i in range(len(hs))]); w/=w.sum()
        sym=hs[rng.choice(len(hs),p=w)]; x.append(sym); truth[sym]=ch
    return np.array(x), truth, k

res={}
for lang in ['latin.txt','italian.txt','german.txt']:
    full=lm_text(lang); cut=int(len(full)*0.8)
    lm=LM(full[:cut]); held=full[cut:]
    sc,key,n=solve_fast(cipher,seg,lm,iters=ITERS,restarts=2,seed=1)
    sc2,_,_=solve_fast(shuf,seg,lm,iters=ITERS,restarts=2,seed=2)
    # synthetic positive control with same #tokens (approx) and segment lengths
    x,truth,S=synth(full,held,len(keep),len(cipher))
    segx=seg[:len(x)]
    sc3,key3,_=solve_fast(x,segx,lm,iters=ITERS,restarts=2,seed=3)
    acc=np.mean([lm.alpha[key3[s_]]==truth[s_] for s_ in x])
    # held-out plain text score (upper reference)
    hx=np.array([lm.ix[ch] for ch in held[:len(cipher)] if ch in lm.ix])
    hv=make_valid(segx[:len(hx)],4)
    ref=lm.score(hx,hv)[0]/hv.sum()
    dec=''.join(lm.alpha[key[s_]] for s_ in cipher[:400])
    res[lang]=dict(voynich=round(sc,3),shuffled=round(sc2,3),synthetic=round(sc3,3),synthetic_acc=round(float(acc),3),plain_ref=round(ref,3),sample=dec)
    print(lang,res[lang],flush=True)
json.dump(res,open(RESULTS+'res_exp3.json','w'),indent=1)
