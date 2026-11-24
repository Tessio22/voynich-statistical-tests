from common import *; from parse import parse
import random, numpy as np
from collections import defaultdict
MATCH="""f1v f102r1 same
f4v f101r sim
f10v f89v2 sim
f13v f100v sim
f18v f102r2 same
f19r f102v1 same
f23r f102r2 same
f24r f89v2 sim
f30r f89r1 sim
f30v f101r sim
f32v f102r2 same
f33r f89r1 sim
f34v f99v sim
f35v f89r2 sim
f36r f102v2 sim
f37r f101v sim
f37v f102r1 same
f44r f99v sim
f47v f102r2 same
f48r f89v2 same
f48v f89v2 same
f49r f101r sim
f51r f99r sim
f53r f102r2 sim
f55v f101r sim"""
M=[l.split() for l in MATCH.splitlines()]
L=parse()
herb=defaultdict(list); first=defaultdict(list); labels=defaultdict(list); htype={}
for l in L:
    if l['loc'] in ('P0','P1','Pb') and l['words']:
        herb[l['folio']]+=l['words']; htype[l['folio']]=l.get('I')
        if l['folio'] not in first: first[l['folio']]=l['words'][:4]
    if l['loc'].startswith('L'):
        labels[l['folio']]+=l['words']
herbal=[f for f in herb if htype.get(f)=='H']
print('pharma pages with labels:',{p:len(labels[p]) for p in set(x[1] for x in M)})
def lev(a,b):
    d=list(range(len(b)+1))
    for i in range(1,len(a)+1):
        p=d[0]; d[0]=i
        for j in range(1,len(b)+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,p+(a[i-1]!=b[j-1])); p=t
    return d[-1]
def sim(a,b):
    a,b=tok(a),tok(b); return 1-lev(a,b)/max(len(a),len(b))
def score(H,P,mode):
    ws=first[H] if mode=='first' else herb[H]
    labs=[w for w in labels[P] if len(tok(w))>=3]
    if not labs or not ws: return None
    return max(sim(a,b) for a in labs for b in ws), any(a==b for a in labs for b in ws)
R=random.Random(1)
for subset in ['same','all']:
    pairs=[(h,p) for h,p,t in M if subset=='all' or t=='same']
    for mode in ['first','page']:
        obs=[score(h,p,mode) for h,p in pairs]; obs=[o for o in obs if o]
        o_s=np.mean([x[0] for x in obs]); o_e=np.mean([x[1] for x in obs])
        nulls_s=[];nulls_e=[]
        for _ in range(300):
            r=[score(R.choice([f for f in herbal if f!=h]),p,mode) for h,p in pairs]; r=[x for x in r if x]
            nulls_s.append(np.mean([x[0] for x in r])); nulls_e.append(np.mean([x[1] for x in r]))
        ps=(1+sum(n>=o_s for n in nulls_s))/301; pe=(1+sum(n>=o_e for n in nulls_e))/301
        print(f'{subset:4s} {mode:5s} n={len(obs):2d}  maxSim {o_s:.3f} vs null {np.mean(nulls_s):.3f} p={ps:.3f} | exact-match {o_e:.2f} vs {np.mean(nulls_e):.2f} p={pe:.3f}')
# show the same-plant pairs
for h,p,t in M:
    if t!='same': continue
    labs=[w for w in labels[p] if len(tok(w))>=3]
    best=max(((sim(a,b),a,b) for a in labs for b in herb[h]),default=None)
    print(f'  {h:5s}<->{p:7s} first words: {" ".join(first[h][:2]):18s} labels: {" ".join(labels[p][:6]):45s} best: {best}')
