# If zodiac nymph labels encode day numbers/names, label at position k should resemble label at position k
# in other signs (allowing a cyclic shift / reversal, since start point on the circle is unknown).
import random, numpy as np
from common import *; from parse import parse
from collections import defaultdict
lab=defaultdict(list)
for l in parse():
    if l.get('I')=='Z' and l['loc']=='Lz' and l['words']: lab[l['folio']].append(''.join(l['words']))
signs={}
for f,v in lab.items():
    if len(v)>=28: signs[f]=v[:30]
if 'f70v1' in lab and 'f71r' in lab: signs['Aries(70v1+71r)']=lab['f70v1']+lab['f71r']
if 'f71v' in lab and 'f72r1' in lab: signs['Taurus(71v+72r1)']=lab['f71v']+lab['f72r1']
print({k:len(v) for k,v in signs.items()})
def lev(a,b):
    d=list(range(len(b)+1))
    for i in range(1,len(a)+1):
        p=d[0]; d[0]=i
        for j in range(1,len(b)+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,p+(a[i-1]!=b[j-1])); p=t
    return d[-1]
def sim(a,b):
    a,b=tok(a),tok(b); return 1-lev(a,b)/max(len(a),len(b))
names=list(signs); S={k:[x for x in v] for k,v in signs.items()}
def best_align(A,B):
    n=min(len(A),len(B)); A=A[:n]; B=B[:n]; best=0
    M=np.array([[sim(a,b) for b in B] for a in A])
    for rev in [False,True]:
        idx=list(range(n))[::-1] if rev else list(range(n))
        for s in range(n):
            v=np.mean([M[i,idx[(i+s)%n]] for i in range(n)])
            best=max(best,v)
    return best, M.mean()
obs=[];base=[]
for i in range(len(names)):
    for j in range(i+1,len(names)):
        b,m=best_align(S[names[i]],S[names[j]]); obs.append(b); base.append(m)
obs=np.mean(obs)
R=random.Random(1); null=[]
for _ in range(100):
    Sh={k:R.sample(v,len(v)) for k,v in S.items()}
    vals=[best_align(Sh[names[i]],Sh[names[j]])[0] for i in range(len(names)) for j in range(i+1,len(names))]
    null.append(np.mean(vals))
null=np.array(null)
print(f'best-aligned positional similarity {obs:.4f}   shuffled-within-sign null {null.mean():.4f}±{null.std():.4f}  p={(1+np.sum(null>=obs))/101:.3f}')
print('average similarity of any two labels:',round(float(np.mean(base)),3))
