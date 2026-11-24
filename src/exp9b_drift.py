from exp9_linekey import *
import numpy as np
P,u=pages_voy()
def frac(ws,a,b):
    us=[x for w in ws for x in u(w)]; na=us.count(a); nb=us.count(b)
    return na/(na+nb) if na+nb>=3 else None
for a,b in [('k','t'),('ch','sh')]:
    out=[]
    for d in range(1,9):
        xs=[];ys=[]
        for pg in P:
            f=[frac(ws,a,b) for ws in pg]
            for i in range(len(f)-d):
                if f[i] is not None and f[i+d] is not None: xs.append(f[i]); ys.append(f[i+d])
        out.append(f'd{d}:{np.corrcoef(xs,ys)[0,1]:.3f}')
    print(a,b,' '.join(out))
# periodicity with drift-preserving null: rotate line order cyclically within page
def rot(pages):
    o=[]
    for pg in pages:
        k=R.randrange(len(pg)); o.append(pg[k:]+pg[:k])
    return o
for k in [2,3,4,5]:
    p=period(P,u,k); pn=[period(rot(P),u,k) for _ in range(40)]
    print(f'period {k}: MI={p:.4f} rot-null={np.mean(pn):.4f}±{np.std(pn):.4f} z={(p-np.mean(pn))/np.std(pn):.1f}')
