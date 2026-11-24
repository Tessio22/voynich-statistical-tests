import numpy as np
def exchange_cluster(seq, seg, K, iters=15, seed=0, init=None):
    """Brown-style exchange clustering of symbols maximizing class-bigram mutual information."""
    rng=np.random.default_rng(seed)
    S=int(seq.max())+1
    C=np.zeros((S,S))
    ok=seg[1:]==seg[:-1]
    np.add.at(C,(seq[:-1][ok],seq[1:][ok]),1)
    freq=np.bincount(seq,minlength=S)
    z=rng.integers(K,size=S) if init is None else init.copy()
    def xlogx(a):
        a=np.asarray(a,float); return np.where(a>0,a*np.log(np.maximum(a,1e-300)),0).sum()
    def obj(M): return xlogx(M)-xlogx(M.sum(1))-xlogx(M.sum(0))
    Z=np.zeros((S,K)); Z[np.arange(S),z]=1
    M=Z.T@C@Z
    order=np.argsort(-freq)
    for it in range(iters):
        moved=0
        for w in order:
            k0=z[w]
            r=C[w,:]@Z; c=Z.T@C[:,w]; cww=C[w,w]
            # remove w from k0
            M[k0,:]-=r; M[:,k0]-=c; M[k0,k0]+=cww
            Z[w,k0]=0
            r=C[w,:]@Z; c=Z.T@C[:,w]
            best=None
            for k in range(K):
                Mk=M.copy(); Mk[k,:]+=r; Mk[:,k]+=c; Mk[k,k]+=cww
                v=obj(Mk)
                if best is None or v>best[0]: best=(v,k)
            k=best[1]; M[k,:]+=r; M[:,k]+=c; M[k,k]+=cww; Z[w,k]=1
            if k!=k0: moved+=1
            z[w]=k
        if moved==0: break
    return z, obj(M)
