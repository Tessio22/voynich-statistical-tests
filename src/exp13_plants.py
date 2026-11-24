# Crib test: plant names vs first words of herbal pages.
# Search a glyph->letter mapping that makes page-initial words resemble the plant names of THEIR pages,
# and compare with the same search when names are shuffled among pages (null).
import sys, json, random
from common import *; from parse import parse

PLANTS={
 'f9v':  ['viola','violatricolor','herbatrinitatis','freisam','stiefmutter','violetta','macesky','iacea','viole'],
 'f65v': ['camomilla','chamomilla','anthemis','kamille','matricaria','hermanek','camamilla','camomille'],
 'f4r':  ['hypericum','ypericon','perforata','iohanniskraut','hartheu','iperico','trezalka','fugademonum'],
 'f29r': ['lappa','bardana','arctium','klette','lopuch','personata','bardane'],
 'f1v':  ['belladonna','solanum','atropa','tollkirsche','morella','strignum','mandragora'],
 'f25v': ['plantago','wegerich','piantaggine','iitrocel','arnoglossa','plantain','lancea'],
 'f31r': ['iusquiamus','hyoscyamus','iusquiamo','bilsenkraut','blin','apollinaris','iusquiame','alterchum'],
 'f32r': ['mentha','menta','minze','mata','balsamita','nepita','mentastrum'],
 'f51v': ['buglossa','linguabouis','borago','ochsenzunge','symphytum','consolida','lycopsis'],
 'f56r': ['borago','borrago','borretsch','borragine','brutnak','bourrache','buglossa'],
 'f2r':  ['centaurea','cyanus','kornblume','fiordaliso','chrpa','centaurium','kantairon','blaver'],
 'f3v':  ['helleborus','elleborus','nieswurz','elleboro','cemerice','ellebore','kaur'],
 'f13r': ['tussilago','ungulacaballina','huflattich','podbel','farfara','bechion','farfaro'],
 'f53r': ['enula','inula','helenium','alant','enulacampana','elecampane','iman'],
 'f44r': ['ueratrum','elleborusalbus','germer','ueratro','helleborusalbus'],
 'f90r1':['papauer','papauero','mohn','mak','papaver','oppium','codia'],
 'f36v': ['cannabis','canapa','hanf','konopi','chanvre','cannabus'],
}
def clean(n): return n.lower().replace('j','i').replace('v','u').replace('w','u')
PLANTS={k:[clean(x) for x in v] for k,v in PLANTS.items()}
ALPHA='abcdefghiklmnopqrstuxyz'
GAL={'k','t','p','f','cth','ckh','cph','cfh'}

def page_words():
    out={}
    for l in parse():
        if l['folio'] in PLANTS and l['loc'] in ('P0','P1','Pb') and l['words'] and l['folio'] not in out:
            out[l['folio']]=l['words'][:3]
    return out

def lev(a,b):
    d=list(range(len(b)+1))
    for i in range(1,len(a)+1):
        p=d[0]; d[0]=i
        for j in range(1,len(b)+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,p+(a[i-1]!=b[j-1])); p=t
    return d[-1]
def sim(dec, name):
    if not dec: return 0.0
    pre=name[:max(len(dec),3)]
    return 1-lev(dec,pre)/max(len(dec),len(pre))

def cands(words):
    c=[tok(words[0])]
    g=tok(words[0])
    if g and g[0] in GAL: c.append(g[1:])
    if len(words)>1: c.append(tok(words[1]))
    return [x for x in c if x]

def score(mapping, items):
    tot=0
    for cw, names in items:
        best=0
        for g in cw:
            dec=''.join(mapping.get(u,'') for u in g)
            for n in names:
                s=sim(dec,n)
                if s>best: best=s
        tot+=best
    return tot/len(items)

def optimize(items, units, iters=6000, restarts=4, seed=0):
    r=random.Random(seed); best=None
    opts=list(ALPHA)+['']
    for _ in range(restarts):
        m={u:r.choice(ALPHA) for u in units}; cur=score(m,items)
        for it in range(iters):
            u=r.choice(units); old=m[u]; m[u]=r.choice(opts)
            s=score(m,items); T=0.02*(1-it/iters)+1e-4
            if s>=cur or r.random()<math.exp((s-cur)/T): cur=s
            else: m[u]=old
        if best is None or cur>best[0]: best=(cur,dict(m))
    return best

def run(items, nperm, tag, seed=0):
    units=sorted({u for cw,_ in items for g in cw for u in g})
    obs,m=optimize(items,units,seed=seed)
    r=random.Random(100+seed); nulls=[]
    names=[n for _,n in items]
    for k in range(nperm):
        while True:
            perm=names[:]; r.shuffle(perm)
            if all(a is not b for a,b in zip(perm,names)): break
        nulls.append(optimize([(cw,p) for (cw,_),p in zip(items,perm)],units,seed=seed+k+1)[0])
    p=(1+sum(x>=obs for x in nulls))/(1+len(nulls))
    print(f'{tag}: observed {obs:.3f}  null {np.mean(nulls):.3f}±{np.std(nulls):.3f}  max null {max(nulls):.3f}  p={p:.3f}',flush=True)
    return obs,m,nulls

if __name__=='__main__':
    import numpy as np
    NP=int(sys.argv[1]) if len(sys.argv)>1 else 20
    pw=page_words()
    items=[(cands(pw[f]),PLANTS[f]) for f in PLANTS if f in pw]
    print('pages used',len(items),'first words:',{f:pw[f][0] for f in PLANTS if f in pw})
    # positive control: encode the true names with a random glyph code (1 letter -> 1 Voynich-like unit), first word = name prefix of 4-7 letters
    rr=random.Random(7); fake_units=['o','ch','e','d','y','a','k','l','r','sh','t','q','iin','s','p','m','f','i','cth','ckh','in','ir','n']
    code={c:fake_units[i%len(fake_units)] for i,c in enumerate(ALPHA)}
    def enc(name):
        n=name[:rr.randint(4,7)]; g=[code[c] for c in n if c in code]
        if rr.random()<0.5: g=['k']+g   # decorative gallows sometimes
        return g
    ctrl=[([enc(rr.choice(names))] + [tok(rr.choice(['daiin','chol','shey']))], names) for _,names in items]
    run(ctrl,NP,'POSITIVE CONTROL (synthetic encoded names)')
    obs,m,nulls=run(items,NP,'VOYNICH herbal pages')
    for (cw,names),f in zip(items,[f for f in PLANTS if f in pw]):
        decs=[''.join(m.get(u,'') for u in g) for g in cw]
        best=max(((sim(d,n),d,n) for d in decs for n in names))
        print(f'  {f:6s} {pw[f][0]:12s} -> {best[1]:10s} ~ {best[2]:16s} sim={best[0]:.2f}')
