import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from exp10_generators import *
    from exp11_wordpairs import wp
MAP={'t':'k','p':'k','f':'k','sh':'ch','cth':'ch','ckh':'ch','cph':'ch','cfh':'ch','s':'d','in':'N','iin':'N','iiin':'N','ir':'N','iir':'N','r':'N','l':'N','ee':'e','eee':'e'}
def dnorm(w, drop_q=True):
    g=[MAP.get(x,x) for x in tok(w)]
    if drop_q and g and g[0]=='q': g=g[1:]
    o=[]
    for x in g:
        if x=='e' and o and o[-1]=='e': continue
        o.append(x)
    return ''.join(o) or w
def run(pages,name):
    e0=wp(pages,unit=list)[2]; P=[[[dnorm(w) for w in l] for l in p] for p in pages]
    e1=wp(P,unit=list)[2]; t0=len({w for p in pages for l in p for w in l}); t1=len({w for p in P for l in p for w in l})
    print(f'{name:22s} types {t0}->{t1}   word-pair excess {e0:.4f} -> {e1:.4f}')
run(VPAGES,'VOYNICH')
for s in [1,2,3]:
    R.seed(s); run(generate(kind='jpos',drift=0.2,copy_p=0.3),f'generator s{s}')
