# Collapse glyph variants that behave interchangeably (ch/sh, k/t/p/f, e/ee/eee, q- prefix, iin/in, ...)
# and check whether the "normalized" text becomes more language-like at the word level.
import io, contextlib, re
with contextlib.redirect_stdout(io.StringIO()):
    from exp10_generators import *
    from exp11_wordpairs import wp
    from exp15_neighbors import neigh_form
import numpy as np
LEVELS={
 0: [],
 1: [('sh','ch')],
 2: [('sh','ch'),('t','k'),('p','k'),('f','k'),('cth','ckh'),('cph','ckh'),('cfh','ckh')],
 3: [('sh','ch'),('t','k'),('p','k'),('f','k'),('cth','ckh'),('cph','ckh'),('cfh','ckh'),('ee','e'),('eee','e'),('iiin','iin'),('in','iin'),('iir','ir')],
 4: 'L3+drop q',
}
def norm_word(w, lvl):
    g=tok(w)
    subs=dict(LEVELS[3] if lvl==4 else LEVELS[lvl])
    g=[subs.get(x,x) for x in g]
    # collapse runs of e after substitution
    out=[]
    for x in g:
        if x=='e' and out and out[-1]=='e' and lvl>=3: continue
        out.append(x)
    if lvl==4 and out and out[0]=='q': out=out[1:]
    return ''.join(out) if out else w
def stats(lines):
    toks=[w for l in lines for w in l]; c=Counter(toks)
    seq=[]
    for w in toks: seq+=list(w)+['_']
    T=toks[:30000]
    return dict(types=len(c), ttr=len(set(T))/len(T), top10share=sum(n for _,n in c.most_common(10))/len(toks))
VL=[l for p in VPAGES for l in p]
print('lvl  types   ttr   top10%  wordpair-excess  nbr-similar% (rand%)')
for lvl in [0,1,2,3,4]:
    pages=[[[norm_word(w,lvl) for w in l] for l in p] for p in VPAGES]
    lines=[l for p in pages for l in p]
    s=stats(lines); o,n,e=wp(pages, unit=list)
    nf=neigh_form(lines,list)
    print(f'{lvl:3d} {s["types"]:6d} {s["ttr"]:.3f} {100*s["top10share"]:6.1f}  {e:15.4f}  {100*nf[2]:6.1f} ({100*nf[3]:.1f})',flush=True)
print('reference word-pair excess: Latin prose 0.084, Italian 0.224, Latin verse 0.140, generator 0.004-0.014')
for n in ['latin.txt','italian.txt']:
    L=text_lines(n,'wrap')[:4000]; s=stats(L); print(n, f'types {s["types"]} ttr {s["ttr"]:.3f} top10 {100*s["top10share"]:.1f}%')
