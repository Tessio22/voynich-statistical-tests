import io,contextlib,sys,numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import exp22_anagram as E
from exp10_generators import generate, R as GR
for lang in sys.argv[1].split(','):
    words=E.words_of_lang(lang); alpha=sorted(set(''.join(words))); cut=int(len(words)*0.8); D=E.build_dict(words[:cut],alpha)
    v=[];g=[]
    for s in range(4):
        t,fr,S,_,_=E.prep(E.VW,E.tok); v.append(E.anagram_solve(t,fr,S,alpha,D,iters=30000,seed=10+s)[0])
        GR.seed(20+s); GW=[w for p in generate(kind='jpos',drift=0.2,copy_p=0.3) for l in p for w in l]
        t,fr,S,_,_=E.prep(GW,E.tok); g.append(E.anagram_solve(t,fr,S,alpha,D,iters=30000,seed=30+s)[0])
    print(f'{lang:12s} Voynich {100*np.mean(v):.1f}±{100*np.std(v):.1f}%   generator {100*np.mean(g):.1f}±{100*np.std(g):.1f}%',flush=True)
