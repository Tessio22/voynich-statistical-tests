import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from exp16_collapse import *
def lat_norm(w,lvl):
    if lvl==0: return w
    m={'b':'p','d':'t','g':'c','q':'c','i':'e','u':'o','m':'n','x':'s'}
    w=''.join(m.get(c,c) for c in w)
    if lvl>=2 and w.startswith('h'): w=w[1:] or w
    if lvl>=2: w=re.sub(r'(.)\1+',r'\1',w)
    return w
print('GENERATOR (drift+copy):')
for s in [1,2]:
    R.seed(s); G=generate(kind='jpos',drift=0.2,copy_p=0.3)
    for lvl in [0,4]:
        pages=[[[norm_word(w,lvl) for w in l] for l in p] for p in G]
        st=stats([l for p in pages for l in p]); e=wp(pages,unit=list)[2]
        print(f'  seed{s} lvl{lvl} types {st["types"]} ttr {st["ttr"]:.3f} wordpair-excess {e:.4f}')
print('LATIN prose with similar-size letter merging:')
L=text_lines('latin.txt','wrap'); P=[L[i:i+20] for i in range(0,len(L)-20,20)]
for lvl in [0,1,2]:
    pages=[[[lat_norm(w,lvl) for w in l] for l in p] for p in P]
    st=stats([l for p in pages for l in p]); e=wp(pages,unit=list)[2]
    print(f'  lvl{lvl} types {st["types"]} ttr {st["ttr"]:.3f} wordpair-excess {e:.4f}')
