# Word-level dependency beyond junction glyphs: MI(word_i ; word_i+1 | junction pair) vs page-shuffle null
from exp10_generators import *
def wp(pages, unit=tok, top=150):
    lines=[l for p in pages for l in p]
    c=Counter(w for l in lines for w in l); keep={w for w,_ in c.most_common(top)}
    m=lambda w: w if w in keep else '#'
    trip=[]
    for l in lines:
        for i in range(len(l)-1):
            a,b=l[i],l[i+1]
            trip.append(((unit(a)[-1],unit(b)[0]),m(a),m(b)))
    def cmi(tr):
        by=defaultdict(list)
        for j,a,b in tr: by[j].append((a,b))
        n=len(tr); return sum(len(v)/n*mi(v) for v in by.values() if len(v)>1)
    obs=cmi(trip); nulls=[]
    for _ in range(10):
        # shuffle second words among pairs with same junction pair (keeps junction stats)
        by=defaultdict(list)
        for k,(j,a,b) in enumerate(trip): by[j].append(k)
        bb=[t[2] for t in trip]
        for j,ix in by.items():
            v=[bb[k] for k in ix]; R.shuffle(v)
            for k,x in zip(ix,v): bb[k]=x
        nulls.append(cmi([(t[0],t[1],x) for t,x in zip(trip,bb)]))
    return obs, np.mean(nulls), obs-np.mean(nulls)
print('Voynich          obs %.4f null %.4f excess %.4f'%wp(VPAGES))
for s in [1,2]:
    R.seed(s); G=generate(kind='jpos',drift=0.2,copy_p=0.3)
    print('Generator seed%d  obs %.4f null %.4f excess %.4f'%((s,)+wp(G)))
def tp(name,mode,size=20,lim=None):
    L=[x for x in text_lines(name,mode) if len(x)>=3]
    if lim: L=L[:lim]
    return [L[i:i+size] for i in range(0,len(L)-size,size)]
print('Latin verse      obs %.4f null %.4f excess %.4f'%wp(tp('latin.txt','orig',lim=1800),list))
print('Latin prose      obs %.4f null %.4f excess %.4f'%wp(tp('latin.txt','wrap'),list))
print('Italian prose    obs %.4f null %.4f excess %.4f'%wp(tp('italian.txt','wrap',lim=4000),list))
