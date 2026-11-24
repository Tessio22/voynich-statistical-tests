# Syllable hypothesis: profile of Voynich units (after k BPE merges) vs syllable streams of real languages.
import re, io, contextlib, numpy as np
from common import *; from parse import parse
V='aeiouy'
def syllabify(w):
    # crude: onset consonants + vowel nucleus + (one coda consonant if next is consonant or end)
    s=re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]+$|[^aeiouy](?=[^aeiouy]))?', w)
    return s if ''.join(s)==w and s else [w]
def profile(words_units):
    seq=[]
    for u in words_units: seq+=list(u)+['_']
    c1=Counter(seq); c2=Counter(zip(seq,seq[1:])); h1=H(c1); h2=H(c2)-h1
    inv=len(c1)-1; per=np.mean([len(u) for u in words_units])
    fr=np.array(sorted(c1.values(),reverse=True)[1:51]); z=np.polyfit(np.log(np.arange(1,len(fr)+1)),np.log(fr),1)[0]
    top10=sum(sorted(c1.values(),reverse=True)[1:11])/sum(v for k,v in c1.items() if k!='_')
    return dict(inventory=inv,h1=round(h1,2),h2=round(h2,2),per_word=round(per,2),zipf50=round(z,2),top10=round(100*top10,1))
VW=[tok(w) for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words'] for w in l['words']]
def bpe(ws,k):
    ws=[tuple(w) for w in ws]
    for _ in range(k):
        pc=Counter()
        for w in ws:
            for x,y in zip(w,w[1:]): pc[(x,y)]+=1
        (a,b),_=pc.most_common(1)[0]
        def m(w):
            o=[];i=0
            while i<len(w):
                if i+1<len(w) and w[i]==a and w[i+1]==b: o.append(a+b); i+=2
                else: o.append(w[i]); i+=1
            return tuple(o)
        ws=[m(w) for w in ws]
    return ws
rows=[]
for k in [0,20,60,120,200]:
    rows.append((f'Voynich units, {k} merges',profile(bpe(VW,k))))
for n in ['latin.txt','italian.txt','czech.txt']:
    ws=[w for l in text_lines(n,'wrap') for w in l][:35000]
    rows.append((f'{n} letters',profile([list(w) for w in ws])))
    rows.append((f'{n} SYLLABLES',profile([syllabify(w) for w in ws])))
for name,p in rows: print(f'{name:28s}',p)
