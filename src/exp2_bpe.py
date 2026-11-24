# Can Voynich glyph stream be re-segmented into units that look like letters of a natural language?
from common import *
from parse import parse
from collections import Counter
L=[l for l in parse() if l['loc'] in ('P0','P1','Pb') and l['words']]
words=[tuple(tok(w)) for l in L for w in l['words']]

def stats(ws):
    seq=[]
    for w in ws: seq+=list(w)+['_']
    c1=Counter(seq); c2=Counter(zip(seq,seq[1:])); c3=Counter(zip(seq,seq[1:],seq[2:]))
    h1=H(c1); h2=H(c2)-h1; h3=H(c3)-H(c2)
    wl=sum(len(w) for w in ws)/len(ws)
    return len(c1)-1, h1, h2, h3, wl

def merge(ws, pair):
    a,b=pair; out=[]
    for w in ws:
        o=[];i=0
        while i<len(w):
            if i+1<len(w) and w[i]==a and w[i+1]==b: o.append(a+b); i+=2
            else: o.append(w[i]); i+=1
        out.append(tuple(o))
    return out

ws=words
print('merges alph   h1    h2   h3  h2/h1 wlen')
for k in range(0,61):
    if k%5==0:
        a,h1,h2,h3,wl=stats(ws); print(f'{k:5d} {a:4d} {h1:5.2f} {h2:5.2f} {h3:5.2f} {h2/h1:5.2f} {wl:4.2f}')
    pc=Counter()
    for w in ws:
        for x,y in zip(w,w[1:]): pc[(x,y)]+=1
    best=pc.most_common(1)[0][0]
    ws=merge(ws,best)
    if k<25: print('   merge',k+1,best)

for name in ['latin.txt','italian.txt','german.txt']:
    T=[tuple(w) for ws_ in text_lines(name,'wrap') for w in ws_][:35000]
    a,h1,h2,h3,wl=stats(T); print(f'{name:12s} alph {a} h1 {h1:.2f} h2 {h2:.2f} h3 {h3:.2f} h2/h1 {h2/h1:.2f} wlen {wl:.2f}')
