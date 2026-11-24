import json, re, math, random, unicodedata
from collections import Counter

import os
D = os.environ.get('VOYNICH_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')) + os.sep
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results') + os.sep
G = ['cth','ckh','cph','cfh','ch','sh','iiin','iin','in','iiir','iir','ir']

def tok(w):
    o=[];i=0
    while i<len(w):
        for g in G:
            if w.startswith(g,i):
                o.append(g); i+=len(g); break
        else:
            o.append(w[i]); i+=1
    return o

def voy_lines(locs=('P0','P1','Pb'), lang=None):
    L=json.load(open(D+'voynich_lines.json'))
    out=[]
    for l in L:
        if locs and l['loc'] not in locs: continue
        if lang and l.get('L')!=lang: continue
        if l['words']: out.append(l)
    return out

def norm(s):
    s=unicodedata.normalize('NFD',s.lower())
    s=''.join(c for c in s if not unicodedata.combining(c))
    s=s.replace('ß','ss').replace('j','i').replace('v','u') if False else s
    return s

def words_of(line):
    return re.findall(r'[a-zäöüß]+', norm(line))

def load_text(name):
    t=open(D+name,encoding='utf-8').read()
    t=t.replace('ſ','s').replace('ľ','l').replace('Ľ','L').replace('·','').replace('ı','i').replace('İ','I')
    t=re.sub(r'\[[^\]]*\]','',t)
    return t

def text_lines(name, mode):
    """mode 'orig' = keep source lines, 'wrap' = rewrap to ~9 words/line"""
    t=load_text(name)
    raw=[words_of(x) for x in t.split('\n')]
    raw=[[w for w in x if not re.fullmatch(r'[ivxlcdm]+',w) or len(w)>4] for x in raw]
    raw=[x for x in raw if len(x)>=3]
    if mode=='orig': return raw
    ws=[w for x in raw for w in x]
    return [ws[i:i+9] for i in range(0,len(ws)-8,9)]

def H(c):
    n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values())

def mi(pairs):
    j=Counter(pairs); a=Counter(p[0] for p in pairs); b=Counter(p[1] for p in pairs)
    return H(a)+H(b)-H(j)

def mi_excess(pairs, reps=20, seed=1):
    r=random.Random(seed); obs=mi(pairs); ys=[p[1] for p in pairs]; ns=[]
    for _ in range(reps):
        r.shuffle(ys); ns.append(mi(list(zip([p[0] for p in pairs],ys))))
    m=sum(ns)/len(ns); return obs, m, obs-m
