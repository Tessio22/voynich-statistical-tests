# Do plants with similar MEDICAL PROPERTIES (from Macer Floridus) have similar VOCABULARY on their Voynich pages?
import re, random, numpy as np, json
from common import *; from parse import parse
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from exp16_collapse import norm_word
from collections import defaultdict

# --- 1. Macer chapters ---
t=open(D+'latin.txt',encoding='utf-8').read()
body=t[t.find('Herbarum quasdam'):]
body=t[t.find('I.  Artemisia.',t.find('77 LXXVII')):]
parts=re.split(r'\n\s*([IVXL]+)\.\s+([A-Z][a-zë ]+)\.\s*\n',body)
chap={}
for i in range(1,len(parts)-2,3):
    name=parts[i+1].strip().lower(); chap[name]=norm(parts[i+2]).replace('j','i').replace('v','u')
print('Macer chapters parsed:',len(chap))
FEATS={'hot':r'calid|feru|calefac','cold':r'frigid|refriger','dry':r'\bsicc','wet':r'humid|humect',
 'stomach':r'stomach','head':r'capit|caput|cephal','urine':r'urin','menses':r'menstru|matri|muliebr','poison':r'uenen|uirus',
 'cough':r'tuss','wounds':r'uulner|plag','eyes':r'oculo|lumin|caligo','teeth':r'dent','belly':r'uentr|aluu|uentris',
 'liver':r'hepat|iecor','spleen':r'splen|lien','kidney':r'renes|renum|calcul|lapid','chest':r'pector|pulmo','fever':r'febr',
 'snake':r'serpen|morsu|anguis|uiper','blood':r'sanguin|cruor','milk':r'lact','sleep':r'somn','vomit':r'uomit|nause',
 'ulcer':r'ulcer|carcin|cancer','worms':r'lumbric|uerm','birth':r'partu|abort|fetu','joints':r'podagr|articul|ischi|sciat|neru',
 'ears':r'aure|auri','skin':r'scabi|lepr|impetig|uerruc','swelling':r'tumor|tumid','bladder':r'uesic','thirst':r'sitis|sitim'}
def fvec(name):
    tx=chap[name]; return {f for f,p in FEATS.items() if re.search(p,tx)}
# --- 2. page -> Macer plant (my visual IDs + literature IDs) ---
CONF={'f9v':'violae','f56r':'buglossa','f51v':'buglossa','f32r':'mentha','f65v':'chamomilla','f25r':'urtica','f38v':'lactuca','f8r':'atriplex'}
LIT={'f25v':'plantago','f31r':'iusquiamus','f90r1':'papauer','f44r':'elleborus albus','f3v':'elleborus niger','f10v':'elleborus niger',
     'f53r':'enula','f7r':'paeonia','f9r':'chelidonia','f27r':'asarum','f6r':'aristolochia','f45v':'chamaedrys','f26v':'uerbena','f32v':'malua','f2r':'centaurea'}
ALL={**LIT,**CONF}
missing=[v for v in ALL.values() if v not in chap]; print('missing chapters:',missing)
# --- 3. page vocab (collapsed words, level 4) ---
pw=defaultdict(list)
for l in parse():
    if l['loc'] in ('P0','P1','Pb') and l['words']: pw[l['folio']]+=[norm_word(w,4) for w in l['words']]
glob=Counter(w for v in pw.values() for w in v)
def pvec(f, minf=3, maxf=400):
    return {w for w in pw[f] if minf<=glob[w]<=maxf}
def jacc(a,b): return len(a&b)/max(1,len(a|b))
def mantel(pages, reps=5000, seed=1):
    pages=[p for p in pages if p in pw and ALL[p] in chap]
    F={p:fvec(ALL[p]) for p in pages}; V={p:pvec(p) for p in pages}
    pairs=[(i,j) for i in range(len(pages)) for j in range(i+1,len(pages))]
    # exclude pairs mapped to the SAME Macer plant (trivial feature identity) -- keep them separately
    pairs=[(i,j) for i,j in pairs if ALL[pages[i]]!=ALL[pages[j]]]
    x=np.array([jacc(F[pages[i]],F[pages[j]]) for i,j in pairs]); y=np.array([jacc(V[pages[i]],V[pages[j]]) for i,j in pairs])
    r=np.corrcoef(x,y)[0,1]; R=random.Random(seed); null=[]
    for _ in range(reps):
        perm=pages[:]; R.shuffle(perm); mp=dict(zip(pages,perm))
        xx=np.array([jacc(F[mp[pages[i]]],F[mp[pages[j]]]) for i,j in pairs]); null.append(np.corrcoef(xx,y)[0,1])
    null=np.array(null); return r,(1+np.sum(null>=r))/(1+reps),len(pages),len(pairs)
for name,pg in [('my confident IDs',list(CONF)),('confident + literature',list(ALL))]:
    r,p,n,m=mantel(pg); print(f'{name:26s} pages={n} pairs={m}  correlation(properties, vocabulary) r={r:.3f}  p={p:.3f}')
for p_,pl in ALL.items(): print(f'  {p_:6s} {pl:16s} {sorted(fvec(pl)) if pl in chap else "-"}')
json.dump({'feats':{p:sorted(fvec(pl)) for p,pl in ALL.items() if pl in chap}},open(RESULTS+'res_exp18.json','w'),indent=1)
