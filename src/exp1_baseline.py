from common import *

def metrics(lines, unit):
    # lines: list of list of words; unit: function word->list of symbols
    seq=[]
    for ws in lines:
        for w in ws: seq += unit(w)+['_']
    c1=Counter(seq); c2=Counter(zip(seq,seq[1:]))
    h1=H(c1); h2=H(c2)-h1
    wl=[len(unit(w)) for ws in lines for w in ws]
    toks=[w for ws in lines for w in ws][:30000]
    ttr=len(set(toks))/len(toks)
    within=[(unit(ws[i])[-1],unit(ws[i+1])[0]) for ws in lines for i in range(len(ws)-1)]
    across=[(unit(lines[k][-1])[-1],unit(lines[k+1][0])[0]) for k in range(len(lines)-1)]
    o1,n1,e1=mi_excess(within); o2,n2,e2=mi_excess(across)
    # repeated adjacent words
    rep=sum(ws[i]==ws[i+1] for ws in lines for i in range(len(ws)-1))/len(within)
    # line-first word length vs others
    fl=sum(len(unit(ws[0])) for ws in lines)/len(lines)
    ol=[len(unit(w)) for ws in lines for w in ws[1:]]
    return dict(alph=len(c1)-1, h1=round(h1,2), h2=round(h2,2), wlen=round(sum(wl)/len(wl),2),
                ttr30k=round(ttr,3), MIwithin=round(e1,3), MIacross=round(e2,3), ratio=round(e1/max(e2,1e-3),1),
                rep=round(100*rep,2), firstLen=round(fl,2), otherLen=round(sum(ol)/len(ol),2), nwords=len(wl))

chars=lambda w: list(w)
V=[l['words'] for l in voy_lines()]
rows={}
rows['Voynich (glyphs)']=metrics(V,tok)
rows['Voynich (EVA chars)']=metrics(V,chars)
rows['Voynich A']=metrics([l['words'] for l in voy_lines(lang='A')],tok)
rows['Voynich B']=metrics([l['words'] for l in voy_lines(lang='B')],tok)
rows['Latin verse (Macer, orig lines)']=metrics(text_lines('latin.txt','orig')[:1800],chars)
rows['Latin (rewrapped)']=metrics(text_lines('latin.txt','wrap'),chars)
rows['Italian Decameron (rewrapped)']=metrics(text_lines('italian.txt','wrap'),chars)
rows['German Ackermann (MS lines)']=metrics(text_lines('german.txt','orig'),chars)
rows['German (rewrapped)']=metrics(text_lines('german.txt','wrap'),chars)
import json
for k,v in rows.items(): print(f'{k:34s}',v)
json.dump(rows,open(RESULTS+'res_exp1.json','w'),indent=1)
