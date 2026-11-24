from common import *; from parse import parse
import random
def test(blocks, unit, name, k=2):
    # blocks: list of list of lines (each line list of words); compare line i with i+d
    r=random.Random(3); res={}
    for d in [1,2,3]:
        m=0;n=0;mr=0;nr=0;fm=0;fr=0
        for B in blocks:
            for i in range(len(B)-d):
                a=unit(B[i][-1]); b=unit(B[i+d][-1])
                m+= tuple(a[-k:])==tuple(b[-k:]); n+=1
                fa=unit(B[i][0]); fb=unit(B[i+d][0]); fm+= tuple(fa[:k])==tuple(fb[:k])
                # random control: random line from another block
                C=blocks[r.randrange(len(blocks))]; c=C[r.randrange(len(C))]
                mr+= tuple(a[-k:])==tuple(unit(c[-1])[-k:]); fr+= tuple(fa[:k])==tuple(unit(c[0])[:k])
        res[d]=f'end {100*m/n:.1f}% vs rnd {100*mr/n:.1f}% | start {100*fm/n:.1f}% vs rnd {100*fr/n:.1f}%'
    print(name, res)
# Voynich paragraphs
blocks=[];cur=[]
for l in parse():
    if l['loc'] not in ('P0','P1','Pb') or not l['words']: 
        continue
    if l['pos'] in '*@' or (cur and cur[-1][0]!=l['folio']):
        if len(cur)>2: blocks.append([x[1] for x in cur])
        cur=[]
    cur.append((l['folio'],l['words']))
if len(cur)>2: blocks.append([x[1] for x in cur])
print('Voynich paragraphs',len(blocks))
test(blocks,tok,'Voynich')
def verse_blocks(name,size=12):
    L=[words_of(re.sub(r'\d+','',x)) for x in load_text(name).split('\n')]
    L=[x for x in L if len(x)>=3]
    return [L[i:i+size] for i in range(0,len(L)-size,size)]
test(verse_blocks('dante.txt'),list,'Dante (terza rima)')
test(verse_blocks('latin.txt')[:150],list,'Latin verse (Macer)')
test([text_lines('italian.txt','wrap')[i:i+12] for i in range(0,3000,12)],list,'Italian prose')
test([text_lines('german.txt','orig')[i:i+12] for i in range(0,1200,12)],list,'German MS lines')
