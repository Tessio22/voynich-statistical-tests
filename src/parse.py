import re, json
from common import D
def parse(path=D+'ZL3b.txt', comma='join'):
    meta={}; out=[]
    for ln in open(path,encoding='latin-1'):
        ln=ln.rstrip('\n')
        m=re.match(r'^<(f[0-9a-z]+)>\s+<!(.*)>',ln)
        if m:
            meta={'folio':m.group(1)}
            for k,v in re.findall(r'\$(\w)=(\w+)',m.group(2)): meta[k]=v
            continue
        m=re.match(r'^<(f[0-9a-z]+)\.(\d+),(.)(\w+)>\s+(.*)$',ln)
        if not m: continue
        s=m.group(5)
        s=re.sub(r'<![^>]*>','',s); s=re.sub(r'<[^>]*>','',s)
        s=re.sub(r'\[([^:\]]*):[^\]]*\]',r'\1',s)
        s=re.sub(r"[{}']",'',s)
        if comma=='join': s=s.replace(',','')
        else: s=s.replace(',','.')
        raw=[w for w in s.split('.') if w]
        words=[w for w in raw if re.fullmatch(r'[a-z]+',w)]
        clean = len(words)==len(raw)   # line with no unreadable/odd tokens
        d=dict(meta); d.update(line=int(m.group(2)),pos=m.group(3),loc=m.group(4),words=words,clean=clean)
        out.append(d)
    return out
