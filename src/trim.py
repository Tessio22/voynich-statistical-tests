from common import tok
PRE={'q','o','y','d','s','a','l'}            # glyphs that typically open a word (prefix slot)
SUF={'y','l','r','iin','in','m','s','ir','d','n','iiin'}  # glyphs that typically close a word (suffix slot)
def trim(w, mode):
    g=tok(w)
    if mode in ('pre','core'):
        while len(g)>1 and g[0] in PRE: g=g[1:]
    if mode in ('suf','core'):
        while len(g)>1 and g[-1] in SUF: g=g[:-1]
    return ''.join(g)
