from exp10_generators import *
import itertools
V=metrics(VPAGES)
keys=['h2','wlen','ttr30k','hapax','zipf','MIin','MIacross','rep','local','sh2','shLast','ktHet']
print(f"{'':26s}"+''.join(f'{k:>8s}' for k in keys))
print(f"{'VOYNICH':26s}"+''.join(f'{V[k]:>8}' for k in keys))
for cp,dr in itertools.product([0.2,0.3,0.4],[0.2]):
    m=metrics(generate(kind='jpos',drift=dr,copy_p=cp))
    print(f"{'G copy=%.2f drift=%.2f'%(cp,dr):26s}"+''.join(f'{m[k]:>8}' for k in keys),flush=True)
