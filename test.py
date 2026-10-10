from metalog import metalog
import numpy as np

dist = metalog(
    b = 4,
    quantiles = [(0.1,0.2),(1.0,0.4),(2.0,0.6),(3.0,0.8)],
    n_terms = 4,
    bounds = (0.0,4.0)
)

N = int(1e5)

sample = [dist.sampler(kind = 'metalog') for n in range(N)]

# print(sample)

b = 5

# print(sample)
print([ (x, (i+1)/(b+1)) for i,x in enumerate(np.quantile(sample,[i/(b+1) for i in range(1,b+1)])) ])

print(metalog.from_data(b,sample,5,(0.0,np.inf)).quantiles)