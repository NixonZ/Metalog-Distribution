from metalog import metalog
import numpy as np

dist = metalog(
    b = 4,
    quantiles = [(0.1,0.2),(1.0,0.4),(2.0,0.6),(3.0,0.8)],
    n_terms = 4,
    bounds = (0.0,np.inf)
)

sample = [dist.sampler(kind = 'metalog') for n in range(10)]

# print(sample)

print(metalog.from_data(4,sample,4,(0.0,np.inf)).quantile_val)