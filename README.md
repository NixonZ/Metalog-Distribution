# Metalog-Distribution
Python implementation of the Metalog distribution.

Keelin, T. W. (2016). The metalog distributions. Decision Analysis: A Journal of the Institute for Operations Research and the Management Sciences, 13(4), 243–277. doi:10.1287/deca.2016.0338

## Install

```
pip install git+https://github.com/NixonZ/Metalog-Distribution
```

## Usage

Fit a metalog to quantiles or data:

```python
from metalog import metalog

dist = metalog(4, [(0.5, 0.2), (1.2, 0.4), (2.0, 0.6), (3.0, 0.8)], bounds=(0, 4))
dist.quantile(0.5), dist.cdf(1.0), dist.pdf(1.0)
dist.rvs(1000, rng=0)
dist.cvar(0.05)          # mean of the worst 5% of outcomes
dist.is_feasible()       # is the quantile function increasing everywhere?

dist = metalog.from_data(99, data, n_terms=5)
```

Learn a metalog from a stream, without storing the data:

```python
from metalog import OnlineMetalog

om = OnlineMetalog(n_terms=5, bounds=(-float('inf'), float('inf')), rng=0)
for x in stream:
    om.update(x)         # also accepts arrays
    risk = om.cvar(0.05)
```

`OnlineMetalog` takes one preconditioned stochastic gradient step on the pinball loss per observation, then
projects the coefficients so the quantile function stays increasing; queries use a weighted average of the
iterates. The first `warmup` observations (default 100) are fitted in one batch before switching to updates.
For a distribution that drifts over time, use `average=False, decay=0, lr=0.02` (a constant, smaller step)
so recent data keeps its influence; the median of a stream that jumps from N(0,1) to N(5,1) recovers within ~1000 steps.

Fewer terms are usually better: a single metalog cannot represent well-separated modes, and least-squares
fits with many terms are often infeasible. `is_feasible()` and the warning on construction catch this.
