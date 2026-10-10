from typing import Callable, List, Optional, Tuple
import warnings
import numpy as np
from random import random

U = lambda : random()
Exp = lambda lmbda: -1.0*np.log(U())/lmbda

MAX_DEFAULT_TERMS = 15

# Logit range used for feasibility checks and numerical integration.
LOGIT_RANGE = 25.0


def _expit(z):
    return 1.0/(1.0 + np.exp(-z))


def _basis(y, n_terms: int):
    # Keelin's metalog basis functions evaluated at probabilities y.
    # Returns an array of shape y.shape + (n_terms,).
    y = np.asarray(y, dtype=float)
    lny = np.log(y/(1-y))
    y_ = y - 0.5

    cols = [np.ones_like(y), lny, y_*lny, y_]
    for i in range(5, n_terms+1):
        if i%2:
            cols.append( np.power( y_, (i-1)//2 ) )
        else:
            cols.append( np.power( y_, i//2-1 )*lny )

    return np.stack(cols[:n_terms], axis=-1)


def _basis_derivative(y, n_terms: int):
    # d/dy of each basis function in _basis.
    y = np.asarray(y, dtype=float)
    lny = np.log(y/(1-y))
    dlny = 1/(y*(1-y))
    y_ = y - 0.5

    cols = [np.zeros_like(y), dlny, lny + y_*dlny, np.ones_like(y)]
    for i in range(5, n_terms+1):
        if i%2:
            k = (i-1)//2
            cols.append( k*np.power( y_, k-1 ) )
        else:
            k = i//2-1
            cols.append( k*np.power( y_, k-1 )*lny + np.power( y_, k )*dlny )

    return np.stack(cols[:n_terms], axis=-1)


class _MetalogBase():
    # Shared machinery for a fitted metalog. Subclasses set self.a (coefficients),
    # self.n_terms, and call _set_bounds.

    def _set_bounds(self, bounds: Tuple[float,float]):
        if bounds[0] >= bounds[1]:
            raise ValueError("Lower bound cannot be greater or equal to Upper bound!")
        if np.isposinf(bounds[0]):
            raise ValueError("Lower bound cannot be infinity!")
        if np.isneginf(bounds[1]):
            raise ValueError("Upper bound cannot be negative infinity!")

        lower_bounded = not np.isneginf(bounds[0])
        upper_bounded = not np.isposinf(bounds[1])
        if lower_bounded and upper_bounded:
            self.kind = 'bounded'
        elif lower_bounded:
            self.kind = 'lowerbounded'
        elif upper_bounded:
            self.kind = 'upperbounded'
        else:
            self.kind = 'unbounded'

        self.bl = bounds[0]
        self.bu = bounds[1]

    def _to_unbounded(self, x):
        # Maps values in the bounded support onto the real line (inverse of sample_transform).
        if self.kind == 'unbounded':
            return x
        elif self.kind == 'lowerbounded':
            return np.log(x-self.bl)
        elif self.kind == 'upperbounded':
            return -1*np.log(self.bu-x)
        elif self.kind == 'bounded':
            return np.log( (x-self.bl)/(self.bu-x) )

    def sample_transform(self,sample):
        # Maps a value on the real line back into the bounded support.
        if self.kind == 'unbounded':
            return sample
        elif self.kind == 'lowerbounded':
            return self.bl + np.exp(sample)
        elif self.kind == 'upperbounded':
            return self.bu - np.exp(-1*sample)
        elif self.kind == 'bounded':
            return self.bl + (self.bu-self.bl)*_expit(sample)

    def _transform_derivative(self, m):
        # d/dm of sample_transform.
        if self.kind == 'unbounded':
            return np.ones_like(m)
        elif self.kind == 'lowerbounded':
            return np.exp(m)
        elif self.kind == 'upperbounded':
            return np.exp(-1*m)
        elif self.kind == 'bounded':
            p = _expit(m)
            return (self.bu-self.bl)*p*(1-p)

    def quantile(self, y):
        # Inverse CDF: the value x with P(X <= x) = y. Accepts scalars or arrays.
        y = np.asarray(y, dtype=float)
        if np.any((y < 0) | (y > 1)):
            raise ValueError("Probabilities must lie in [0, 1].")

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            x = self.sample_transform(_basis(y, self.n_terms) @ self.a)
        x = np.where(y == 0, self.bl, x)
        x = np.where(y == 1, self.bu, x)
        return x[()]

    def _quantile_derivative(self, y):
        # dQ/dy, the reciprocal of the density at Q(y).
        m = _basis(y, self.n_terms) @ self.a
        dm = _basis_derivative(y, self.n_terms) @ self.a
        return self._transform_derivative(m)*dm

    def cdf(self, x, tol: float = 1e-12, max_iter: int = 200):
        # P(X <= x), found by bisecting the quantile function in logit space.
        x = np.asarray(x, dtype=float)
        lo = np.full(x.shape, -36.0)
        hi = np.full(x.shape, 36.0)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            for _ in range(max_iter):
                mid = (lo+hi)/2
                below = self.quantile(_expit(mid)) < x
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
                if np.all(hi-lo < tol):
                    break

        y = _expit((lo+hi)/2)
        y = np.where(x <= self.bl, 0.0, y)
        y = np.where(x >= self.bu, 1.0, y)
        return y[()]

    def pdf(self, x):
        # Probability density at x, via 1/Q'(F(x)).
        x = np.asarray(x, dtype=float)
        y = np.asarray(self.cdf(x))
        inside = (x > self.bl) & (x < self.bu) & (y > 0) & (y < 1)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            density = 1.0/self._quantile_derivative(np.where(inside, y, 0.5))
        return np.where(inside, density, 0.0)[()]

    def rvs(self, size=None, rng=None):
        # Draws samples by inverse-transform sampling. rng may be a seed or a numpy Generator.
        rng = np.random.default_rng(rng)
        return self.quantile(rng.random(size))

    def cvar(self, alpha: float = 0.05, grid_size: int = 4000) -> float:
        # Conditional value at risk of the lower tail: the mean of the worst alpha fraction of
        # outcomes, (1/alpha) * integral_0^alpha Q(y) dy. Integrated in logit space so the tail is resolved.
        if not 0 < alpha <= 1:
            raise ValueError("alpha must lie in (0, 1].")

        upper = min(np.log(alpha/(1-alpha)), LOGIT_RANGE) if alpha < 1 else LOGIT_RANGE
        z = np.linspace(-LOGIT_RANGE, upper, grid_size)
        y = _expit(z)
        f = self.quantile(y)*y*(1-y)
        return float(np.sum((f[1:]+f[:-1])*np.diff(z))/2/alpha)

    def is_feasible(self, grid_size: int = 10000) -> bool:
        # A metalog is valid only if its quantile function is strictly increasing on (0, 1).
        y = _expit(np.linspace(-LOGIT_RANGE, LOGIT_RANGE, grid_size))
        return bool(np.all(_basis_derivative(y, self.n_terms) @ self.a > 0))


class metalog(_MetalogBase):

    def __init__(self,b: int,quantiles: List[Tuple], n_terms: Optional[int] = None,bounds:Tuple[float,float] = (-np.inf,np.inf)):
        # Properties of a meta-logistic distribution.
        # quantiles is a list of (value, probability) pairs.
        # n_terms defaults to min(b, 15) and cannot exceed b.

        if n_terms is None:
            n_terms = min(b, MAX_DEFAULT_TERMS)

        self.b = b # no. of quantiles
        self.quantiles = quantiles # List of quantiles
        self.n_terms = n_terms

        if len(quantiles) != b:
            raise ValueError(f"Expected {b} quantiles, got {len(quantiles)}.")
        if n_terms < 2:
            raise ValueError("n_terms must be at least 2.")
        if n_terms > b:
            raise ValueError(f"n_terms ({n_terms}) cannot exceed the number of quantiles ({b}): the fit would be underdetermined.")

        self._set_bounds(bounds)

        # Quantiles must lie strictly inside the bounds, and the 0% and 100% quantiles are not allowed.
        for value, prob in quantiles:
            if not self.bl < value < self.bu:
                raise ValueError(f"Quantile value {value} must lie strictly inside the bounds ({self.bl}, {self.bu}).")
            if not 0.0 < prob < 1.0:
                raise ValueError(f"Quantile probability {prob} must lie strictly between 0 and 1.")

        # Estimating parameters using least squares.
        X = self._to_unbounded(np.array([value for value, _ in quantiles], dtype=float))
        Y = _basis(np.array([prob for _, prob in quantiles], dtype=float), self.n_terms)

        self.a, _, rank, _ = np.linalg.lstsq(Y, X, rcond=None)
        if rank < self.n_terms:
            raise ValueError(
                f"Only {rank} of the {self.n_terms} terms can be determined from these quantiles: "
                "the probabilities repeat, or high-order terms are numerically indistinguishable. Use fewer terms."
            )
        self.err = np.linalg.norm( X - Y @ self.a, ord=2)

        if not self.is_feasible():
            warnings.warn(
                "The fitted metalog is not feasible: its quantile function is not increasing everywhere. "
                "Try fewer terms or different quantiles.",
                RuntimeWarning, stacklevel=2,
            )

    @property
    def quantile_val(self):
        return [quantile[0] for quantile in self.quantiles]

    @classmethod
    def from_sampler(self,b: int,sampler: Callable[[],float],n_terms: Optional[int] = None,bounds:Tuple[float,float] = (-np.inf,np.inf),num_data: int = 10000):
        # Generating data from a distribution
        data = [ sampler() for _ in range(num_data) ]
        return self.from_data(b,data,n_terms,bounds)

    @classmethod
    def from_data(self,b: int,data,n_terms: Optional[int] = None,bounds:Tuple[float,float] = (-np.inf,np.inf)):
        # Generating Quantiles from data
        quantiles = [ (x, (i+1)/(b+1)) for i,x in enumerate(np.quantile(data,[i/(b+1) for i in range(1,b+1)])) ]
        return metalog(b,quantiles,n_terms,bounds)

    def sampler(self,kind = 'metalog'):
        # Sampling from a linear piecewise CDF.
        if kind == "piecewise linear":
            rn = U()
            idx = int(self.b*rn)
            if idx == self.b-1:
                return self.quantiles[self.b-1][0]
            else:
                return (self.quantiles[idx+1][0] - self.quantiles[idx][0])*(self.b*rn-idx) + self.quantiles[idx][0]

        # Sampling from the fitted metalog.
        elif kind == "metalog":
            return float(self.quantile(U()))

    def distance(self,dist):
        assert(self.b == dist.b)
        distance = 0.0
        for i in range(self.b):
            temp = self.quantile_val[i] - dist.quantile_val[i]
            distance += np.abs(temp)
        return distance/self.b


class OnlineMetalog(_MetalogBase):
    # A metalog learned from a stream of observations without storing them.
    #
    # Each observation triggers one projected stochastic gradient step on the pinball (quantile) loss,
    # evaluated at n_taus jittered probabilities. The loss is convex in the coefficients, and
    # "the quantile function is increasing" is a set of linear constraints, so after every step the
    # coefficients are projected back onto that set and the distribution stays valid at all times.
    # Queries use the running average of the iterates (Polyak averaging), which smooths out
    # single-observation noise; set average=False and decay=0 to track a drifting distribution instead.

    def __init__(self, n_terms: int = 5, bounds: Tuple[float,float] = (-np.inf,np.inf), lr: float = 1.0,
                 decay: float = 0.5, n_taus: int = 64, average: bool = True, warmup: int = 100,
                 n_anchors: int = 200, rng=None):
        if n_terms < 2:
            raise ValueError("n_terms must be at least 2.")
        if n_taus < 1:
            raise ValueError("n_taus must be at least 1.")
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if not 0 <= decay <= 1:
            raise ValueError("decay must lie in [0, 1].")
        if warmup < 0:
            raise ValueError("warmup cannot be negative.")

        self._set_bounds(bounds)
        self.n_terms = n_terms
        self.lr = lr
        self.decay = decay
        self.n_taus = n_taus
        self.average = average
        self.warmup = warmup
        self._rng = np.random.default_rng(rng)

        self.n = 0 # no. of observations seen
        self._w = np.zeros(n_terms) # current SGD iterate
        self._w_avg = np.zeros(n_terms) # running average of the iterates
        self._buffer = [] # the first `warmup` observations, batch-fitted before switching to SGD

        # Running mean and variance of the transformed observations, used to scale the step size.
        self._mean = 0.0
        self._m2 = 0.0

        # Feasibility constraints: Q'(y) > 0 at anchor probabilities spread evenly in logit space.
        # Rows are normalized so the projection works in units of the coefficients.
        anchors = _expit(np.linspace(-LOGIT_RANGE, LOGIT_RANGE, n_anchors))
        G = _basis_derivative(anchors, n_terms)
        self._G = G/np.linalg.norm(G, axis=1, keepdims=True)

        # The basis functions differ in scale by an order of magnitude, which makes plain gradient steps
        # crawl along the small-scale (tail-shaping) directions. Steps are preconditioned with the inverse
        # Gram matrix of the basis under uniform probabilities.
        grid = (np.arange(4000) + 0.5)/4000
        B = _basis(grid, n_terms)
        self._P = np.linalg.inv(B.T @ B/len(grid))

    @property
    def a(self):
        return self._w_avg if self.average else self._w

    @property
    def _scale(self):
        if self.n > 1 and self._m2 > 0:
            return np.sqrt(self._m2/(self.n-1))
        return 1.0

    def update(self, x):
        # Learns from one observation or an array of observations, in order.
        z_all = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        if np.any((z_all <= self.bl) | (z_all >= self.bu)) or np.any(np.isnan(z_all)):
            raise ValueError(
                f"Observations must lie strictly inside the bounds ({self.bl}, {self.bu}); "
                "widen the bounds slightly if values can land exactly on them."
            )

        for z in self._to_unbounded(z_all):
            self._step(z)
        return self

    def _step(self, z: float):
        self.n += 1
        delta = z - self._mean
        self._mean += delta/self.n
        self._m2 += delta*(z - self._mean)

        if self._buffer is not None:
            self._buffer.append(z)
            self._w[:] = self._fit_buffer()
            self._project()
            self._w_avg[:] = self._w
            if self.n >= self.warmup:
                self._buffer = None
            return

        # Jittered probabilities, one per stratum of (0, 1).
        taus = (np.arange(self.n_taus) + self._rng.random(self.n_taus))/self.n_taus
        taus = np.clip(taus, 1e-12, 1-1e-12)
        B = _basis(taus, self.n_terms)

        # Subgradient of the mean pinball loss: -(tau - 1[z < Q(tau)]) * basis(tau).
        below = (z < B @ self._w).astype(float)
        grad = -((taus - below) @ B)/self.n_taus

        self._w -= self.lr*self._scale/self.n**self.decay*(self._P @ grad)
        self._project()
        self._w_avg += 2*(self._w - self._w_avg)/(self.n+1)

    def _fit_buffer(self):
        # Least-squares metalog through the buffered order statistics. Few points get few terms,
        # so a handful of observations cannot produce a wildly oscillating fit.
        z = np.sort(self._buffer)
        m = len(z)
        k = max(min(m, 2), min(self.n_terms, m//5))
        w = np.zeros(self.n_terms)
        w[:k] = np.linalg.lstsq(_basis((np.arange(m) + 0.5)/m, k), z, rcond=None)[0]
        return w

    def _project(self, margin: float = 1e-9, max_sweeps: int = 100):
        # Cyclic projections onto the violated half-spaces G w >= margin.
        w = self._w
        for _ in range(max_sweeps):
            violated = np.flatnonzero(self._G @ w < margin)
            if violated.size == 0:
                return
            for k in violated:
                gap = margin - self._G[k] @ w
                if gap > 0:
                    w += gap*self._G[k]
