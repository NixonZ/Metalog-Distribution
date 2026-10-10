from typing import Callable, List, Optional, Tuple
import warnings
import numpy as np
from random import random

U = lambda : random()
Exp = lambda lmbda: -1.0*np.log(U())/lmbda

MAX_DEFAULT_TERMS = 15


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


class metalog():

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
            raise ValueError(f"The quantile probabilities do not determine {self.n_terms} terms (rank {rank}); use distinct probabilities or fewer terms.")
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

    def is_feasible(self, grid_size: int = 10000) -> bool:
        # A metalog is valid only if its quantile function is strictly increasing on (0, 1).
        y = _expit(np.linspace(-25, 25, grid_size))
        return bool(np.all(_basis_derivative(y, self.n_terms) @ self.a > 0))

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
