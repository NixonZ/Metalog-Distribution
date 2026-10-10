import warnings

import numpy as np
import pytest

from metalog import metalog

QUANTILES = [(0.5, 0.2), (1.2, 0.4), (2.0, 0.6), (3.0, 0.8)]
PROBS = np.array([p for _, p in QUANTILES])
VALUES = np.array([x for x, _ in QUANTILES])

BOUNDS = {
    'unbounded': (-np.inf, np.inf),
    'lowerbounded': (0.0, np.inf),
    'upperbounded': (-np.inf, 4.0),
    'bounded': (0.0, 4.0),
}


@pytest.fixture(params=list(BOUNDS))
def dist(request):
    return metalog(4, QUANTILES, n_terms=4, bounds=BOUNDS[request.param])


def test_kind_matches_bounds(dist):
    assert BOUNDS[dist.kind] == (dist.bl, dist.bu)


def test_exact_fit_reproduces_quantiles(dist):
    assert dist.is_feasible()
    np.testing.assert_allclose(dist.quantile(PROBS), VALUES, atol=1e-9)
    assert dist.err < 1e-9


def test_quantile_endpoints_are_bounds(dist):
    assert dist.quantile(0.0) == dist.bl
    assert dist.quantile(1.0) == dist.bu


def test_cdf_inverts_quantile(dist):
    y = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(dist.cdf(dist.quantile(y)), y, atol=1e-9)


def test_cdf_outside_bounds():
    dist = metalog(4, QUANTILES, n_terms=4, bounds=(0.0, 4.0))
    assert dist.cdf(-1.0) == 0.0
    assert dist.cdf(5.0) == 1.0
    assert dist.pdf(-1.0) == 0.0
    assert dist.pdf(5.0) == 0.0


def test_pdf_matches_cdf_derivative(dist):
    x = dist.quantile(np.linspace(0.05, 0.95, 20))
    h = 1e-6
    numeric = (dist.cdf(x + h) - dist.cdf(x - h))/(2*h)
    np.testing.assert_allclose(dist.pdf(x), numeric, rtol=1e-4)


def test_samples_match_quantiles_and_bounds(dist):
    samples = dist.rvs(200_000, rng=0)
    assert np.all((samples >= dist.bl) & (samples <= dist.bu))
    np.testing.assert_allclose(np.quantile(samples, PROBS), VALUES, atol=0.05)


def test_rvs_is_reproducible_with_seed(dist):
    np.testing.assert_array_equal(dist.rvs(10, rng=42), dist.rvs(10, rng=42))


def test_legacy_sampler(dist):
    for _ in range(100):
        x = dist.sampler(kind='metalog')
        assert isinstance(x, float)
        assert dist.bl <= x <= dist.bu


def test_too_many_terms_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        metalog(4, QUANTILES, n_terms=5)


def test_default_terms_capped_by_quantiles():
    assert metalog(4, QUANTILES).n_terms == 4


def test_duplicate_probabilities_rejected():
    with pytest.raises(ValueError, match="do not determine"):
        metalog(3, [(1.0, 0.5), (2.0, 0.5), (3.0, 0.5)], n_terms=3)


@pytest.mark.parametrize('quantiles, bounds', [
    ([(-1.0, 0.2), (1.0, 0.4), (2.0, 0.6), (3.0, 0.8)], (0.0, np.inf)),
    ([(0.5, 0.2), (1.2, 0.4), (2.0, 0.6), (5.0, 0.8)], (-np.inf, 4.0)),
    ([(0.5, 0.0), (1.2, 0.4), (2.0, 0.6), (3.0, 0.8)], (-np.inf, np.inf)),
])
def test_invalid_quantiles_rejected(quantiles, bounds):
    with pytest.raises(ValueError):
        metalog(4, quantiles, n_terms=4, bounds=bounds)


@pytest.mark.parametrize('bounds', [(1.0, 1.0), (np.inf, np.inf), (-np.inf, -np.inf)])
def test_invalid_bounds_rejected(bounds):
    with pytest.raises(ValueError):
        metalog(4, QUANTILES, n_terms=4, bounds=bounds)


def test_infeasible_fit_warns():
    # Values decrease as probability increases, so the quantile function cannot be increasing.
    with pytest.warns(RuntimeWarning, match="not feasible"):
        dist = metalog(2, [(2.0, 0.25), (1.0, 0.75)], n_terms=2)
    assert not dist.is_feasible()


def test_feasible_fit_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        metalog(4, QUANTILES, n_terms=4)


def test_from_data_recovers_normal():
    data = np.random.default_rng(0).normal(size=100_000)
    dist = metalog.from_data(9, data, n_terms=5)
    assert dist.n_terms == 5
    assert dist.is_feasible()
    assert abs(dist.quantile(0.5)) < 0.02
    assert abs(dist.quantile(0.9) - 1.2816) < 0.05


def test_from_sampler_bounded():
    rng = np.random.default_rng(1)
    dist = metalog.from_sampler(5, lambda: rng.exponential(), n_terms=5, bounds=(0.0, np.inf))
    assert dist.kind == 'lowerbounded'
    assert abs(dist.quantile(0.5) - np.log(2)) < 0.05
