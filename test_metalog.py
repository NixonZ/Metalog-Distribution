import warnings

import numpy as np
import pytest

from metalog import OnlineMetalog, metalog

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
    with pytest.raises(ValueError, match="can be determined"):
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


NORMAL_CVAR_5 = -2.0627  # -pdf(z_0.05)/0.05 for a standard normal


def test_cvar_of_batch_fit_matches_normal():
    data = np.random.default_rng(0).normal(size=200_000)
    dist = metalog.from_data(99, data, n_terms=5)
    assert abs(dist.cvar(0.05) - NORMAL_CVAR_5) < 0.05


def test_cvar_whole_distribution_is_mean(dist):
    samples = dist.rvs(200_000, rng=0)
    assert abs(dist.cvar(1.0) - samples.mean()) < 0.02


@pytest.mark.parametrize('alpha', [0.0, -0.1, 1.5])
def test_cvar_rejects_bad_alpha(dist, alpha):
    with pytest.raises(ValueError):
        dist.cvar(alpha)


def test_online_learns_normal():
    data = np.random.default_rng(0).normal(size=20_000)
    om = OnlineMetalog(rng=0).update(data)
    assert om.n == 20_000
    np.testing.assert_allclose(om.quantile([0.05, 0.5, 0.95]), [-1.645, 0.0, 1.645], atol=0.1)
    assert abs(om.cvar(0.05) - NORMAL_CVAR_5) < 0.2


def test_online_stays_feasible_throughout():
    data = np.random.default_rng(1).standard_t(3, size=2_000)
    om = OnlineMetalog(rng=1)
    for chunk in np.array_split(data, 40):
        om.update(chunk)
        assert om.is_feasible()


def test_online_bounded():
    data = np.random.default_rng(2).lognormal(0, 0.5, size=5_000)
    om = OnlineMetalog(bounds=(0.0, np.inf), rng=2).update(data)
    assert om.kind == 'lowerbounded'
    assert abs(om.quantile(0.5) - 1.0) < 0.05
    assert np.all(om.rvs(1_000, rng=0) > 0)


def test_online_single_observations_and_warmup_release():
    om = OnlineMetalog(warmup=10, rng=0)
    for x in np.random.default_rng(3).normal(size=50):
        om.update(float(x))
    assert om.n == 50
    assert om._buffer is None
    assert om.is_feasible()


def test_online_without_warmup():
    data = np.random.default_rng(4).normal(size=5_000)
    om = OnlineMetalog(warmup=0, rng=0).update(data)
    assert om.is_feasible()
    assert abs(om.quantile(0.5)) < 0.1


def test_online_is_reproducible_with_seed():
    data = np.random.default_rng(5).normal(size=500)
    a = OnlineMetalog(rng=7).update(data).a
    b = OnlineMetalog(rng=7).update(data).a
    np.testing.assert_array_equal(a, b)


def test_online_rejects_observations_outside_bounds():
    om = OnlineMetalog(bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="strictly inside"):
        om.update([0.5, 1.0])
    with pytest.raises(ValueError):
        om.update(float('nan'))


@pytest.mark.parametrize('kwargs', [dict(n_terms=1), dict(lr=0), dict(decay=1.5), dict(n_taus=0), dict(warmup=-1)])
def test_online_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        OnlineMetalog(**kwargs)
