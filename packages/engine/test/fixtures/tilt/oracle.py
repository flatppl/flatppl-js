"""Independent analytic and SciPy oracles. No FlatPPL engine is imported."""

import json
import math

import numpy as np
from scipy.integrate import dblquad, quad
from scipy.special import digamma, gammainc, gammaln
from scipy.stats import gamma, multivariate_normal, norm


def normal_condition(mu, sigma):
    variance = np.square(sigma)
    total = variance.sum()
    mean = mu - variance * mu.sum() / total
    covariance = np.diag(variance) - np.outer(variance, variance) / total
    return mean, covariance


def gamma_order_moments(shape):
    """Integrate the marginal and pair densities of three order statistics."""
    def pdf(x):
        return math.exp((shape - 1) * math.log(x) - x - gammaln(shape)) if x else 0.0

    def cdf(x):
        return gammainc(shape, x)

    moments = np.array([
        [quad(lambda x: x**power * coefficient * pdf(x) * cdf(x)**i * (1-cdf(x))**(2-i),
              0, np.inf, epsabs=1e-10)[0] for power in (1, 2)]
        for i, coefficient in enumerate((3, 6, 3))
    ])
    mean = moments[:, 0]
    raw = np.diag(moments[:, 1])
    for i in range(3):
        for j in range(i+1, 3):
            def pair(x, y):
                return 6*x*y*pdf(x)*pdf(y)*cdf(x)**i*(cdf(y)-cdf(x))**(j-i-1)*(1-cdf(y))**(2-j)
            raw[i, j] = raw[j, i] = dblquad(pair, 0, np.inf, 0, lambda y: y,
                                          epsabs=1e-8, epsrel=1e-8)[0]
    covariance = raw - np.outer(mean, mean)
    assert abs(mean.sum() - 3*shape) < 1e-8
    assert abs(covariance.sum() - 3*shape) < 1e-7
    return mean, covariance


def derive():
    x = np.array([0.1, 0.5, 2.0])
    exponential = -x.sum() + math.log(6)
    increments = 1 / np.arange(3, 0, -1, dtype=float)
    exp_covariance = np.minimum.outer(
        np.cumsum(increments**2), np.cumsum(increments**2)
    )
    shape = 2.5
    gamma_mean, gamma_covariance = gamma_order_moments(shape)
    gamma_score = (shape - 1) * np.log(x).sum() - x.sum() - 3 * gammaln(shape) + math.log(6)
    assert abs(gamma_score - (gamma.logpdf(x, a=shape).sum() + math.log(6))) < 1e-12

    mu = np.array([1.0, -2.0, 0.5])
    sigma = np.array([1.0, 2.0, 0.5])
    point = np.array([0.3, -0.5, 0.2])
    mean, covariance = normal_condition(mu, sigma)
    variance = sigma**2
    total, location = variance.sum(), mu.sum()
    log_z = -0.5 * math.log(2 * math.pi * total) - location**2 / (2 * total)
    score = norm.logpdf(point, loc=mu, scale=sigma).sum() - log_z
    chart_score = multivariate_normal.logpdf(point[:2], mean=mean[:2], cov=covariance[:2, :2])
    assert abs(score - chart_score) < 1e-12

    def chart_density(second, first):
        values = np.array([first, second, -first - second])
        return math.exp(norm.logpdf(values, loc=mu, scale=sigma).sum())

    integral, error = dblquad(chart_density, -np.inf, np.inf, -np.inf, np.inf,
                             epsabs=1e-10, epsrel=1e-9)
    assert abs(integral - math.exp(log_z)) < 1e-9

    mu2, sigma2 = mu[:2], sigma[:2]
    log_z2 = norm.logpdf(0, loc=mu2.sum(), scale=np.linalg.norm(sigma2))
    normal_mass = quad(lambda y: math.exp(norm.logpdf([y, -y], mu2, sigma2).sum() - log_z2),
                       -np.inf, np.inf, epsabs=1e-10)[0]
    exp_mass = quad(lambda y: 2 * math.exp(-y) * (-math.expm1(-y)), 0, np.inf)[0]
    gamma_mass = quad(lambda y: 2 * gamma.pdf(y, a=shape) * gamma.cdf(y, a=shape),
                     0, np.inf, epsabs=1e-10)[0]
    assert np.allclose([normal_mass, exp_mass, gamma_mass], 1, atol=1e-9, rtol=0)
    return {
        "ordered_exponential": {
            "point": x.tolist(), "logdensity": float(exponential),
            "rate_gradient": float(3 - x.sum()),
            "mean": np.cumsum(increments).tolist(), "covariance": exp_covariance.tolist(),
        },
        "ordered_gamma": {
            "shape": shape, "rate": 1, "point": x.tolist(),
            "logdensity": float(gamma_score),
            "shape_gradient": float(np.log(x).sum() - 3 * digamma(shape)),
            "mean": gamma_mean.tolist(), "covariance": gamma_covariance.tolist(),
        },
        "zerosum_normal": {
            "mu": mu.tolist(), "sigma": sigma.tolist(), "point": point.tolist(),
            "logdensity": float(score), "chart_logdensity": float(chart_score),
            "log_normalizer": float(log_z), "numeric_normalizer": integral,
            "integration_error": error, "mean": mean.tolist(), "covariance": covariance.tolist(),
            "mu_gradient": ((point - mu) / variance + location / total).tolist(),
            "sigma_gradient": (((point - mu)**2 / variance - 1) / sigma
                               + sigma / total - location**2 * sigma / total**2).tolist(),
        },
        "unit_mass_n2": {"ordered_exponential": exp_mass, "ordered_gamma": gamma_mass,
                         "zerosum_normal": normal_mass},
    }


if __name__ == "__main__":
    print(json.dumps(derive(), indent=2, allow_nan=False))
