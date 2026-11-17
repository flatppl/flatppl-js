'use strict';

// Stable tails, logarithms and scaled values for the special-functions module.
// The gamma/beta kernels share the probability-transform implementations;
// Bessel kernels accumulate their logarithms before any exponentiation.
const gamma = require('./incomplete-gamma.ts');
const beta = require('./incomplete-beta.ts');
const bessel = require('./bessel-real-order.ts');
const erfc = require('@stdlib/math-base-special-erfc');
const erfcx = require('@stdlib/math-base-special-erfcx');

function gammaDomain(a: number, x: number): void {
  if (!(a > 0) || !Number.isFinite(a) || !(x >= 0)) {
    throw new Error('Incomplete gamma requires finite a > 0 and x >= 0');
  }
}

function betaDomain(a: number, b: number): void {
  if (!(a > 0 && b > 0) || !Number.isFinite(a) || !Number.isFinite(b)) {
    throw new Error('Beta requires finite a > 0 and b > 0');
  }
}

function incompleteBetaDomain(a: number, b: number, x: number): void {
  betaDomain(a, b);
  if (!(x >= 0 && x <= 1)) throw new Error('Incomplete beta requires 0 <= x <= 1');
}

function besselDomain(v: number, z: number, logI = false): void {
  if (!Number.isFinite(v) || !Number.isFinite(z) || (logI && !(v >= 0)) || !(z >= 0)) {
    throw new Error(logI ? 'log_bessel_i requires finite v >= 0 and z >= 0'
      : 'Modified Bessel K requires finite v and z >= 0');
  }
}

/** The erfcx identity retains normal log tails beyond probability underflow.
 * For positive x, log1p keeps the small departure from a CDF of one. */
function logNdtr(x: number): number {
  if (Number.isNaN(x)) throw new Error('log_ndtr requires a real argument');
  if (x < 0) return Math.log(erfcx(-x / Math.SQRT2)) - Math.LN2 - (x / 2) * x;
  return Math.log1p(-erfc(x / Math.SQRT2) / 2);
}

module.exports = {
  gammaincc(a: number, x: number): number {
    gammaDomain(a, x); return gamma.regularizedGamma(a, x, true);
  },
  log_gammainc(a: number, x: number): number {
    gammaDomain(a, x); return gamma.logRegularizedGamma(a, x);
  },
  log_gammaincc(a: number, x: number): number {
    gammaDomain(a, x); return gamma.logRegularizedGamma(a, x, true);
  },
  betaincc(a: number, b: number, x: number): number {
    incompleteBetaDomain(a, b, x); return beta.regularizedBeta(a, b, x, true);
  },
  log_betainc(a: number, b: number, x: number): number {
    incompleteBetaDomain(a, b, x); return beta.logRegularizedBeta(a, b, x);
  },
  log_betaincc(a: number, b: number, x: number): number {
    incompleteBetaDomain(a, b, x); return beta.logRegularizedBeta(a, b, x, true);
  },
  beta(a: number, b: number): number {
    betaDomain(a, b); return Math.exp(beta.logBeta(a, b));
  },
  logbeta(a: number, b: number): number {
    betaDomain(a, b); return beta.logBeta(a, b);
  },
  bessel_i_scaled(v: number, z: number): { re: number; im: number } {
    if (!Number.isFinite(v) || !Number.isFinite(z)) throw new Error('Modified Bessel I requires finite real v and z');
    const value = bessel.besselIScaled(v, z);
    return typeof value === 'number' ? { re: value, im: 0 } : value;
  },
  bessel_k_scaled(v: number, z: number): number {
    besselDomain(v, z); return bessel.besselKScaled(v, z);
  },
  log_bessel_i(v: number, z: number): number {
    besselDomain(v, z, true); return bessel.logBesselI(v, z);
  },
  log_bessel_k(v: number, z: number): number {
    besselDomain(v, z); return bessel.logBesselK(v, z);
  },
  log_ndtr: logNdtr,
};
