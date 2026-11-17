'use strict';

// Real-order Bessel functions, §09. The convergent series and real integrals
// are DLMF 10.2.2, 10.9.6–7, 10.25.2, 10.32.9. Hankel's expansions (10.17,
// 10.40) avoid resolving a rapidly oscillating integral at large arguments.
// J/I continue negative real arguments on the principal complex branch.
const V = require('./value.ts');
const logGamma = (x: number): number => require('./value-ops.ts').loggammaElem(V.scalar(x)).data[0];
const PI = Math.PI;

function sinpi(x: number): number {
  const n = Math.round(x);
  return (n % 2 === 0 ? 1 : -1) * Math.sin(PI * (x - n));
}
function cospi(x: number): number {
  const n = Math.round(x);
  if (Math.abs(x - n) === 0.5) return 0;
  return (n % 2 === 0 ? 1 : -1) * Math.cos(PI * (x - n));
}

const { integral } = require('./scalar-quadrature.ts');

/** The positive series is accumulated with rescaling, before its log prefactor. */
function series(v: number, x: number, modified: boolean): number {
  let term = 1, sum = 1, scale = 0;
  for (let k = 1; k < 100000; k++) {
    term *= (modified ? 1 : -1) * (x / 2 / k) * (x / 2 / (v + k));
    sum += term;
    if (Math.abs(term) <= 2 * Number.EPSILON * Math.abs(sum)) {
      const log = v * Math.log(x / 2) - logGamma(v + 1) + Math.log(Math.abs(sum)) + scale;
      return Math.sign(sum) * Math.exp(log);
    }
    if (Math.abs(sum) > 1e200) { term *= 1e-200; sum *= 1e-200; scale += Math.log(1e200); }
  }
  throw new Error('Bessel series did not converge');
}

/** Optimal truncation stops before divergent asymptotic terms grow. */
function hankel(v: number, x: number): [number, number] {
  const mu = 4 * v * v;
  let p = 1, q = 0, term = 1, previous = Infinity;
  for (let k = 1; k < 100; k++) {
    term *= (mu - (2 * k - 1) ** 2) / (8 * k * x);
    if (Math.abs(term) > previous) break;
    previous = Math.abs(term);
    if (k % 2 === 0) p += (k % 4 === 0 ? 1 : -1) * term;
    else q += (k % 4 === 1 ? 1 : -1) * term;
    if (Math.abs(term) < Number.EPSILON) break;
  }
  const phase = x - v * PI / 2 - PI / 4, scale = Math.sqrt(2 / PI / x);
  return [scale * (Math.cos(phase) * p - Math.sin(phase) * q),
    scale * (Math.sin(phase) * p + Math.cos(phase) * q)];
}

function logcosh(x: number): number {
  const a = Math.abs(x);
  return a + Math.log1p(Math.exp(-2 * a)) - Math.LN2;
}

/** Integrate the nonoscillating tail after scaling near its saddle point. */
function tail(v: number, x: number, kind: 'k' | 'positive' | 'negative'): number {
  const logx = Math.log(x);
  const exponent = (t: number) => {
    const e = Math.exp(logx + t - Math.LN2), r = Math.exp(logx - t - Math.LN2);
    return kind === 'k' ? -(e + r) + logcosh(v * t)
      : -(e - r) + (kind === 'positive' ? v * t : -v * t);
  };
  const saddle = kind === 'negative' ? 0 : Math.asinh(v / x);
  const peak = Math.max(exponent(0), exponent(saddle));
  if (peak > Math.log(Number.MAX_VALUE)) return Infinity;
  let upper = Math.max(1, saddle + 1);
  while (exponent(upper) > peak - 45) upper += 1;
  const value = integral((t: number) => Math.exp(exponent(t) - peak), 0, upper, Math.ceil(upper / 2));
  return Math.exp(peak + Math.log(value));
}

function positiveJ(v: number, x: number): number {
  if (x === 0) return v === 0 ? 1 : 0;
  if (x < 10 || x * x < 8 * (v + 1)) return series(v, x, false);
  if (x > Math.max(40, 2 * v * v)) return hankel(v, x)[0];
  if (v > 8 && v > x / 2) return millerJ(v, x);
  const oscillating = integral((t: number) => Math.cos(v * t - x * Math.sin(t)), 0, PI,
    Math.max(1, Math.ceil((x + v) / 8)));
  return (oscillating - sinpi(v) * tail(v, x, 'negative')) / PI;
}

/** Miller recursion avoids subtracting two O(1) integrals for a tiny Jν. */
function millerJ(v: number, x: number): number {
  const order = Math.floor(v), fraction = v - order;
  const j0 = positiveJ(fraction, x), j1 = positiveJ(fraction + 1, x);
  function recur(top: number): number {
    let next = 0, cur = 1, selected = 0, b1 = 0;
    for (let n = top; n > 0; n--) {
      if (n === order) selected = cur;
      const prev = 2 * (n + fraction) / x * cur - next;
      next = cur; cur = prev;
      if (Math.abs(cur) > 1e150) { cur *= 1e-150; next *= 1e-150; selected *= 1e-150; }
      if (n === 1) b1 = next;
    }
    // Either adjacent normalization point may lie near a zero.
    return Math.abs(cur) > Math.abs(b1) ? selected / cur * j0 : selected / b1 * j1;
  }
  let top = Math.ceil(Math.max(v, x)) + 32 + Math.ceil(Math.sqrt(40 * x));
  let previous = recur(top);
  for (let k = 0; k < 8; k++) {
    top += 16;
    const value = recur(top);
    // Oscillatory zeros are conditioned in absolute error: rounding the
    // normalization can exceed a near-zero result's relative precision.
    // Below the turning point retain relative precision for tiny Jν values.
    const scale = x > v ? Math.max(Math.abs(value), Math.abs(j0), Math.abs(j1)) : Math.abs(value);
    if (Math.abs(value - previous) <= 64 * Number.EPSILON * scale) return value;
    previous = value;
  }
  throw new Error('Bessel backward recurrence did not converge');
}

function positiveY(v: number, x: number): number {
  if (x === 0) return -Infinity;
  if (x > Math.max(40, 2 * v * v)) return hankel(v, x)[1];
  const oscillating = integral((t: number) => Math.sin(x * Math.sin(t) - v * t), 0, PI,
    Math.max(1, Math.ceil((x + v) / 8)));
  return (oscillating - tail(v, x, 'positive') - cospi(v) * tail(v, x, 'negative')) / PI;
}

function positiveI(v: number, x: number): number {
  if (x === 0) return v === 0 ? 1 : 0;
  if (x <= Math.max(40, 2 * v * v)) return series(v, x, true);
  let term = 1, sum = 1, previous = Infinity;
  for (let k = 1; k < 100; k++) {
    term *= -(4 * v * v - (2 * k - 1) ** 2) / (8 * k * x);
    if (Math.abs(term) > previous) break;
    previous = Math.abs(term); sum += term;
    if (Math.abs(term) < Number.EPSILON) break;
  }
  return Math.exp(x - Math.log(2 * PI * x) / 2 + Math.log(sum));
}

function besselY(v: number, x: number): number {
  if (!(x >= 0) || !Number.isFinite(v)) return NaN;
  if (x === Infinity) return 0;
  if (v >= 0) return positiveY(v, x);
  if (Number.isInteger(v)) return cospi(v) * positiveY(-v, x);
  if (cospi(v) === 0) return -sinpi(v) * positiveJ(-v, x);
  return cospi(v) * positiveY(-v, x) - sinpi(v) * positiveJ(-v, x);
}

function besselK(v: number, x: number): number {
  if (!(x >= 0) || !Number.isFinite(v)) return NaN;
  if (x === 0) return Infinity;
  if (x === Infinity) return 0;
  return tail(Math.abs(v), x, 'k');
}

/** Principal continuation through arg(z)=+π for the negative real axis. */
function besselJI(v: number, x: number, modified: boolean): number | { re: number; im: number } {
  if (Number.isNaN(x) || !Number.isFinite(v)) return NaN;
  if (x < 0) {
    const magnitude = besselJI(v, -x, modified) as number;
    return Number.isInteger(v) ? cospi(v) * magnitude
      : { re: cospi(v) * magnitude, im: sinpi(v) * magnitude };
  }
  if (x === Infinity) return modified ? Infinity : 0;
  const n = Math.abs(v), base = modified ? positiveI(n, x) : positiveJ(n, x);
  if (v >= 0) return base;
  if (Number.isInteger(v)) return modified ? base : cospi(v) * base;
  return modified ? base - 2 / PI * sinpi(v) * besselK(n, x)
    : cospi(v) * base + sinpi(v) * positiveY(n, x);
}

function airy(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x === 0) return Math.exp(-2 / 3 * Math.log(3) - logGamma(2 / 3));
  if (x === Infinity || x === -Infinity) return 0;
  const z = 2 / 3 * Math.abs(x) ** 1.5;
  if (x > 0) return Math.sqrt(x / 3) / PI * besselK(1 / 3, z);
  return Math.sqrt(-x) / 3 * ((besselJI(-1 / 3, z, false) as number) + positiveJ(1 / 3, z));
}

module.exports = { besselJ: (v: number, x: number) => besselJI(v, x, false),
  besselI: (v: number, x: number) => besselJI(v, x, true), besselY, besselK, airy };
