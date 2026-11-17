'use strict';

// Pure numerical leaves for the splines module. Knots and ordinates are checked
// at the value boundary; these routines operate on finite dense real vectors.

/** Natural end conditions determine the tridiagonal system of second derivatives. */
function naturalSlopes(knots: Float64Array, values: Float64Array): Float64Array {
  const n = knots.length, diagonal = new Float64Array(n), rhs = new Float64Array(n);
  const h = Float64Array.from({ length: n - 1 }, (_, i) => knots[i + 1] - knots[i]);
  const secant = Float64Array.from(h, (width, i) => (values[i + 1] - values[i]) / width);
  for (let i = 1; i < n - 1; i++) {
    diagonal[i] = 2 * (h[i - 1] + h[i]);
    rhs[i] = 6 * (secant[i] - secant[i - 1]);
    if (i > 1) {
      const factor = h[i - 1] / diagonal[i - 1];
      diagonal[i] -= factor * h[i - 1];
      rhs[i] -= factor * rhs[i - 1];
    }
  }
  const second = new Float64Array(n), slopes = new Float64Array(n);
  for (let i = n - 2; i > 0; i--) second[i] = (rhs[i] - h[i] * second[i + 1]) / diagonal[i];
  for (let i = 0; i < n - 1; i++) {
    slopes[i] = secant[i] - h[i] * (2 * second[i] + second[i + 1]) / 6;
  }
  slopes[n - 1] = secant[n - 2] + h[n - 2] * (second[n - 2] + 2 * second[n - 1]) / 6;
  return slopes;
}

/** PCHIP uses weighted harmonic interior slopes and limited one-sided endpoints. */
function pchipSlopes(knots: Float64Array, values: Float64Array): Float64Array {
  const n = knots.length, h = new Float64Array(n - 1), d = new Float64Array(n - 1);
  const slopes = new Float64Array(n);
  for (let i = 0; i < n - 1; i++) {
    h[i] = knots[i + 1] - knots[i]; d[i] = (values[i + 1] - values[i]) / h[i];
  }
  if (n === 2) { slopes.fill(d[0]); return slopes; }
  function endpoint(h0: number, h1: number, d0: number, d1: number): number {
    const ratio = h0 / (h0 + h1);
    const slope = (1 + ratio) * d0 - ratio * d1;
    if (Math.sign(slope) !== Math.sign(d0)) return 0;
    return Math.sign(d0) !== Math.sign(d1) && Math.abs(slope) > 3 * Math.abs(d0) ? 3 * d0 : slope;
  }
  slopes[0] = endpoint(h[0], h[1], d[0], d[1]);
  slopes[n - 1] = endpoint(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] === 0 || Math.sign(d[i - 1]) !== Math.sign(d[i])) continue;
    const weight = (1 + h[i] / (h[i - 1] + h[i])) / 3;
    // Scale by the smaller secant to keep the harmonic mean finite.
    const scale = Math.min(Math.abs(d[i - 1]), Math.abs(d[i]));
    slopes[i] = scale / (weight * (scale / d[i - 1]) + (1 - weight) * (scale / d[i]));
  }
  return slopes;
}

/** Right-hand interval at an interior knot; the last interval includes its end. */
function interval(x: number, knots: Float64Array): number {
  let lo = 0, hi = knots.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >>> 1;
    if (x < knots[mid]) hi = mid; else lo = mid;
  }
  return Math.min(lo, knots.length - 2);
}

/** Hermite evaluation also gives natural-cubic and PCHIP endpoint tangents. */
function interpolate(x: number, knots: Float64Array, values: Float64Array,
  method: string, extrapolation: string): number {
  const n = knots.length, slopes = method === 'cubic' ? naturalSlopes(knots, values)
    : method === 'pchip' ? pchipSlopes(knots, values) : null;
  if (x < knots[0] || x > knots[n - 1]) {
    if (extrapolation === 'error') throw new Error('splines: x is outside the interpolation domain');
    const i = x < knots[0] ? 0 : n - 1;
    if (extrapolation === 'constant') return values[i];
    const j = i === 0 ? 0 : n - 2;
    const slope = slopes ? slopes[i] : (values[j + 1] - values[j]) / (knots[j + 1] - knots[j]);
    return values[i] + (x - knots[i]) * slope;
  }
  const i = interval(x, knots), h = knots[i + 1] - knots[i], t = (x - knots[i]) / h;
  if (!slopes) return (1 - t) * values[i] + t * values[i + 1];
  return (1 - t) * values[i] + t * values[i + 1]
    + t * (1 - t) * ((1 - t) * (h * slopes[i] - (values[i + 1] - values[i]))
      - t * (h * slopes[i + 1] - (values[i + 1] - values[i])));
}

/** Cox-de Boor and its differentiated recurrence, with zero-width terms omitted. */
function bsplineValues(x: number, knots: Float64Array, degree: number, order = 0): Float64Array {
  const n = knots.length - degree - 1, out = new Float64Array(n);
  if (order > degree || x < knots[degree] || x > knots[n]) return out;
  let base = new Float64Array(knots.length - 1);
  for (let i = 0; i < base.length; i++) {
    if (x === knots[n] ? knots[i] < x && x <= knots[i + 1]
      : knots[i] <= x && x < knots[i + 1]) base[i] = 1;
  }
  let previous: Float64Array[] = [base];
  for (let p = 1; p <= degree; p++) {
    const current: Float64Array[] = [];
    for (let d = 0; d <= Math.min(p, order); d++) {
      const row = new Float64Array(knots.length - p - 1);
      for (let i = 0; i < row.length; i++) {
        const left = knots[i + p] - knots[i], right = knots[i + p + 1] - knots[i + 1];
        if (d === 0) {
          if (left > 0) row[i] += (x - knots[i]) / left * previous[0][i];
          if (right > 0) row[i] += (knots[i + p + 1] - x) / right * previous[0][i + 1];
        } else {
          if (left > 0) row[i] += p / left * previous[d - 1][i];
          if (right > 0) row[i] -= p / right * previous[d - 1][i + 1];
        }
      }
      current.push(row);
    }
    previous = current;
  }
  return previous[order];
}

/** Gauss-Legendre integrates each polynomial span exactly through degree 2n-1. */
function gaussLegendre(n: number): [Float64Array, Float64Array] {
  const nodes = new Float64Array(n), weights = new Float64Array(n);
  for (let i = 0; i < Math.ceil(n / 2); i++) {
    let x = Math.cos(Math.PI * (i + .75) / (n + .5)), derivative = 0;
    for (let iteration = 0; iteration < 64; iteration++) {
      let p0 = 1, p1 = x;
      for (let j = 2; j <= n; j++) { const p = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j; p0 = p1; p1 = p; }
      derivative = n * (x * p1 - p0) / (x * x - 1);
      const step = p1 / derivative;
      if (Math.abs(step) <= 4 * Number.EPSILON) break;
      x -= step;
    }
    nodes[i] = -x; nodes[n - 1 - i] = x;
    weights[i] = weights[n - 1 - i] = 2 / ((1 - x * x) * derivative * derivative);
  }
  return [nodes, weights];
}

/** Integrate only the base domain; repeated knots delimit empty spans. */
function bsplineIntegral(a: number, b: number, knots: Float64Array, degree: number): Float64Array {
  const n = knots.length - degree - 1, data = new Float64Array(n);
  const sign = a <= b ? 1 : -1, lo = Math.max(Math.min(a, b), knots[degree]);
  const hi = Math.min(Math.max(a, b), knots[n]);
  const [nodes, weights] = gaussLegendre(Math.ceil((degree + 1) / 2));
  for (let span = degree; span < n; span++) {
    const left = Math.max(lo, knots[span]), right = Math.min(hi, knots[span + 1]);
    if (left >= right) continue;
    const half = (right - left) / 2;
    for (let q = 0; q < nodes.length; q++) {
      const basis = bsplineValues(left + half * (1 + nodes[q]), knots, degree);
      for (let i = 0; i < n; i++) data[i] += sign * half * weights[q] * basis[i];
    }
  }
  return data;
}

/** Derivative Gram matrix; the module boundary checks Sobolev continuity. */
function bsplinePenalty(knots: Float64Array, degree: number, order: number): Float64Array {
  const n = knots.length - degree - 1, data = new Float64Array(n * n);
  const [nodes, weights] = gaussLegendre(degree - order + 1);
  for (let span = degree; span < n; span++) {
    const left = knots[span], half = (knots[span + 1] - left) / 2;
    if (half === 0) continue;
    for (let q = 0; q < nodes.length; q++) {
      const derivative = bsplineValues(left + half * (1 + nodes[q]), knots, degree, order);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) data[i * n + j] += half * weights[q] * derivative[i] * derivative[j];
      }
    }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) data[j * n + i] = data[i * n + j];
  return data;
}

module.exports = { interpolate, bsplineValues, bsplineIntegral, bsplinePenalty };
