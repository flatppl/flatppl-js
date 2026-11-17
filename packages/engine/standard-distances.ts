'use strict';

// §09 distances: metrics consume complete vectors. The ordinary module
// dispatcher binds callable inputs, including user-defined distance functions.
const T = require('./types.ts');
const V = require('./value.ts');
type Value = import('./engine-types').Value;

function vector(value: unknown): Float64Array {
  const v: Value = V.asValue(value);
  if (v.shape.length !== 1 || V.isComplexValue(v)) {
    throw new Error('distances: expected a real vector');
  }
  return V._logicalDense(v).data;
}

/** Equal lengths are a metric-domain requirement, not broadcast expansion. */
function vectors(u: unknown, v: unknown): [Float64Array, Float64Array] {
  const a = vector(u), b = vector(v);
  if (a.length !== b.length) throw new Error('distances: vector lengths must agree');
  return [a, b];
}

/** Scaling before powers avoids overflow in finite Euclidean/Minkowski norms. */
function minkowski(u: unknown, v: unknown, p: number): number {
  if (!(p > 0)) throw new Error('distances.minkowski: p must be positive');
  const [a, b] = vectors(u, v);
  let scale = 0;
  for (let i = 0; i < a.length; i++) scale = Math.max(scale, Math.abs(a[i] - b[i]));
  if (scale === 0 || !Number.isFinite(scale)) return scale;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (Math.abs(a[i] - b[i]) / scale) ** p;
  return scale * sum ** (1 / p);
}

function squaredEuclidean(u: unknown, v: unknown): number {
  const distance = minkowski(u, v, 2);
  return distance * distance;
}

function manhattan(u: unknown, v: unknown): number {
  const [a, b] = vectors(u, v);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum;
}

function chebyshev(u: unknown, v: unknown): number {
  const [a, b] = vectors(u, v);
  let distance = 0;
  for (let i = 0; i < a.length; i++) distance = Math.max(distance, Math.abs(a[i] - b[i]));
  return distance;
}

/** Normalize each vector separately so large finite coordinates stay finite. */
function cosine(u: unknown, v: unknown): number {
  const [a, b] = vectors(u, v);
  let sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) {
    sa = Math.max(sa, Math.abs(a[i])); sb = Math.max(sb, Math.abs(b[i]));
  }
  if (sa === 0 || sb === 0) throw new Error('distances.cosine: vectors must have nonzero magnitude');
  let aa = 0, bb = 0, ab = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] / sa, y = b[i] / sb;
    aa += x * x; bb += y * y; ab += x * y;
  }
  return Math.max(0, Math.min(2, 1 - ab / Math.sqrt(aa * bb)));
}

/** §09 fixes the natural-log KL convention and the continuous 0 log 0 limit. */
function jensenShannon(u: unknown, v: unknown): number {
  const [a, b] = vectors(u, v);
  let sum = 0, totalA = 0, totalB = 0;
  for (let i = 0; i < a.length; i++) {
    if (!(a[i] >= 0) || !(b[i] >= 0)) throw new Error('distances.jensenshannon: expected probability vectors');
    totalA += a[i]; totalB += b[i];
    const mid = a[i] / 2 + b[i] / 2;
    if (mid === 0) continue;
    const t = (a[i] - b[i]) / (a[i] + b[i]);
    if (Math.abs(t) < 0.1) {
      // (1+t)log(1+t)+(1-t)log(1-t) has no linear term.
      // Its even series retains tiny distances between near-equal vectors.
      let power = t * t, g = power;
      for (let k = 2; k < 12; k++) { power *= t * t; g += power / (k * (2 * k - 1)); }
      sum += mid * g;
    } else {
      if (a[i] > 0) sum += a[i] * (Math.log(a[i]) - Math.log(mid));
      if (b[i] > 0) sum += b[i] * (Math.log(b[i]) - Math.log(mid));
    }
  }
  const tolerance = 16 * Number.EPSILON * Math.max(1, a.length);
  if (Math.abs(totalA - 1) > tolerance || Math.abs(totalB - 1) > tolerance) {
    throw new Error('distances.jensenshannon: expected probability vectors');
  }
  return Math.sqrt(Math.max(0, sum / 2));
}

/** A user distance need not be symmetric and need not vanish on the diagonal. */
function crossDistance(distance: (u: unknown, v: unknown) => unknown, x: unknown, y: unknown): Value {
  const xs = Array.isArray(x) && x.length === 0 ? { kind: 'js-array', items: x } : V.asVectorOfVectors(x);
  const ys = Array.isArray(y) && y.length === 0 ? { kind: 'js-array', items: y } : V.asVectorOfVectors(y);
  if (!xs || !ys) throw new Error('distances: expected vectors of vectors');
  const n = V.vovLength(xs), m = V.vovLength(ys), data = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      data[i * m + j] = V.asScalar(V.asValue(distance(V.vovRowAt(xs, i), V.vovRowAt(ys, j))));
    }
  }
  return { shape: [n, m], data };
}

function distancesModule() {
  const realVector = T.array(1, ['%dynamic'], T.REAL);
  const nested = T.array(1, ['%dynamic'], realVector);
  const metric = T.funcType([{ name: 'u', type: realVector }, { name: 'v', type: realVector }], T.REAL);
  const bindings = new Map<string, any>();
  for (const [name, impl] of [
    ['euclidean', (u: unknown, v: unknown) => minkowski(u, v, 2)],
    ['squared_euclidean', squaredEuclidean], ['manhattan', manhattan],
    ['chebyshev', chebyshev], ['cosine', cosine], ['jensenshannon', jensenShannon],
  ] as const) bindings.set(name, { kind: 'function', sig: metric, impl });
  bindings.set('minkowski', { kind: 'function', sig: T.funcType([
    ...metric.inputs, { name: 'p', type: T.REAL },
  ], T.REAL), impl: minkowski });
  const distanceInput = { name: 'distance', type: metric }, xInput = { name: 'x', type: nested };
  const matrix = T.array(2, ['%dynamic', '%dynamic'], T.REAL);
  bindings.set('pairwise_distance', { kind: 'function',
    sig: T.funcType([distanceInput, xInput], matrix),
    impl: (distance: (u: unknown, v: unknown) => unknown, x: unknown) => crossDistance(distance, x, x) });
  bindings.set('cross_distance', { kind: 'function',
    sig: T.funcType([distanceInput, xInput, { name: 'y', type: nested }], matrix), impl: crossDistance });
  return { name: 'distances', compat: '0.1', bindings };
}

module.exports = { distancesModule };
