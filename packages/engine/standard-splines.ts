'use strict';

// Spline value functions: thin-plate coordinates, arbitrary-knot interpolation,
// and full-knot B-spline bases, derivatives, integrals and L2 penalties.
// Ordinary standard-module dispatch owns argument binding and outer axes.
const T = require('./types.ts');
const V = require('./value.ts');
const spline = require('./spline-numeric.ts');
type Value = import('./engine-types').Value;

function realCenters(value: unknown): Float64Array {
  const centers: Value = V.asValue(value);
  if (centers.shape.length !== 1 || V.outerRankOf(centers) !== 1
      || V.isComplexValue(centers) || centers.shape[0] === 0) {
    throw new Error('splines: centers must be a nonempty real vector');
  }
  return V._logicalDense(centers).data;
}

function thinplateBasis(x: unknown, value: unknown): Value {
  const point: Value = V.asValue(x);
  if (point.shape.length !== 0 || V.isComplexValue(point)) {
    throw new Error('splines.thinplate_basis: x must be a real scalar');
  }
  const centers = realCenters(value), n = centers.length;
  const scalar = point.data[0], data = new Float64Array(n + 2);
  for (let i = 0; i < n; i++) data[i] = Math.abs(scalar - centers[i]) ** 3 / 12;
  data[n] = 1;
  data[n + 1] = scalar;
  return { shape: [n + 2], data };
}

function thinplatePenalty(value: unknown): Value {
  const centers = realCenters(value), n = centers.length, size = n + 2;
  const data = new Float64Array(size * size);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const kernel = Math.abs(centers[i] - centers[j]) ** 3 / 12;
      data[i * size + j] = kernel;
      data[j * size + i] = kernel;
    }
  }
  return { shape: [size, size], data };
}

/** The interpolation and B-spline APIs require finite, flat real inputs. */
function realVector(value: unknown): Float64Array {
  const v: Value = V.asValue(value);
  if (v.shape.length !== 1 || V.outerRankOf(v) !== 1 || V.isComplexValue(v)) {
    throw new Error('splines: expected a finite real vector');
  }
  const data = V._logicalDense(v).data;
  if (!data.every(Number.isFinite)) throw new Error('splines: expected a finite real vector');
  return data;
}

/** Scalar controls are checked here so direct and sampled calls agree. */
function realScalar(value: unknown): number {
  const v: Value = V.asValue(value);
  if (v.shape.length !== 0 || V.isComplexValue(v) || !Number.isFinite(v.data[0])) {
    throw new Error('splines: expected a finite real scalar');
  }
  return v.data[0];
}

function nonnegativeInteger(value: unknown): number {
  const n = realScalar(value);
  if (!Number.isSafeInteger(n) || n < 0) throw new Error('splines: expected a nonnegative integer');
  return n;
}

/** Interpolation ordinates are scalar values at strictly increasing abscissae. */
function interpolation(method: string, x: unknown, knots: unknown, values: unknown,
  extrapolation: unknown = 'constant'): number {
  const point = realScalar(x), t = realVector(knots), y = realVector(values);
  if (t.length < 2 || t.length !== y.length || !t.every((v, i) => i === 0 || v > t[i - 1])) {
    throw new Error('splines: interpolation needs equal-length vectors and at least two strictly increasing knots');
  }
  if (!['constant', 'linear', 'error'].includes(extrapolation as string)) {
    throw new Error('splines: extrapolation must be constant, linear or error');
  }
  return spline.interpolate(point, t, y, method, extrapolation);
}

/** A full knot vector supplies the degree+1 exterior knots on either side. */
function bsplineInputs(knots: unknown, degree: unknown): [Float64Array, number] {
  const t = realVector(knots), p = nonnegativeInteger(degree), n = t.length - p - 1;
  if (t.length < 2 * p + 2 || !(t[p] < t[n])
      || !t.every((v, i) => i === 0 || v >= t[i - 1])) {
    throw new Error('splines: expected nondecreasing full knots with a nonempty B-spline base interval');
  }
  let multiplicity = 1;
  for (let i = 1; i < t.length; i++) {
    multiplicity = t[i] === t[i - 1] ? multiplicity + 1 : 1;
    if (multiplicity > p + 1) throw new Error('splines: knot multiplicity exceeds degree + 1');
  }
  return [t, p];
}

/** A genuine L2 derivative requires continuity through order-1 at interior knots. */
function roughness(knots: unknown, degree: unknown, order: unknown = 2): Value {
  const [t, p] = bsplineInputs(knots, degree), r = nonnegativeInteger(order), n = t.length - p - 1;
  if (r > p) throw new Error('splines: penalty order must not exceed degree');
  let multiplicity = 0;
  for (let i = 0; i < t.length; i++) {
    multiplicity = i > 0 && t[i] === t[i - 1] ? multiplicity + 1 : 1;
    if (t[i] > t[p] && t[i] < t[n] && multiplicity > p - r + 1) {
      throw new Error('splines: interior knot multiplicity does not admit the requested L2 derivative');
    }
  }
  return { shape: [n, n], data: spline.bsplinePenalty(t, p, r) };
}

function splinesModule() {
  const vector = T.array(1, ['%dynamic'], T.REAL);
  const matrix = T.array(2, ['%dynamic', '%dynamic'], T.REAL);
  const bindings = new Map<string, any>([
      ['thinplate_basis', {
        kind: 'function' as const,
        sig: T.funcType([{ name: 'x', type: T.REAL }, { name: 'centers', type: vector }], vector),
        impl: thinplateBasis,
      }],
      ['thinplate_penalty', {
        kind: 'function' as const,
        sig: T.funcType([{ name: 'centers', type: vector }],
          T.array(2, ['%dynamic', '%dynamic'], T.REAL)),
        impl: thinplatePenalty,
      }],
    ]);
  const xInput = { name: 'x', type: T.REAL }, knotsInput = { name: 'knots', type: vector };
  const degreeInput = { name: 'degree', type: T.INTEGER, fixed: true };
  for (const method of ['linear', 'cubic', 'pchip']) {
    bindings.set('interp_' + method, { kind: 'function', sig: T.funcType([
      xInput, knotsInput, { name: 'values', type: vector },
      { name: 'extrapolation', type: T.STRING, optional: true, fixed: true },
    ], T.REAL), impl: (x: unknown, knots: unknown, values: unknown, extrapolation?: unknown) =>
      interpolation(method, x, knots, values, extrapolation) });
  }
  bindings.set('bspline_basis', { kind: 'function', sig: T.funcType([xInput, knotsInput, degreeInput], vector),
    impl: (x: unknown, knots: unknown, degree: unknown) => {
      const [t, p] = bsplineInputs(knots, degree), data = spline.bsplineValues(realScalar(x), t, p);
      return { shape: [data.length], data };
    } });
  bindings.set('bspline_derivative', { kind: 'function', sig: T.funcType([
    xInput, knotsInput, degreeInput, { name: 'order', type: T.INTEGER, optional: true, fixed: true },
  ], vector), impl: (x: unknown, knots: unknown, degree: unknown, order: unknown = 1) => {
    const [t, p] = bsplineInputs(knots, degree);
    const data = spline.bsplineValues(realScalar(x), t, p, nonnegativeInteger(order));
    return { shape: [data.length], data };
  } });
  bindings.set('bspline_integral', { kind: 'function', sig: T.funcType([
    { name: 'a', type: T.REAL }, { name: 'b', type: T.REAL }, knotsInput, degreeInput,
  ], vector), impl: (a: unknown, b: unknown, knots: unknown, degree: unknown) => {
    const [t, p] = bsplineInputs(knots, degree), data = spline.bsplineIntegral(realScalar(a), realScalar(b), t, p);
    return { shape: [data.length], data };
  } });
  bindings.set('bspline_penalty', { kind: 'function', sig: T.funcType([
    knotsInput, degreeInput, { name: 'order', type: T.INTEGER, optional: true, fixed: true },
  ], matrix), impl: roughness });
  return { name: 'splines', compat: '0.1', bindings };
}

module.exports = { splinesModule };
