'use strict';

// §09 one-dimensional thin-plate basis and second-derivative penalty.
// Ordinary standard-module dispatch owns argument binding and outer axes.
const T = require('./types.ts');
const V = require('./value.ts');
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

function splinesModule() {
  const vector = T.array(1, ['%dynamic'], T.REAL);
  return {
    name: 'splines', compat: '0.1',
    bindings: new Map([
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
    ]),
  };
}

module.exports = { splinesModule };
