'use strict';

// §09 pyhf helpers. Reuse particle-physics interpolation and evaluate the
// yield sums/products directly. Outer broadcasts belong to ordinary dispatch.
const T = require('./types.ts');
const V = require('./value.ts');
type Value = import('./engine-types').Value;

/** §03 keeps flat tensors distinct from nested arrays. Resolve storage views
 * only after checking that each argument is one real-valued model cell. */
function realArray(value: unknown, rank: number, name: string): Value {
  const array: Value = V.asValue(value);
  if (array.shape.length !== rank || V.outerRankOf(array) !== rank || V.isComplexValue(array)) {
    throw new Error(`pyhf_helpers.${name}: expected a real rank-${rank} array`);
  }
  return V._logicalDense(array);
}

/** §09 sample_yields: sum additive shifts before multiplying factors.
 * Empty modifier axes use 0/1; never divide by a factor or clip a yield. */
function sampleYields(n: unknown, a: unknown, m: unknown): Value {
  const nominal = realArray(n, 2, 'sample_yields nominal');
  const shifts = realArray(a, 3, 'sample_yields shifts');
  const factors = realArray(m, 3, 'sample_yields factors');
  const [samples, bins] = nominal.shape;
  if (shifts.shape[0] !== samples || factors.shape[0] !== samples
      || shifts.shape[2] !== bins || factors.shape[2] !== bins) {
    throw new Error('pyhf_helpers.sample_yields: sample and bin extents must agree');
  }
  const additive = shifts.shape[1], multiplicative = factors.shape[1];
  const data = new Float64Array(samples * bins);
  for (let sample = 0; sample < samples; sample++) {
    for (let bin = 0; bin < bins; bin++) {
      let sum = 0, product = 1;
      for (let i = 0; i < additive; i++) sum += shifts.data[(sample * additive + i) * bins + bin];
      for (let i = 0; i < multiplicative; i++) product *= factors.data[(sample * multiplicative + i) * bins + bin];
      const index = sample * bins + bin;
      data[index] = (nominal.data[index] + sum) * product;
    }
  }
  return { shape: [samples, bins], data };
}

/** §09 expected_counts: reduce only the sample axis, retaining every bin.
 * Float64 storage promotes Boolean/integer values before arithmetic. */
function expectedCounts(value: unknown): Value {
  const samples = realArray(value, 2, 'expected_counts');
  const [count, bins] = samples.shape;
  const data = new Float64Array(bins);
  for (let sample = 0; sample < count; sample++) {
    for (let bin = 0; bin < bins; bin++) data[bin] += samples.data[sample * bins + bin];
  }
  return { shape: [bins], data };
}

/** Unit-centered HistFactory code4, matching the FlatPPL reference definition:
 * normsys_factor(lo, hi, alpha) = hep.interp_poly6_exp(lo, 1.0, hi, alpha).
 * Ordinary call dispatch and broadcast handle argument binding and axes. */
function pyhfModule(
  interpPoly6Exp: (lo: number, nominal: number, hi: number, alpha: number) => number,
  interpPoly6Lin: (lo: number, nominal: number, hi: number, alpha: number) => number,
) {
  const matrix = T.array(2, ['%dynamic', '%dynamic'], T.REAL);
  const modifiers = T.array(3, ['%dynamic', '%dynamic', '%dynamic'], T.REAL);
  return {
    name: 'pyhf_helpers',
    compat: '0.1',
    bindings: new Map([
      ['normsys_factor', {
        kind: 'function' as const,
        sig: T.funcType([
          { name: 'lo', type: T.REAL },
          { name: 'hi', type: T.REAL },
          { name: 'alpha', type: T.REAL },
        ], T.REAL),
        impl: (lo: number, hi: number, alpha: number) => interpPoly6Exp(lo, 1, hi, alpha),
      }],
      ['histosys_shift', {
        kind: 'function' as const,
        sig: T.funcType([
          { name: 'lo', type: T.REAL },
          { name: 'nominal', type: T.REAL },
          { name: 'hi', type: T.REAL },
          { name: 'alpha', type: T.REAL },
        ], T.REAL),
        impl: (lo: number, nominal: number, hi: number, alpha: number) =>
          interpPoly6Lin(lo - nominal, 0, hi - nominal, alpha),
      }],
      ['sample_yields', {
        kind: 'function' as const,
        sig: T.funcType([
          { name: 'nominal', type: matrix },
          { name: 'shifts', type: modifiers },
          { name: 'factors', type: modifiers },
        ], matrix),
        impl: sampleYields,
      }],
      ['expected_counts', {
        kind: 'function' as const,
        sig: T.funcType([{ name: 'samples', type: matrix }], T.array(1, ['%dynamic'], T.REAL)),
        impl: expectedCounts,
      }],
    ]),
  };
}

module.exports = { pyhfModule };
