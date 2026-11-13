'use strict';

// Distribution parameters outside their declared value sets (spec §08, §09).
// =====================================================================
//
// Each §08 and §09 parameter is declared over a value set, for example
// `sigma = elementof(posreals)`. A density, CDF or sampler outside that set has
// no meaning, and its closed form returns NaN or a plausible number. So a value
// outside its set raises an error that names the distribution and the
// parameter. The StableHLO emitter reports the same condition as status 9.
//
// NaN propagates without an error, as the StableHLO status does: every test is
// an ordered comparison, which NaN fails. A `reals` parameter therefore needs no
// entry. A value outside the SUPPORT of the variate is not an error: its
// density is zero.

const valueLib = require('./value.ts');

type DomainSet = 'posreals' | 'nonnegreals' | 'unitinterval' | 'posintegers'
  | 'stdsimplex' | 'posdefinite';

// Parameters over `reals` and set-valued parameters have no entry.
const PARAM_SETS: Record<string, Record<string, DomainSet>> = {
  // §08
  Normal: { sigma: 'posreals' },
  GeneralizedNormal: { alpha: 'posreals', beta: 'posreals' },
  Cauchy: { scale: 'posreals' },
  StudentT: { nu: 'posreals' },
  Logistic: { s: 'posreals' },
  LogNormal: { sigma: 'posreals' },
  Exponential: { rate: 'posreals' },
  Gamma: { shape: 'posreals', rate: 'posreals' },
  Weibull: { shape: 'posreals', scale: 'posreals' },
  Rayleigh: { scale: 'posreals' },
  Pareto: { shape: 'posreals', scale: 'posreals' },
  InverseGamma: { shape: 'posreals', scale: 'posreals' },
  Beta: { alpha: 'posreals', beta: 'posreals' },
  ChiSquared: { k: 'posreals' },
  VonMises: { kappa: 'posreals' },
  Laplace: { scale: 'posreals' },
  Bernoulli: { p: 'unitinterval' },
  Categorical: { p: 'stdsimplex' },
  Categorical0: { p: 'stdsimplex' },
  Binomial: { n: 'posintegers', p: 'unitinterval' },
  Geometric: { p: 'unitinterval' },
  NegativeBinomial: { alpha: 'posreals', beta: 'posreals' },
  NegativeBinomial2: { mu: 'posreals', psi: 'posreals' },
  Poisson: { rate: 'nonnegreals' },
  MvNormal: { cov: 'posdefinite' },
  Wishart: { nu: 'posreals', scale: 'posdefinite' },
  InverseWishart: { nu: 'posreals', scale: 'posdefinite' },
  LKJ: { n: 'posintegers', eta: 'posreals' },
  LKJCholesky: { n: 'posintegers', eta: 'posreals' },
  Dirichlet: { alpha: 'posreals' },
  Multinomial: { n: 'posintegers', p: 'stdsimplex' },
  // §09 particle-physics
  CrystalBall: { sigma: 'posreals', alpha: 'posreals', n: 'posreals' },
  DoubleSidedCrystalBall: {
    sigmaL: 'posreals', sigmaR: 'posreals', alphaL: 'posreals',
    alphaR: 'posreals', nL: 'posreals', nR: 'posreals',
  },
  Argus: { resonance: 'posreals', power: 'posreals' },
  RelativisticBreitWigner: { mean: 'posreals', width: 'posreals' },
  Voigtian: { width: 'posreals', sigma: 'posreals' },
  Landau: { scale: 'posreals' },
  BifurcatedNormal: { sigmaL: 'posreals', sigmaR: 'posreals' },
  ContinuedPoisson: { rate: 'nonnegreals' },
};

const SECTION_09 = new Set([
  'CrystalBall', 'DoubleSidedCrystalBall', 'Argus', 'RelativisticBreitWigner',
  'Voigtian', 'Landau', 'BifurcatedNormal', 'ContinuedPoisson',
]);

// Vector and matrix values already checked. A probability vector is scored
// once per observation, and its check is linear in its length. Raw typed
// arrays are evaluation buffers that may be reused, so they are never cached.
const checkedValues = new WeakSet<object>();

/** The numbers of `v` in row-major order: a number, a boolean, a tensor
 *  Value, a typed array, or a nested array. */
function flatten(v: any, out: number[]): number[] {
  if (typeof v === 'number') out.push(v);
  else if (typeof v === 'boolean') out.push(+v);
  else if (valueLib.isValue(v)) for (let i = 0; i < v.data.length; i++) out.push(+v.data[i]);
  else if (v != null && typeof v === 'object' && typeof v.length === 'number') {
    for (let i = 0; i < v.length; i++) flatten(v[i], out);
  }
  return out;
}

/** Length of the innermost axis, the simplex dimension. */
function innerLength(v: any): number {
  if (valueLib.isValue(v)) return v.shape.length ? v.shape[v.shape.length - 1] : 1;
  if (Array.isArray(v) && v.length && Array.isArray(v[0])) return innerLength(v[0]);
  return v != null && typeof v.length === 'number' ? v.length : 1;
}

/** The relative rounding allowance on the sum of an `n`-cell probability
 *  vector, `8 n eps`, as for `in(p, stdsimplex(n))`. It scales with
 *  `max(sum, 1)`. */
function simplexTolerance(n: number): number {
  return 8 * Math.max(n, 1) * Number.EPSILON;
}

function violatesScalar(set: DomainSet, x: number): boolean {
  switch (set) {
    case 'posreals': return x <= 0;
    case 'nonnegreals': return x < 0;
    case 'unitinterval': return x < 0 || x > 1;
    case 'posintegers': return x < 1 || x - Math.floor(x) > 0 || x === Infinity;
    default: return false;
  }
}

/** A Cholesky factorization fails: a diagonal is nonpositive or not finite. */
function violatesPositiveDefinite(values: number[]): boolean {
  const n = Math.round(Math.sqrt(values.length));
  if (n * n !== values.length) return false;
  const l = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    let d = values[j * n + j];
    for (let k = 0; k < j; k++) d -= l[j * n + k] * l[j * n + k];
    if (!(d > 0 && d < Infinity)) return true;
    d = Math.sqrt(d);
    l[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let s = values[i * n + j];
      for (let k = 0; k < j; k++) s -= l[i * n + k] * l[j * n + k];
      l[i * n + j] = s / d;
    }
  }
  return false;
}

function violates(set: DomainSet, value: any): boolean {
  if (typeof value === 'number') return violatesScalar(set, value);
  if (typeof value === 'boolean') return violatesScalar(set, +value);
  if (value == null || typeof value !== 'object') return false;
  const cacheable = !ArrayBuffer.isView(value);
  if (cacheable && checkedValues.has(value)) return false;
  const values = flatten(value, []);
  let bad: boolean;
  if (set === 'stdsimplex') {
    const n = innerLength(value);
    bad = false;
    for (let start = 0; start < values.length && !bad; start += n) {
      let total = 0;
      for (let i = start; i < start + n; i++) {
        if (values[i] < 0) bad = true;
        total += values[i];
      }
      if (total === Infinity || Math.abs(total - 1) > simplexTolerance(n) * Math.max(total, 1)) bad = true;
    }
  } else if (set === 'posdefinite') {
    bad = violatesPositiveDefinite(values);
  } else {
    bad = values.some(x => violatesScalar(set, x));
  }
  if (!bad && cacheable) checkedValues.add(value);
  return bad;
}

function describe(value: any): string {
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  const values = flatten(value, []);
  const shown = values.slice(0, 8).join(', ');
  return '[' + shown + (values.length > 8 ? ', …' : '') + ']';
}

const SET_NAMES: Record<DomainSet, string> = {
  posreals: 'posreals', nonnegreals: 'nonnegreals', unitinterval: 'unitinterval',
  posintegers: 'posintegers', stdsimplex: 'stdsimplex(n)',
  posdefinite: 'positive definite matrices',
};

/** Throw when parameter `name` of `kernel` lies outside its declared set. */
function checkParam(kernel: string, name: string, value: any): void {
  const set = PARAM_SETS[kernel]?.[name];
  if (!set || !violates(set, value)) return;
  const err: any = new Error(`${kernel}: parameter '${name}' = ${describe(value)} is outside its `
    + `declared value set ${SET_NAMES[set]} (spec ${SECTION_09.has(kernel) ? '§09' : '§08'})`);
  err.flatpplParameterDomain = true;
  throw err;
}

/** Whether `err` reports a parameter outside its declared value set. */
function isParameterDomainError(err: any): boolean {
  return err?.flatpplParameterDomain === true;
}

/** `checkParam` over a positional list in declared order, or a record. */
function checkParams(kernel: string, names: readonly string[], params: any): void {
  const sets = PARAM_SETS[kernel];
  if (!sets || params == null) return;
  if (Array.isArray(params)) {
    for (let i = 0; i < names.length && i < params.length; i++) {
      if (sets[names[i]]) checkParam(kernel, names[i], params[i]);
    }
    return;
  }
  for (const name of Object.keys(sets)) {
    if (Object.prototype.hasOwnProperty.call(params, name)) checkParam(kernel, name, params[name]);
  }
}

function hasDeclaredSets(kernel: string): boolean {
  return Object.prototype.hasOwnProperty.call(PARAM_SETS, kernel);
}

module.exports = {
  checkParam, checkParams, hasDeclaredSets, isParameterDomainError, simplexTolerance, PARAM_SETS,
};
