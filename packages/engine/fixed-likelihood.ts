'use strict';

// §04 fixed observations / §06 IID / §08 normalized scalar densities.
// Only a prepared scorer's private snapshot owns these summaries. Arbitrary
// caller arrays and values generated while walking a measure never enter the
// cache. Parameters and environments are resolved anew by density.ts.

const valueLib = require('./value.ts');
const gammaln = require('@stdlib/math-base-special-gammaln');
const betaln = require('@stdlib/math-base-special-betaln');
const binomcoefln = require('@stdlib/math-base-special-binomcoefln');

type Summary = { sum?: number; logs?: number; failures?: number; constant?: number;
  center?: number; offset?: number; squares?: number };

/** Copy observation structure without exposing storage shared with the host.
 * Preserve Value metadata, tables and nested arrays for the general walker. */
function copyObservations(value: any, own?: (value: any) => void, seen = new Map<any, any>()): any {
  if (value == null || typeof value !== 'object') return value;
  if (seen.has(value)) return seen.get(value);
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    const copied = new (value.constructor as any)(value);
    seen.set(value, copied);
    own?.(copied);
    return copied;
  }
  const copied: any = Array.isArray(value) ? [] : {};
  seen.set(value, copied);
  // Shape is metadata: copy it without treating it as observable storage.
  for (const key of Object.keys(value)) copied[key] = valueLib.isValue(value) && key === 'shape'
    ? value.shape.slice() : copyObservations(value[key], own, seen);
  if (Array.isArray(copied)) own?.(copied);
  return copied;
}

/** Compensated sums keep setup error independent of observation ordering.
 * A non-finite term invalidates the summary and keeps the ordinary scorer. */
function sumTerms(length: number, term: (i: number) => number): number {
  let sum = 0, correction = 0;
  for (let i = 0; i < length; i++) {
    const x = term(i);
    if (!Number.isFinite(x)) return NaN;
    const next = sum + x;
    correction += Math.abs(sum) >= Math.abs(x) ? (sum - next) + x : (x - next) + sum;
    sum = next;
  }
  return sum + correction;
}

/** Summarize a proved-owned region once, including support and normalization.
 * Gamma/Beta endpoints and Loader-scale Poisson counts deliberately fall back.
 * Normal's centered second pass avoids subtracting raw sums of squares. */
function summarize(kernel: string, data: any, length: number, n: number | null): Summary | null {
  const at = (i: number) => typeof data[i] === 'boolean' ? +data[i] : data[i];
  for (let i = 0; i < length; i++) {
    const x = at(i);
    if (typeof x !== 'number' || !Number.isFinite(x)) return null;
    if ((kernel === 'Gamma' && x <= 0) || (kernel === 'Beta' && !(x > 0 && x < 1))
        || (kernel === 'Exponential' && x < 0)
        || (kernel === 'Poisson' && (!Number.isInteger(x) || x < 0 || x >= 64))
        || (kernel === 'Binomial' && (!Number.isInteger(x) || x < 0 || x > n!))) return null;
  }
  let s: Summary;
  if (kernel === 'Normal') {
    const center = at(0);
    const offset = sumTerms(length, i => at(i) - center) / length;
    s = { center, offset, squares: sumTerms(length, i => {
      const residual = (at(i) - center) - offset;
      const square = residual * residual;
      return square === 0 && residual !== 0 ? NaN : square;
    }) };
  } else if (kernel === 'Beta') {
    s = { logs: sumTerms(length, i => Math.log(at(i))), failures: sumTerms(length, i => Math.log1p(-at(i))) };
  } else {
    s = { sum: sumTerms(length, at) };
    if (kernel === 'Gamma') s.logs = sumTerms(length, i => Math.log(at(i)));
    if (kernel === 'Poisson') s.constant = sumTerms(length, i => gammaln(at(i) + 1));
    if (kernel === 'Binomial') {
      s.failures = sumTerms(length, i => n! - at(i));
      s.constant = sumTerms(length, i => binomcoefln(n, at(i)));
    }
  }
  return Object.values(s).every(Number.isFinite) ? s : null;
}

/** Evaluate the full §08 score; invalid parameters retain registry semantics.
 * Finite arithmetic only: overflow/cancellation beyond the summary's range
 * returns null rather than manufacturing a different support decision. */
function score(kernel: string, s: Summary, length: number, params: number[]): number | null {
  if (!params.every(x => typeof x === 'number' && Number.isFinite(x))) return null;
  const [a, b] = params;
  let result: number;
  switch (kernel) {
    case 'Normal': {
      if (!(b > 0)) return null;
      const delta = ((s.center! - a) + s.offset!) / b;
      result = -length * (Math.log(b) + .5 * Math.log(2 * Math.PI))
        - .5 * (s.squares! / b / b + length * delta * delta);
      break;
    }
    case 'Gamma': {
      if (!(a > 0 && b > 0)) return null;
      const constant = length * (a * Math.log(b) - gammaln(a));
      const logarithmic = (a - 1) * s.logs!, linear = b * s.sum!;
      result = constant + logarithmic - linear;
      // Concentrated Gamma laws have a cancellation-safe registry scorer.
      // Keep it when roundoff in these competing terms can dominate the score.
      if ((Math.abs(constant) + Math.abs(logarithmic) + Math.abs(linear)) * Number.EPSILON
          > 1e-12 * Math.max(1, Math.abs(result))) return null;
      break;
    }
    case 'Beta':
      if (!(a > 0 && b > 0)) return null;
      result = (a - 1) * s.logs! + (b - 1) * s.failures! - length * betaln(a, b);
      break;
    case 'Poisson':
      if (a < 0) return null;
      if (a === 0) return s.sum === 0 ? 0 : -Infinity;
      result = s.sum! * Math.log(a) - length * a - s.constant!;
      break;
    case 'Binomial':
      if (b < 0 || b > 1) return null;
      if (b === 0) return s.sum === 0 ? 0 : -Infinity;
      if (b === 1) return s.failures === 0 ? 0 : -Infinity;
      result = s.constant! + s.sum! * Math.log(b) + s.failures! * Math.log1p(-b);
      break;
    case 'Exponential':
      if (!(a > 0)) return null;
      result = length * Math.log(a) - a * s.sum!;
      break;
    default: return null;
  }
  return Number.isFinite(result) ? result : null;
}

/** Build a private observation snapshot and its region cache. Typed subarrays
 * retain ownership through their copied buffer; offset and consumed length
 * distinguish suffixes and prefixes. Plain array copies made by the walker
 * are unowned and fall back. A Binomial node owns at most one fixed n. */
function prepareObservations(observed: any, fixedTrialCount: (ir: any) => number | null,
  resolveFixedParameter: (ir: any) => any) {
  const storage = new WeakMap<object, Map<string, Summary | null>>();
  const own = (v: any) => storage.set(v instanceof Float64Array ? v.buffer : v, new Map());
  const snapshot = copyObservations(observed, own);
  const trials = new WeakMap<object, number | null>();
  const parameters = new WeakMap<object, { value: any; scalar?: number; refs: string[] } | null>();
  // §04 broadcast: fixed homogeneous vectors are shared parameters too. Prove
  // this once from copied fixed values; runtime vectors never get scanned.
  function fixedParameter(ir: any): { value: any; scalar?: number; refs: string[] } | null {
    if (!ir || typeof ir !== 'object') return null;
    if (!parameters.has(ir)) {
      let prepared = null;
      try {
        const value = copyObservations(resolveFixedParameter(ir));
        const vector = valueLib.isValue(value)
          ? value.shape.length === 1 && !value.im && !valueLib.isTransposeView(value) ? value.data : null
          : Array.isArray(value) || value instanceof Float64Array ? value : null;
        const first = vector?.[0];
        const scalar = typeof value === 'number' ? value : typeof first === 'number'
          && vector.every((x: any) => x === first) ? first : undefined;
        const refs = new Set<string>();
        const collect = (node: any) => {
          if (!node || typeof node !== 'object') return;
          if (node.kind === 'ref') refs.add(node.name);
          for (const key of Object.keys(node)) collect(node[key]);
        };
        collect(ir);
        prepared = { value, scalar, refs: Array.from(refs) };
      } catch (_) { /* Runtime parameters retain ordinary per-call evaluation. */ }
      parameters.set(ir, prepared);
    }
    return parameters.get(ir)!;
  }
  function reduce(ir: any, value: any, length: number, params: number[], kernel = ir.op): number | null {
    if (!['Normal', 'Gamma', 'Beta', 'Poisson', 'Binomial', 'Exponential'].includes(kernel)) return null;
    if (valueLib.isValue(value)) {
      if (value.shape.length !== 1 || value.im || valueLib.isTransposeView(value)) return null;
      value = value.data;
    }
    const typed = value instanceof Float64Array;
    const cache = value && storage.get(typed ? value.buffer : value);
    if (!cache || value.length < length) return null;
    let n = null;
    if (kernel === 'Binomial') {
      if (!trials.has(ir)) {
        const fixed = ir.op === 'broadcast'
          ? fixedParameter(ir.kwargs?.n ?? ir.args?.[1])?.scalar : fixedTrialCount(ir);
        trials.set(ir, typeof fixed === 'number' && Number.isSafeInteger(fixed) && fixed >= 0 ? fixed : null);
      }
      n = trials.get(ir)!;
      if (n === null || params[0] !== n) return null;
    }
    const key = `${kernel}:${typed ? value.byteOffset : 0}:${length}:${n}`;
    if (!cache.has(key)) cache.set(key, summarize(kernel, value, length, n));
    const summary = cache.get(key);
    return summary ? score(kernel, summary, length, params) : null;
  }
  return { snapshot, reduce, fixedParameter };
}

module.exports = { prepareObservations, copyObservations };
