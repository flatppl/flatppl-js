'use strict';

// §08: disjoint bin counts have rates equal to their intensity masses.
// Share exact scalar interval and independent-record rectangle masses between
// sampling and density. Stochastic record ancestry needs joint integration.
const sampler = require('./sampler.ts');
const valueLib = require('./value.ts');
const { regionBoundsFromIR } = require('./sampler-registry.ts');
const { cdf, hasCdf, intervalProbability } = require('./forward-cdf.ts');
const { logSumExp } = require('./empirical.ts');
// Bound amplification of CDF input rounding by subtraction. This does not
// certify the Beta routine's own accuracy (1e4 * epsilon is about2.2e-12).
const MAX_NB_CDF_SUBTRACTION_CONDITION = 1e4;

function resolveBinnedRates(ir: any, env: any): { shape: number[], data: Float64Array } {
  const kw = ir.kwargs || {}, args = ir.args || [];
  const binsIR = kw.bins || (args.length >= 2 ? args[0] : null);
  let rates: Float64Array;
  let shape: number[];
  if (!binsIR) {
    // Retain the existing direct-rate sampler and density interfaces.
    const rateIR = kw.rates || kw.intensity || args[0];
    if (!rateIR) throw new Error('BinnedPoissonProcess: requires rates or bins and intensity');
    rates = vector(sampler.evaluateExpr(rateIR, env), 'rates');
    shape = [rates.length];
  } else {
    const intensity = kw.intensity || args[1];
    const bins = sampler.evaluateExpr(binsIR, env);
    const names = !valueLib.isValue(bins) && !Array.isArray(bins) && !ArrayBuffer.isView(bins)
      ? Object.keys(bins) : null;
    if (names && names.length === 0) throw new Error('BinnedPoissonProcess: bins must have coordinates');
    const edges = names ? names.map(name => vector(bins[name], 'bins.' + name)) : [vector(bins, 'bins')];
    for (const axis of edges) {
      if (axis.length < 2) throw new Error('BinnedPoissonProcess: requires at least two bin edges');
      for (let k = 1; k < axis.length; k++) {
        if (!(axis[k] > axis[k - 1])) throw new Error('BinnedPoissonProcess: bin edges must increase');
      }
    }
    shape = edges.map(axis => axis.length - 1);
    rates = new Float64Array(shape.reduce((a, b) => a * b, 1));
    for (let k = 0; k < rates.length; k++) {
      let index = k;
      const bounds = new Array(edges.length);
      for (let j = edges.length - 1; j >= 0; j--) {
        const cell = index % shape[j];
        index = Math.floor(index / shape[j]);
        bounds[j] = { name: names && names[j], lo: edges[j][cell], hi: edges[j][cell + 1],
          rightClosed: cell === shape[j] - 1 };
      }
      rates[k] = Math.exp(logBinMass(intensity, bounds, env));
    }
  }
  for (let k = 0; k < rates.length; k++) {
    const rate = rates[k];
    if (!(rate >= 0) || !Number.isFinite(rate)) {
      throw new Error('BinnedPoissonProcess: rates[' + k + '] = ' + rate + ' must be non-negative and finite');
    }
  }
  return { shape, data: rates };
}

function logBinMass(ir: any, bounds: any[], env: any): number {
  if (!ir || ir.kind !== 'call') throw new Error('BinnedPoissonProcess: unresolved intensity measure');
  if (bounds.some(b => b.lo > b.hi || (b.lo === b.hi && (!b.rightClosed || b.leftClosed === false)))) return -Infinity;
  if (ir.op === 'weighted' || ir.op === 'logweighted') {
    if (ir.args[0]?.op === 'functionof') {
      throw new Error('BinnedPoissonProcess: event-dependent intensity weights require integration not supported by the exact bin-mass path');
    }
    const wv = +sampler.evaluateExpr(ir.args[0], env);
    const logWeight = ir.op === 'logweighted' ? wv : Math.log(wv);
    if (Number.isNaN(logWeight) || logWeight === Infinity) {
      throw new Error('BinnedPoissonProcess: intensity weight must be non-negative and finite in log space');
    }
    return logWeight === -Infinity ? -Infinity : logWeight + logBinMass(ir.args[1], bounds, env);
  }
  if (ir.op === 'superpose' || (ir.op === 'select' && ir.logweights == null && ir.selectorName == null)) {
    const parts = ir.op === 'superpose' ? ir.args : ir.branches;
    return logSumExp(parts.map((part: any) => logBinMass(part, bounds, env)));
  }
  if (ir.op === 'normalize') {
    const total = logBinMass(ir.args[0], bounds.map(b => ({ ...b, lo: -Infinity, hi: Infinity,
      leftClosed: true, rightClosed: true })), env);
    if (!Number.isFinite(total)) throw new Error('BinnedPoissonProcess: cannot normalize a zero or non-finite intensity');
    return logBinMass(ir.args[0], bounds, env) - total;
  }
  if (bounds[0].name != null) {
    if (ir.op !== 'joint' || !ir.fields || ir.fields.length !== bounds.length
        || ir.fields.some((f: any) => f.source != null)) {
      throw new Error('BinnedPoissonProcess: record bin integration requires an independent joint intensity');
    }
    return bounds.reduce((mass, bound) => {
      const field = ir.fields.find((f: any) => f.name === bound.name);
      if (!field) throw new Error('BinnedPoissonProcess: intensity fields must match bin coordinates');
      return mass + logBinMass(field.value, [{ ...bound, name: null }], env);
    }, 0);
  }
  const { lo, hi, rightClosed, leftClosed = true } = bounds[0];
  if (ir.op === 'truncate') {
    const [lower, upper] = regionBoundsFromIR(ir.args[1], env);
    const closed = !(ir.args[1].kind === 'const' && ir.args[1].name === 'posreals');
    return logBinMass(ir.args[0], [{ lo: Math.max(lo, lower), hi: Math.min(hi, upper),
      leftClosed: lower > lo ? closed : lower < lo ? leftClosed : closed && leftClosed,
      rightClosed: upper < hi || rightClosed }], env);
  }
  if (ir.op === 'Dirac') {
    const point = +sampler.resolveParams(ir, sampler.lookupDistribution(ir), env)[0];
    return (point > lo || (leftClosed && point === lo))
      && (point < hi || (rightClosed && point === hi)) ? 0 : -Infinity;
  }
  if (!sampler.isKnownDistribution(ir.op)) {
    throw new Error('BinnedPoissonProcess: unsupported bin integration for ' + ir.op);
  }
  const entry = sampler.lookupDistribution(ir);
  const values = sampler.resolveParams(ir, entry, env);
  if (entry.discrete) {
    // Integer atoms in [lo,hi), except the final bin includes hi.
    const lower = leftClosed ? Math.ceil(lo) - 1 : Math.floor(lo);
    const upper = rightClosed ? Math.floor(hi) : Math.ceil(hi) - 1;
    if (upper <= lower) return -Infinity;
    if (ir.op === 'Categorical' || ir.op === 'Categorical0') {
      const probabilities = valueLib.isValue(values[0]) ? values[0].data : values[0];
      const offset = ir.op === 'Categorical' ? 1 : 0;
      const first = Math.max(0, lower + 1 - offset);
      const last = Math.min(probabilities.length - 1, upper - offset);
      let mass = 0;
      for (let k = first; k <= last; k++) mass += probabilities[k];
      return Math.log(mass);
    }
    if (ir.op === 'Geometric') {
      const p = values[0], first = Math.max(0, lower + 1);
      if (!(p > 0 && p <= 1)) throw new Error('BinnedPoissonProcess: Geometric p must be in (0,1]');
      if (upper < first) return -Infinity;
      if (p === 1) return first === 0 ? 0 : -Infinity;
      const logQ = Math.log1p(-p);
      return first * logQ + Math.log(-Math.expm1((upper - first + 1) * logQ));
    }
    if (ir.op === 'NegativeBinomial' || ir.op === 'NegativeBinomial2') {
      if (upper < 0) return -Infinity;
      if (lower < 0 && upper === Infinity) return 0;
      if (upper === lower + 1) return entry.logpdfFn(upper, ...values);
      const shape = ir.op === 'NegativeBinomial' ? values[0] : values[1];
      const rate = ir.op === 'NegativeBinomial' ? values[1] : values[1] / values[0];
      // Compute both odds separately: subtracting either from one loses the
      // small tail. NB2 uses the smaller ratio, so finite inputs cannot overflow.
      const inverseRate = ir.op === 'NegativeBinomial' ? 1 / values[1] : values[0] / values[1];
      const logQ = ir.op === 'NegativeBinomial' ? -Math.log1p(values[1])
        : values[0] < values[1] ? Math.log(values[0]) - Math.log(values[1]) - Math.log1p(inverseRate)
        : -Math.log1p(rate);
      // Short bins use every PMF term directly: even a positive difference
      // of two CDF values can lose most of a narrow bin's relative precision.
      const first = Math.max(0, lower + 1), count = upper - first + 1;
      // Shape1 is geometric, so its complete interval sum needs no CDF or
      // finite enumeration, including an infinite upper endpoint.
      if (shape === 1) return first * logQ + Math.log(-Math.expm1(count * logQ));
      if (count > 0 && count <= 256 && Number.isSafeInteger(first) && Number.isSafeInteger(upper)) {
        let term = entry.logpdfFn(first, ...values);
        if (!Number.isFinite(term)) throw new Error('BinnedPoissonProcess: NB bin has no finite log-PMF seed');
        const terms = [term];
        for (let k = first; k < upper; k++) {
          term += Math.log(k + shape) - Math.log(k + 1) + logQ;
          terms.push(term);
        }
        return logSumExp(terms);
      }
      const p = rate <= 1 ? rate / (1 + rate) : 1 / (1 + inverseRate);
      const q = inverseRate <= 1 ? inverseRate / (1 + inverseRate) : 1 / (1 + rate);
      const F = (k: number) => k < 0 ? 0 : k === Infinity ? 1
        : cdf('Beta', p, { alpha: shape, beta: k + 1 });
      const S = (k: number) => k < 0 ? 1 : k === Infinity ? 0
        : cdf('Beta', q, { alpha: k + 1, beta: shape });
      const lowerTail = p > 0 && p < 1;
      const upperTail = q > 0 && q < 1;
      const Flo = lowerTail ? F(lower) : NaN;
      const useSurvival = upperTail && (!lowerTail || Flo > 0.5);
      const a = useSurvival ? S(lower) : lowerTail ? F(upper) : NaN;
      const b = useSurvival ? S(upper) : Flo;
      const mass = a - b;
      if (mass > 0 && mass <= 1
          && (Math.abs(a) + Math.abs(b)) / mass <= MAX_NB_CDF_SUBTRACTION_CONDITION) {
        return Math.log(mass);
      }
      // Never truncate an interval to the finite recurrence work bound.
      throw new Error('BinnedPoissonProcess: NB bin needs a trustworthy analytic mass or at most 256 finite atoms');
    }
    const distribution = new entry.Ctor(...values);
    if (typeof distribution.cdf !== 'function') {
      throw new Error('BinnedPoissonProcess: discrete bin integration requires a registered CDF; unsupported ' + ir.op);
    }
    const Flo = distribution.cdf(lower);
    // Survival identities avoid losing small upper-tail bin masses to 1-1.
    const survival = (k: number): number => {
      if (k < 0) return 1;
      if (k === Infinity) return 0;
      if (ir.op === 'Poisson') return cdf('Gamma', values[0], { shape: k + 1, rate: 1 });
      if (ir.op === 'Bernoulli') return k < 1 ? values[0] : 0;
      if (ir.op === 'Binomial') return k >= values[0] ? 0
        : cdf('Beta', values[1], { alpha: k + 1, beta: values[0] - k });
      return 1 - distribution.cdf(k);
    };
    return Math.log(Flo > 0.5 ? survival(lower) - survival(upper)
      : distribution.cdf(upper) - Flo);
  }
  if (!hasCdf(ir.op)) {
    throw new Error('BinnedPoissonProcess: bin integration requires a scalar CDF or Dirac; unsupported ' + ir.op);
  }
  const names = ir.op === 'Uniform' ? ['lo', 'hi'] : entry.params;
  const params: Record<string, number> = {};
  for (let j = 0; j < names.length; j++) params[names[j]] = values[j];
  return Math.log(intervalProbability(ir.op, params, lo, hi));
}

function vector(v: any, name: string): Float64Array {
  const value = valueLib.asValue(v);
  if (value.shape.length !== 1) {
    throw new Error('BinnedPoissonProcess: ' + name + ' must be a vector');
  }
  if (value.shape[0] === 0) throw new Error('BinnedPoissonProcess: ' + name + ' must be non-empty');
  return value.data;
}

module.exports = { resolveBinnedRates };
