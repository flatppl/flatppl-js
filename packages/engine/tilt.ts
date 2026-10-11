'use strict';

// Spec §06 tilt: retain the density expression and the explicit reference
// measure until scoring/sampling. In particular, an inferred Normal scale must
// never be replaced by a normalizer evaluated at a materialised parameter draw.

const call = (op: string, args: any[]) => ({ kind: 'call', op, args });

/** Recognise closed §06 normalizers before the generic measure expansion. */
function exactLaw(ir: any, resolveValue?: (ir: any) => any): any {
  if (ir?.op !== 'tilt' || ir.args?.length !== 2) return null;
  const [m, b] = ir.args;
  if (b?.op !== 'Lebesgue') return null;
  const fixed = (x: any) => x?.kind === 'lit' ? x.value : resolveValue?.(x);
  const support = b.kwargs?.support || b.args?.[0];
  const set = b.vectorSet?.set || support?.op;
  if (set !== 'ordered' && set !== 'zerosum') return null;
  const n = b.vectorSet?.n ?? fixed(support?.args?.[0]);
  if (!Number.isInteger(n) || n < 1) {
    throw new Error('tilt on ordered/zerosum requires a positive static size');
  }
  if (m?.op === 'iid' && m.args?.length === 2) {
    if (fixed(m.args[1]) !== n) {
      throw new Error('tilt source must have the same static vector size as its base');
    }
    const inner = m.args[0];
    const leaf = require('./leaf-params.ts').leafParamIRs(inner);
    if (leaf?.kernel === 'Normal') {
      const [mu, sigma] = leaf.paramIRs;
      return { kind: 'iidNormal', n, set, mu, sigma };
    }
    if (set === 'ordered' && continuousScalar(inner)) {
      return { kind: 'orderedIid', n, set, source: m };
    }
  }
  const head = m?.args?.[0];
  if (set === 'zerosum' && m?.op === 'broadcast' && head?.name === 'Normal') {
    const leaf = require('./leaf-params.ts').leafParamIRs({
      kind: 'call', op: 'Normal', args: m.args.slice(1), kwargs: m.kwargs,
    });
    const [mu, sigma] = leaf.paramIRs;
    return { kind: 'zeroSumNormals', n, set, mu, sigma };
  }
  const law = m?.op === 'iid' ? m.args[0]?.op : m?.op;
  throw new Error(set === 'ordered'
    ? `tilt on ordered requires iid of a continuous scalar law; ${law} is unsupported`
    : `tilt on zerosum has no implemented closed-form sum law for ${law}; independent Normals are supported`);
}

/** Scalar Lebesgue laws retain continuity through weights and restriction. */
function continuousScalar(ir: any, normalized = true): boolean {
  const leaf = require('./leaf-params.ts').leafParamIRs(ir);
  if (leaf) return leaf.kernel !== 'Dirac' && !leaf.entry.discrete && !leaf.entry.densityOnly;
  if (ir?.op === 'normalize') return continuousScalar(ir.args[0], false);
  if (!normalized && ir?.op === 'truncate') return continuousScalar(ir.args[0], false);
  if (!normalized && (ir?.op === 'weighted' || ir?.op === 'logweighted')) return continuousScalar(ir.args[1], false);
  return false;
}

/** Defining measure algebra for cases served by the existing normalizer. */
function definition(ir: any): any {
  const point = { kind: 'ref', ns: '%local', name: '%tilt_variate' };
  const weight = {
    kind: 'call', op: 'functionof', params: [point.name], paramKwargs: [point.name],
    body: call('logdensityof', [ir.args[0], point]),
  };
  return call('normalize', [call('logweighted', [weight, ir.args[1]])]);
}

/** Validate the source law even where its common mean cancels algebraically. */
function checkParameters(mu: number, sigma: number) {
  if (!Number.isFinite(mu) || !Number.isFinite(sigma) || !(sigma > 0)) {
    throw new Error('tilt: exact Normal laws require finite mu and finite sigma > 0');
  }
}

/** Sort or center iid Normal draws; centering avoids cancellation in the mean. */
function transform(x: Float64Array, set: string) {
  if (set === 'ordered') x.sort();
  else {
    const mean = x.reduce((a, b) => a + b, 0) / x.length;
    for (let j = 0; j < x.length; j++) x[j] -= mean;
  }
}

/** Expand one atom's scalar/singleton/vector parameter using §04 broadcasting. */
function parameterVector(value: any, n: number): Float64Array {
  if (typeof value === 'number') return new Float64Array(n).fill(value);
  const data = value?.data || value;
  if ((value?.shape && value.shape.length > 1) || (data?.length !== 1 && data?.length !== n)) {
    throw new Error(`tilt: Normal parameters must broadcast to a vector of length ${n}`);
  }
  return data.length === 1 ? new Float64Array(n).fill(data[0]) : Float64Array.from(data);
}

/** Complete the square on the unit-Jacobian zero-sum chart (§03). */
function conditionalNormal(muValue: any, sigmaValue: any, n: number): any {
  const mu = parameterVector(muValue, n);
  const sigma = parameterVector(sigmaValue, n);
  for (let j = 0; j < n; j++) checkParameters(mu[j], sigma[j]);
  const scale = sigma.reduce((a, b) => Math.max(a, b), 0);
  // Relative variances avoid overflow/underflow when sigma itself is finite.
  const weights = sigma.map((s) => (s / scale) ** 2);
  const total = weights.reduce((a, b) => a + b, 0);
  for (let j = 0; j < n; j++) weights[j] /= total;
  const sumMu = mu.reduce((a, b) => a + b, 0);
  const center = mu.map((m, j) => m - weights[j] * sumMu);
  const logConstant = n === 1 ? 0 : Math.log(scale) + Math.log(total) / 2
    - sigma.reduce((a, s) => a + Math.log(s), 0) - (n - 1) * Math.log(2 * Math.PI) / 2;
  return { center, sigma, weights, logConstant };
}

/** Share the existing broadcast shape rules across scoring and materialisation. */
function conditionalLaws(spec: any, refArrays: any, count: number, env: any, overlay?: any): any {
  const sampler = require('./sampler.ts');
  const shared = require('./materialiser-shared.ts');
  const valueLib = require('./value.ts');
  const { collectSelfRefs } = require('./ir-shared.ts');
  const names = Object.keys(refArrays || {});
  const rows = [spec.mu, spec.sigma].map((param) => {
    const refs = collectSelfRefs(param);
    const v = sampler.evaluateExprN(param, refArrays, count, env, overlay ? { overlay } : undefined);
    const usesAtom = names.some((name) => refs.has(name))
      || (valueLib.isValue(v) && v.shape.length >= 2 && valueLib.isAtomBatched(v, count));
    const shape = shared.classifyBroadcastArg(v, count, usesAtom);
    if (!['scalar', 'K', 'N', 'NK'].includes(shape.kind)
        || (shape.K !== undefined && shape.K !== 1 && shape.K !== spec.n)) {
      throw new Error(`tilt: Normal parameters must broadcast to a vector of length ${spec.n}`);
    }
    if (shape.kind === 'N' || shape.kind === 'NK') {
      const columns = Array.from({ length: spec.n }, (_, j) => shared.perAtomColumnAtJ(v, j, count));
      return (i: number) => Float64Array.from(columns, (column: any) => column[i]);
    }
    const row = Float64Array.from({ length: spec.n }, (_, j) => shared.elemScalarAtJ(v, j, count));
    return (_i: number) => row;
  });
  return (i: number) => conditionalNormal(rows[0](i), rows[1](i), spec.n);
}

/** Condition one standard-Normal draw; callers retain its RNG successor. */
function transformConditional(z: Float64Array, law: any) {
  if (z.length === 1) { z[0] = 0; return; }
  for (let j = 0; j < z.length; j++) z[j] *= law.sigma[j];
  const sum = z.reduce((a, b) => a + b, 0);
  for (let j = 0; j < z.length; j++) z[j] = law.center[j] + z[j] - law.weights[j] * sum;
}

module.exports = { exactLaw, definition, checkParameters, transform, conditionalNormal, conditionalLaws, transformConditional };
