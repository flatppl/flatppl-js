'use strict';

// matKernelBroadcast — array-valued independent-product measure
// `broadcast(Dist, c1, c2, …)` (spec §04 sec:higher-order).
//
// Element j of every atom is drawn from Dist(params_j), where
// params_j is the j-th element of each collection arg (rank-1) or the
// held-constant scalar (rank-0 / length-1 singleton). Result is a
// vector-atom measure shape=[N, K], atom-major, mirroring matIid.
//
// Supports atom-independent params (fixed-phase μ, σ — closed-form
// Normal MvNormal pipeline takes the hot path) AND per-atom params
// (stochastic-ancestor expressions like `means = α .+ β .* x_data` —
// evaluated batched via `evaluateExprN`, then K worker sampleN calls
// with per-atom refArrays carrying each param's atom-i value).
// Symmetric with the density side (`density.walkBroadcast`) which
// dispatches off the same shape contract.

import type {
  DerivationKernelBroadcast,
} from './engine-types';

const empirical    = require('./empirical.ts');
const orchestrator = require('./orchestrator.ts');
const valueLib     = require('./value.ts');
const shared       = require('./materialiser-shared.ts');

const { nameSeed, measureFromValue, prepareDensityRefs, pushFixedEnv } = shared;

function matKernelBroadcast(name: string, d: DerivationKernelBroadcast, ctx: any) {
  const sampler = require('./sampler.ts');
  const params: string[] = (sampler._internal.REGISTRY[d.distOp] || {}).params;
  if (!params) {
    return Promise.reject(new Error(
      'broadcast: unknown distribution kernel ' + d.distOp));
  }
  // Map parameter slot → its IR (kwarg form wins, else positional).
  const paramIRs: Record<string, any> = {};
  if (d.kwargIRs && Object.keys(d.kwargIRs).length > 0) {
    for (const pn of params) {
      if (Object.prototype.hasOwnProperty.call(d.kwargIRs, pn)) paramIRs[pn] = d.kwargIRs[pn];
    }
    for (const pn of Object.keys(d.kwargIRs)) {
      if (paramIRs[pn] === undefined) paramIRs[pn] = d.kwargIRs[pn];
    }
  } else if (Array.isArray(d.argIRs)) {
    if (d.argIRs.length !== params.length) {
      return Promise.reject(new Error('broadcast(' + d.distOp + '): expected '
        + params.length + ' parameter args (' + params.join(', ')
        + '), got ' + d.argIRs.length));
    }
    for (let i = 0; i < params.length; i++) paramIRs[params[i]] = d.argIRs[i];
  }
  const pnames = Object.keys(paramIRs);
  // Synthesize a containing IR so prepareDensityRefs collects every
  // free name across all parameter expressions in one pass.
  const aggregateIR: any = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: d.distOp }],
    kwargs: paramIRs,
  };
  return prepareDensityRefs(aggregateIR, ctx, 'broadcast').then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const baseEnv: Record<string, any> = {};
    for (const k in fixedEnv) baseEnv[k] = fixedEnv[k];
    const refNames = Object.keys(refArrays);
    const N = ctx.sampleCount;
    // Evaluate each parameter expression once over the atom batch.
    // The shape disambiguates: per-atom vs atom-indep; scalar vs
    // length-K vector. Mirrors density.walkBroadcast's classification.
    const usesAtomBy: Record<string, boolean> = {};
    const paramVals: Record<string, any> = {};
    let K = 1;
    for (const pn of pnames) {
      const paramRefs = orchestrator.collectSelfRefs(paramIRs[pn]);
      const usesAtom = refNames.some((n) => paramRefs.has(n));
      usesAtomBy[pn] = usesAtom;
      let v: any;
      try {
        v = sampler.evaluateExprN(paramIRs[pn], refArrays, N, baseEnv, undefined);
      } catch (err) {
        return Promise.reject(err instanceof Error ? err : new Error(String(err)));
      }
      paramVals[pn] = v;
      // Determine intrinsic K from this param (the non-batch axis).
      let intrinsicK = 1;
      if (typeof v === 'number' || typeof v === 'boolean') {
        intrinsicK = 1;
      } else if (valueLib.isValue(v)) {
        const shape = v.shape;
        if (shape.length === 0) intrinsicK = 1;
        else if (shape.length === 1) {
          intrinsicK = (usesAtom && shape[0] === N) ? 1 : shape[0];
        } else if (shape.length === 2 && shape[0] === N) {
          intrinsicK = shape[1];
        } else {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): parameter \'' + pn + '\' resolved to unsupported shape '
            + JSON.stringify(shape) + ' (v1 supports scalar / [K] / [N] / [N, K])'));
        }
      } else if (v && v.BYTES_PER_ELEMENT !== undefined
                  && typeof v.length === 'number') {
        intrinsicK = (usesAtom && v.length === N) ? 1 : v.length;
      } else if (Array.isArray(v)) {
        intrinsicK = v.length;
      } else {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): parameter \'' + pn + '\' resolved to non-numeric value'));
      }
      if (intrinsicK > 1) {
        if (K === 1) K = intrinsicK;
        else if (K !== intrinsicK) {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): incompatible collection lengths (' + K + ' vs '
            + intrinsicK + ') on parameter \'' + pn + '\''));
        }
      }
    }
    if (K < 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp
        + '): empty collection argument'));
    }
    const anyAtomDep = pnames.some((pn) => usesAtomBy[pn]);

    // ATOM-INDEP HOT PATH (closed-form Normal): broadcast(Normal,
    // μ_vec, σ_vec) with both params atom-independent ⇒ MvNormal(μ,
    // diag(σ²)). Uses the structured-matrix diag Cholesky for O(N·K)
    // generation.
    if (!anyAtomDep && d.distOp === 'Normal' && paramVals.mu && paramVals.sigma) {
      const valueOps = require('./value-ops.ts');
      const muV  = valueLib.asValue(paramVals.mu);
      const sigV = valueLib.asValue(paramVals.sigma);
      const muLen  = muV.shape.length === 0 ? 1 : muV.shape[0];
      const sigLen = sigV.shape.length === 0 ? 1 : sigV.shape[0];
      const muVec  = new Float64Array(K);
      const sigSq  = new Float64Array(K);
      for (let j = 0; j < K; j++) {
        muVec[j] = muV.data[muLen === 1 ? 0 : j];
        const s = sigV.data[sigLen === 1 ? 0 : j];
        sigSq[j] = s * s;
      }
      const cov = valueLib.diagMatrix(sigSq);
      let L;
      try {
        L = sampler._internal.ARITH_OPS.lower_cholesky(cov);
      } catch (err) {
        return Promise.reject(new Error('broadcast(Normal): ' + (err as any).message));
      }
      const stdNormalIR = {
        kind: 'call', op: 'Normal',
        kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } },
      };
      return ctx.sendWorker({
        type: 'sampleN', ir: stdNormalIR, count: N, repeat: K,
        refArrays: {}, seed: nameSeed(name, ctx.rootSeed),
      }).then((reply: any) => {
        const z = { shape: [N, K], data: reply.samples };
        const Lz = valueOps.mulN(L, z, N);
        const result = valueOps.addN({ shape: [K], data: muVec }, Lz, N);
        return measureFromValue(result, {
          logWeights: null, logTotalmass: 0, n_eff: N,
        });
      });
    }

    // GENERAL PATH: K worker sampleN calls, each with per-atom
    // refArrays carrying the (i, j) value of each per-atom parameter.
    // Atom-indep params pass as lit kwargs. Same shape contract as
    // walkBroadcast on the density side; reflects the engine's
    // batched scalar/array storage (§2.1 leading-axis batch).
    const elemScalar = (v: any, j: number): number => {
      if (typeof v === 'number') return v;
      if (typeof v === 'boolean') return v ? 1 : 0;
      if (valueLib.isValue(v)) {
        const shape = v.shape;
        if (shape.length === 0) return v.data[0];
        if (shape.length === 1) {
          return v.data[shape[0] === 1 ? 0 : j];
        }
      }
      if (v && v.BYTES_PER_ELEMENT !== undefined
          && typeof v.length === 'number') {
        return v[v.length === 1 ? 0 : j];
      }
      if (Array.isArray(v)) return +v[v.length === 1 ? 0 : j];
      throw new Error('broadcast: cannot scalarise atom-indep value at j=' + j);
    };
    const perAtomColumn = (v: any, j: number): Float64Array => {
      const col = new Float64Array(N);
      if (valueLib.isValue(v)) {
        const shape = v.shape;
        const data = v.data;
        if (shape.length === 1 && shape[0] === N) {
          col.set(data);
        } else if (shape.length === 2 && shape[0] === N) {
          const stride = shape[1];
          const off = stride === 1 ? 0 : j;
          for (let i = 0; i < N; i++) col[i] = data[i * stride + off];
        } else {
          throw new Error('broadcast: unexpected per-atom shape ' + JSON.stringify(shape));
        }
      } else if (v && v.BYTES_PER_ELEMENT !== undefined && v.length === N) {
        col.set(v);
      } else {
        throw new Error('broadcast: per-atom param has unexpected type ' + (typeof v));
      }
      return col;
    };
    const cols = new Array(K);
    let chain = Promise.resolve();
    for (let j = 0; j < K; j++) {
      const jj = j;
      chain = chain.then(() => {
        const kwargs: Record<string, any> = {};
        const perAtomRefs: Record<string, any> = {};
        let safeIdx = 0;
        for (const pn of pnames) {
          if (usesAtomBy[pn]) {
            const refName = '__bc_' + pn + '_' + (safeIdx++);
            try {
              const col = perAtomColumn(paramVals[pn], jj);
              perAtomRefs[refName] = valueLib.batchedScalar(col);
            } catch (err) {
              throw err instanceof Error ? err : new Error(String(err));
            }
            kwargs[pn] = { kind: 'ref', ns: 'self', name: refName };
          } else {
            kwargs[pn] = { kind: 'lit', value: elemScalar(paramVals[pn], jj) };
          }
        }
        const distIR = { kind: 'call', op: d.distOp, kwargs: kwargs };
        return ctx.sendWorker({
          type: 'sampleN', ir: distIR, count: N,
          refArrays: perAtomRefs,
          seed: nameSeed(name + ':' + jj, ctx.rootSeed),
        }).then((reply: any) => { cols[jj] = reply.samples; });
      });
    }
    return pushFixedEnv(ctx, fixedEnv).then(() => chain).then(() => {
      const out = new Float64Array(N * K);
      for (let j = 0; j < K; j++) {
        const col = cols[j];
        for (let i = 0; i < N; i++) out[i * K + j] = col[i];
      }
      const value = { shape: [N | 0, K], data: out };
      return Object.assign(
        empirical.arrayMeasure(out, [K], null),
        { value: value, logTotalmass: 0, n_eff: N },
      );
    });
  });
}

module.exports = { matKernelBroadcast };
