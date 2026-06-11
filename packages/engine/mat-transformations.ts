'use strict';

// Measure-transformation materialiser handlers: matPushfwd (variable
// transformation) and matTruncate (support restriction). Both produce
// a transformed scalar measure from a parent measure.
//
// matPushfwd: per spec §06, samples are { f(x) : x ~ M }. Runs one
// batched evaluateN over f's body with refArrays binding f's param
// name to M's samples. For density, density.walkPushfwd consults the
// `bijection(f, f_inv, logvolume)` annotation on f.
//
// matTruncate: per spec §06, restricts the parent's support to a set
// S. Three paths chosen at materialise time — CDF-inverse,
// rejection-redraw, filter-only fallback. See the inline doc.

import type {
  DerivationPushfwd,
  DerivationTruncate,
} from './engine-types';

const empirical    = require('./empirical.ts');
const orchestrator = require('./orchestrator.ts');
const shared       = require('./materialiser-shared.ts');

const {
  nameSeed,
  collectRefArrays,
  scalarMeasureN,
  measureFromReply,
  measureFromValue,
  setBoundsForMat,
  resolveFnBody,
  measureToParamValue,
  measureToPerAtomRecords,
  inlineBoundaryDerivations,
  measureN,
} = shared;
// Bijection-registry handle (Phase 5.1 Session 5d): when the bijection
// binding carries `registryName` + `paramIRs`, matPushfwd short-
// circuits the AST evaluateN path and calls the registry's atom-
// batched forward directly — same hot path matMvNormal consumes.
const bijRegistry = require('./bijection-registry.ts');
const derivations = require('./derivations.ts');

function matPushfwd(name: string, d: DerivationPushfwd, ctx: any) {
  // pushfwd(f, M): the pushforward of M through function f. Per spec
  // §06, samples are { f(x) : x ~ M }.
  //
  // Two paths:
  //   (1) **Registry fast path (Phase 5.1 Session 5d).** When f is a
  //       bijection binding carrying `registryName` + `paramIRs` AND
  //       the base measure is vector-atom (M.value with outerRank===1,
  //       i.e. an iid(scalar, D) shape), short-circuit through the
  //       bijection-registry's atomBatchedForward. Same hot code path
  //       matMvNormal consumes for the scalar-broadcast case — now
  //       generalised across multivariate dists once their lowering
  //       (Session 5e+) lands.
  //   (2) **Scalar AST path (pre-§22 baseline).** Run one batched
  //       evaluateN over f's body with refArrays binding f's param
  //       name to M's samples. Mass / logWeights / n_eff propagate
  //       through unchanged.
  //
  // For density evaluation, density.walkPushfwd's registry fast path
  // (Session 5d commit 3) mirrors path (1); the legacy AST f_inv path
  // mirrors path (2).
  const fBinding = ctx.bindings && ctx.bindings.get(d.fnRef);
  if (!fBinding) {
    return Promise.reject(new Error(`pushfwd: function binding '${d.fnRef}' not found`));
  }
  const fnInfo = resolveFnBody(fBinding, ctx.bindings);
  if (!fnInfo) {
    return Promise.reject(new Error(`pushfwd: function binding '${d.fnRef}' has no callable body`
      + ` (type=${fBinding.type})`));
  }
  // Resolve the bijection-binding's metadata once (registryName /
  // paramIRs). resolveBijectionMeta also validates fInv / logVolume
  // remain present per the additive invariant; failure returns null,
  // and we fall through to the scalar path which still works for
  // pre-§22 bindings (no registryName).
  const bijMeta = (fBinding.bijection)
    ? derivations.resolveBijectionMeta(fBinding.bijection, ctx.bindings)
    : null;
  return ctx.getMeasure(d.from).then((M: any) => {
    // Registry fast path. Activates only when ALL three conditions
    // hold: (a) bijection-binding marks registryName, (b) paramIRs is
    // attached (additive-invariant pair), (c) M is vector-atom
    // (M.value.shape=[N,D] with outerRank===1 — the iid-of-scalar
    // shape). Otherwise fall through to the scalar AST path; this
    // matches the §22.5 doctrine that the registry path is an
    // OPTIMISATION, not a REPLACEMENT.
    if (bijMeta && bijMeta.registryName
        && M.value && M.value.outerRank === 1
        && Array.isArray(M.value.shape) && M.value.shape.length === 2) {
      if (!bijMeta.paramIRs) {
        return Promise.reject(new Error(
          `pushfwd: bijection binding '${d.fnRef}' has registryName=`
          + `'${bijMeta.registryName}' set without paramIRs `
          + `(additive-invariant violation — producer bug)`));
      }
      const entry = bijRegistry.getBijection(bijMeta.registryName);
      if (!entry) {
        return Promise.reject(new Error(
          `pushfwd: bijection binding '${d.fnRef}' has registryName=`
          + `'${bijMeta.registryName}' which is not found in `
          + `bijection-registry`));
      }
      if (!entry.atomBatchedForward) {
        return Promise.reject(new Error(
          `pushfwd: bijection binding '${d.fnRef}' uses registry entry `
          + `'${bijMeta.registryName}' which has no atomBatchedForward `
          + `(sample-side not yet implemented)`));
      }
      const N = M.value.shape[0];
      const valueLib = require('./value.ts');
      const valueOps = require('./value-ops.ts');
      // Resolve each paramIR to a Value at materialise-time scope.
      //   - Atom-INDEPENDENT / fixed-phase params (literal IRs, refs to
      //     fixed bindings, `lower_cholesky(<fixed cov>)`, …) resolve via
      //     resolveIRToValue — the fast, synchronous path.
      //   - Atom-DEPENDENT params (5f-2): a paramIR that is a direct ref
      //     to a per-atom (stochastic / parameterized) binding can't
      //     resolve that way — resolveIRToValue is fixed-phase only and
      //     THROWS on a `draw`. We instead materialise that binding's
      //     measure and use its per-atom `[N, …]` samples as an
      //     atom-batched param (the registry's affine entry consumes
      //     b=[N,D] / L=[N,D,D] per Session 5f-2). The reparameterisation
      //     X_n = b_n + L·z_n draws (b_n, z_n) independently at atom n —
      //     a valid sample from the hierarchical marginal of X.
      const paramKeys = Object.keys(bijMeta.paramIRs);
      const paramFetches = paramKeys.map((k: string) => {
        const pir = bijMeta.paramIRs[k];
        // Decide the resolution path by INSPECTING the paramIR's
        // self-refs against fixed-phase — NOT by catching resolve
        // errors. A paramIR all of whose self-refs are fixed-phase (or
        // which has none — inline literals) is resolvable synchronously
        // via resolveIRToValue, and any error it raises (non-PD
        // Cholesky, singular matrix, …) is a GENUINE fixed-phase failure
        // that must propagate, not be misread as "atom-dependent". A
        // paramIR with a non-fixed self-ref is atom-dependent.
        const refs = orchestrator.collectSelfRefs(pir);
        let allFixed = true;
        for (const r of refs) {
          if (ctx.fixedValues && ctx.fixedValues.has(r)) continue;
          allFixed = false;
          break;
        }
        if (allFixed) {
          const v = orchestrator.resolveIRToValue(pir, ctx.bindings, ctx.fixedValues);
          if (v == null) {
            return Promise.reject(new Error(
              `pushfwd: bijection binding '${d.fnRef}' cannot resolve `
              + `fixed-phase paramIRs.${k} for registryName='${bijMeta.registryName}'`));
          }
          // Wrap arrays/matrices into Value form. Nested arrays go
          // through valueOps._nestedToValue; rank-1 arrays use asValue.
          const wrapped = (Array.isArray(v) && v.length > 0 && Array.isArray(v[0]))
            ? valueOps._nestedToValue(v)
            : valueLib.asValue(v);
          return Promise.resolve({ k, value: wrapped });
        }
        // Atom-dependent param (5f-2). Only a DIRECT binding ref is
        // supported — a compound expression over a per-atom binding
        // (e.g. `lower_cholesky(<per-atom random cov>)`) would need
        // per-atom expression evaluation, which is deferred.
        if (pir && pir.kind === 'ref' && pir.ns === 'self'
            && pir.name && ctx.bindings && ctx.bindings.has(pir.name)) {
          return ctx.getMeasure(pir.name).then((pm: any) => {
            // Flatten the param binding's measure to an atom-batched
            // [N, …] Value. Handles .value (already a Value), composite
            // .elems (e.g. `muv = [m1, m2]` → [N, D]), and scalar
            // .samples — see measureToParamValue.
            return {
              k,
              value: measureToParamValue(pm, pir.name,
                `pushfwd param '${k}' for '${d.fnRef}'`),
            };
          });
        }
        return Promise.reject(new Error(
          `pushfwd: bijection binding '${d.fnRef}' paramIRs.${k} for `
          + `registryName='${bijMeta.registryName}' is a compound atom-`
          + `dependent expression (not a direct binding ref) — e.g. `
          + `lower_cholesky of a per-atom covariance; deferred`));
      });
      return Promise.all(paramFetches).then((entries: any[]) => {
        const params: any = {};
        for (const e of entries) params[e.k] = e.value;
        // Hand off to the registry. atomBatchedForward returns a Value
        // of the bijection's output shape (per its shapeContract). For
        // 'affine' the contract is [D]→[D], so result is shape [N, D]
        // atom-major mirroring the base.
        const z = M.value;
        let result;
        try {
          result = entry.atomBatchedForward(z, params, N);
        } catch (err) {
          return Promise.reject(new Error(
            `pushfwd: bijection binding '${d.fnRef}' registry call to '`
            + `${bijMeta.registryName}'.atomBatchedForward failed: `
            + (err as any).message));
        }
        // Preserve the outerRank=1 marker from the base. Iid measures
        // carry outerRank=1 (materialiser.ts:459); a pushfwd of an
        // iid base preserves that structure — the result is still an
        // atom-batched stack of D-vectors. Downstream consumers
        // (kernel-broadcast classifier, density walker) read outerRank
        // to distinguish iid output from a per-atom k-vector.
        if (M.value.outerRank === 1 && result && !('outerRank' in result)) {
          result.outerRank = 1;
        }
        // Mass / weights propagate through unchanged — pushfwd preserves
        // total mass regardless of the bijection.
        return measureFromValue(result, {
          logWeights:   M.logWeights || null,
          logTotalmass: (typeof M.logTotalmass === 'number') ? M.logTotalmass : 0,
          n_eff:        (typeof M.n_eff === 'number') ? M.n_eff : N,
        });
      });
    }
    // AST path (pre-§22 baseline).
    //
    // H5 (transitive boundary substitution): f's body may reach its
    // parameter THROUGH derived value bindings (`b = 2 * x;
    // f = functionof(exp(b), x = x)`). Inline those down to the param
    // set FIRST — otherwise collectRefArrays resolves `b` via
    // getMeasure to the like-named MODULE-graph value (computed from
    // the module's decoupled `x` draw), silently transforming the
    // wrong atoms (the audit-§3 boundary-conflation, on pushfwd's
    // sample side).
    const paramSet = new Set<string>(fnInfo.params);
    const body = inlineBoundaryDerivations(fnInfo.body, paramSet, ctx);
    // Bind the param(s) to the base variate per spec §04 application:
    //   - single param, scalar base   → the sample column;
    //   - single param, record base   → per-atom records (the same
    //     whole-record contract feedInputs uses; get_field consumes);
    //   - multi param, record base    → auto-splat by field name;
    //   - multi param, scalar base    → loud error (no splat source).
    const N = measureN(M);
    const paramBind: Record<string, any> = {};
    if (fnInfo.params.length === 1) {
      const p0 = fnInfo.params[0];
      if (M.samples) {
        paramBind[p0] = M.samples;
      } else if (M.fields) {
        paramBind[p0] = measureToPerAtomRecords(M, p0, 'pushfwd');
      } else {
        return Promise.reject(new Error(`pushfwd: base measure '${d.from}' has `
          + `neither scalar samples nor record fields`));
      }
    } else {
      if (!M.fields) {
        return Promise.reject(new Error(`pushfwd: function '${d.fnRef}' takes `
          + fnInfo.params.length + ` inputs but base measure '${d.from}' is `
          + `not record-shaped (auto-splat needs named fields, spec §04)`));
      }
      for (const p of fnInfo.params) {
        const fm = M.fields[p];
        if (!fm || !fm.samples) {
          return Promise.reject(new Error(`pushfwd: function input '${p}' has `
            + `no matching field in the record variate of '${d.from}' `
            + `(fields: ${Object.keys(M.fields).join(', ')})`));
        }
        paramBind[p] = fm.samples;
      }
    }
    // The body may also reference values BEYOND f's parameters — e.g.
    // `pushfwd(fn(_ + mu), base)` where `mu` is a per-atom draw or a
    // fixed binding. Collect those external self-refs (fixed refs are
    // auto-pushed to the worker session env by collectRefArrays). The
    // params ride the `_extraRefArrays` overlay so the collector never
    // resolves them via getMeasure — a body param ref names the BOUND
    // input (spec §04 boundary substitution), not the like-named module
    // binding; resolving it would re-materialise the module draw (the
    // audit-§3 conflation) or throw for a free `elementof` boundary.
    // Under an iid composite fallback ctx the external refs are the
    // tiled per-atom values (the repeat axis), so f's body sees atom
    // i's shared `mu` across the k inner draws.
    const fedCtx = Object.assign({}, ctx, {
      _extraRefArrays: Object.assign({}, ctx._extraRefArrays, paramBind),
      // Phase-4 safety net: a param ref reaching getMeasure uncovered is
      // a feed gap — throw rather than conflate (audit §3).
      _boundaryNames: new Set([
        ...(ctx._boundaryNames || []), ...fnInfo.params,
      ]),
    });
    return collectRefArrays(body, fedCtx).then((bodyRefs: any) => ctx.sendWorker({
      type: 'evaluateN',
      ir: body,
      count: N,
      refArrays: Object.assign({}, bodyRefs, paramBind),
    })).then((reply: any) => {
      return measureFromReply(reply, N, {
        logWeights: M.logWeights,
        logTotalmass: M.logTotalmass,
        n_eff: M.n_eff,
      });
    });
  });
}

/**
 * truncate(M, S) — restrict M's support to S per spec §06.
 *
 * Three paths, chosen at materialise time:
 *
 *   (A) CDF-inverse — when the parent expands to a self-contained
 *       call IR for a known stdlib distribution with static params
 *       (no value-refs in kwargs) and the set descriptor maps to
 *       numeric [lo, hi] bounds. Worker computes Q(F(lo)+u·ΔF) per
 *       atom. Exact; no NaNs; uniform weights. logTotalmass shifts
 *       by log(ΔF).
 *
 *   (B) Rejection-redraw — when the parent expands to a sampleable
 *       IR but isn't (or isn't suitable as) a built-in CDF target.
 *       Worker draws from the expanded IR up to ctx.rejectionBudget
 *       times per atom; budget-exhausted atoms become NaN. n_eff
 *       drops to the count of valid atoms.
 *
 *   (C) Filter-only fallback — when expandMeasure returns null (the
 *       parent's derivation chain hits a kind we can't lift to a
 *       self-contained IR, e.g. a chain through `normalize` or
 *       `superpose`). We keep the parent's atoms that lie in S and
 *       NaN the rest.
 *
 * Per-atom rejection budget reads from ctx.rejectionBudget (defaults
 * to 1000). Configurable per host.
 */
function matTruncate(d: DerivationTruncate, ctx: any) {
  return ctx.getMeasure(d.from).then((parent: any) => {
    const N = ctx.sampleCount;
    const parentLTM = (typeof parent.logTotalmass === 'number') ? parent.logTotalmass : 0;
    const bounds = setBoundsForMat(d.setDescr);
    if (!bounds) {
      return Promise.reject(new Error(
        'truncate: unsupported set kind \'' + (d.setDescr && d.setDescr.kind) + '\''));
    }
    const [lo, hi] = bounds;

    const expanded = orchestrator.expandMeasure(d.from,
      { derivations: ctx.derivations, bindings: ctx.bindings });
    const parentUniform = !parent.logWeights;
    if (expanded && expanded.kind === 'call' && parentUniform) {
      const valueRefs = orchestrator.collectSelfRefs(expanded);
      const hasRefs = valueRefs.size > 0 || (Array.isArray(valueRefs) && valueRefs.length > 0);
      const cdfEligible = !hasRefs && orchestrator.SAMPLEABLE_DISTRIBUTIONS
        && orchestrator.SAMPLEABLE_DISTRIBUTIONS.has(expanded.op);
      const seed = nameSeed(d.from + '|truncate', ctx.rootKey);

      if (cdfEligible) {
        return ctx.sendWorker({
          type: 'truncateSampleN',
          ir: expanded,
          setDescr: d.setDescr,
          count: N,
          mode: 'cdf',
          seed: seed,
        }).then((reply: any) => scalarMeasureN(reply.samples, {
          logWeights: null,
          logTotalmass: parentLTM + reply.logShift,
          n_eff: reply.n_eff,
        }));
      }

      // Rejection-redraw. Need refArrays for parametric params.
      return collectRefArrays(expanded, ctx)
        .then((refArrays: any) => ctx.sendWorker({
          type: 'truncateSampleN',
          ir: expanded,
          setDescr: d.setDescr,
          count: N,
          mode: 'rejection',
          budget: ctx.rejectionBudget != null ? ctx.rejectionBudget : 1000,
          refArrays: refArrays,
          seed: seed,
        }))
        .then((reply: any) => {
          const samples = reply.samples;
          let anyNaN = false;
          for (let i = 0; i < samples.length; i++) {
            if (Number.isNaN(samples[i])) { anyNaN = true; break; }
          }
          let logWeights: Float64Array | null = null;
          if (anyNaN) {
            logWeights = new Float64Array(samples.length);
            for (let i = 0; i < samples.length; i++) {
              logWeights[i] = Number.isNaN(samples[i]) ? -Infinity : 0;
            }
          }
          return scalarMeasureN(samples, {
            logWeights: logWeights,
            logTotalmass: parentLTM + reply.logShift,
            n_eff: reply.n_eff,
          });
        });
    }

    // Filter-only fallback.
    const parentSamples = parent.samples;
    const out = new Float64Array(parentSamples.length);
    const outW = parent.logWeights
      ? new Float64Array(parentSamples.length)
      : null;
    let n_eff = 0;
    if (parent.logWeights) {
      const acceptedWeights: number[] = [];
      const totalWeights: number[] = [];
      const outWnn = outW!;
      for (let i = 0; i < parentSamples.length; i++) {
        const x = parentSamples[i];
        totalWeights.push(parent.logWeights[i]);
        if (x >= lo && x <= hi) {
          out[i] = x;
          outWnn[i] = parent.logWeights[i];
          acceptedWeights.push(parent.logWeights[i]);
          n_eff++;
        } else {
          out[i] = NaN;
          outWnn[i] = -Infinity;
        }
      }
      const lseA = acceptedWeights.length
        ? empirical.logSumExp(Float64Array.from(acceptedWeights))
        : -Infinity;
      const lseT = empirical.logSumExp(Float64Array.from(totalWeights));
      const logShift = isFinite(lseA) && isFinite(lseT) ? (lseA - lseT) : -Infinity;
      return scalarMeasureN(out, {
        logWeights: outW,
        logTotalmass: parentLTM + logShift,
        n_eff: n_eff,
      });
    }
    let anyNaN = false;
    for (let i = 0; i < parentSamples.length; i++) {
      const x = parentSamples[i];
      if (x >= lo && x <= hi) { out[i] = x; n_eff++; }
      else { out[i] = NaN; anyNaN = true; }
    }
    let logWeights: Float64Array | null = null;
    if (anyNaN) {
      logWeights = new Float64Array(parentSamples.length);
      for (let i = 0; i < parentSamples.length; i++) {
        logWeights[i] = Number.isNaN(out[i]) ? -Infinity : 0;
      }
    }
    const logShift = n_eff > 0
      ? Math.log(n_eff / parentSamples.length)
      : -Infinity;
    return scalarMeasureN(out, {
      logWeights: logWeights,
      logTotalmass: parentLTM + logShift,
      n_eff: n_eff,
    });
  });
}

module.exports = {
  matPushfwd,
  matTruncate,
};
