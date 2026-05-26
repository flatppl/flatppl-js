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
  setBoundsForMat,
  resolveFnBody,
} = shared;

function matPushfwd(name: string, d: DerivationPushfwd, ctx: any) {
  // pushfwd(f, M): the pushforward of M through function f. Per spec
  // §06, samples are { f(x) : x ~ M }. We get M's samples via the
  // recursive getMeasure, then run one batched evaluateN over f's
  // body with refArrays binding f's param name to M's samples — same
  // mass / logWeights / n_eff propagate through unchanged (the
  // pushforward map preserves total mass; bijection or not).
  //
  // For density evaluation, density.walkPushfwd consults f's
  // bijection annotation via opts.resolveBijection (set up by
  // matLogdensityof / matBayesupdate when they encounter pushfwd in
  // the expanded IR).
  const fBinding = ctx.bindings && ctx.bindings.get(d.fnRef);
  if (!fBinding) {
    return Promise.reject(new Error(`pushfwd: function binding '${d.fnRef}' not found`));
  }
  const fnInfo = resolveFnBody(fBinding, ctx.bindings);
  if (!fnInfo) {
    return Promise.reject(new Error(`pushfwd: function binding '${d.fnRef}' has no callable body`
      + ` (type=${fBinding.type})`));
  }
  return ctx.getMeasure(d.from).then((M: any) => {
    if (!M.samples) {
      return Promise.reject(new Error(`pushfwd: base measure '${d.from}' is not scalar `
        + `(record/tuple/iid not yet supported for first-class pushfwd materialisation)`));
    }
    return ctx.sendWorker({
      type: 'evaluateN',
      ir: fnInfo.body,
      count: M.samples.length,
      refArrays: { [fnInfo.paramName]: M.samples },
    }).then((reply: any) => {
      return measureFromReply(reply, M.samples.length, {
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
      const seed = nameSeed(d.from + '|truncate', ctx.rootSeed);

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
      return collectRefArrays(expanded, ctx.fixedValues, ctx.getMeasure)
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
