'use strict';

// model-spec.ts — build a ModelView spec from a classified 'bayesupdate'
// derivation for the backend:'mh' posterior path.
//
// buildPosteriorSpec(d, ctx) → { latents, logLikelihood }
//
// `latents` — transitively-required `draw`-type bindings of the prior,
// each carrying distOp, resolved scalar params, support, and discrete flag.
// Enumerates by walking binding.deps from the prior root name (mirrors
// buildSampleChain's dep walk in orchestrator.ts:195–265), using the same
// classifyForChain + lowerExpr approach to extract distIR.
//
// `logLikelihood(theta)` — synchronously scores the likelihood body at a
// single theta point using density.logDensityN with refArrays built from
// theta. Uses the body obtained from clm.lowerMeasure (the same lowering
// matBayesupdate uses for the IS path, so identical boundary coverage).

import type { IRNode } from './engine-types';

const orchestrator = require('./orchestrator.ts');
const transforms    = require('./transforms.ts');
const density       = require('./density.ts');
const clm           = require('./clm.ts');

const { lowerExpr }             = require('./lower.ts');
const { DISCRETE_DISTRIBUTIONS } = orchestrator;

// Collect transitively-required `draw`-type bindings of `rootName`.
// Mirrors the DFS dep walk in orchestrator.buildSampleChain (orchestrator.ts:195–265).
// For each draw-type binding, extracts distIR via classifyForChain (same accessor
// used at orchestrator.ts:244), resolves scalar params via orchestrator.resolveIRToValue,
// and looks up support via transforms.supportOf.
function collectLatents(rootName: string, ctx: any): any[] {
  const bindings = ctx.bindings;
  const fixedValues = ctx.fixedValues || new Map();
  const seen    = new Set<string>();
  const latents: any[] = [];

  function visit(name: string) {
    if (seen.has(name)) return;
    seen.add(name);
    const b = bindings.get(name);
    if (!b) return;

    // Recurse deps first (topological order, same as buildSampleChain).
    for (const dep of (b.deps || [])) visit(dep);

    if (b.type === 'draw') {
      // Lower the RHS (same as orchestrator.ts:228).
      let rhsIR: IRNode | null = null;
      try {
        rhsIR = lowerExpr(b.effectiveValue || b.node.value);
      } catch (_) { return; }

      // Classify to extract distIR (same call as orchestrator.ts:244).
      const orc = require('./orchestrator.ts');
      const stepKind = orc._internal.classifyForChain(b, rhsIR, bindings);
      if (!stepKind || stepKind.kind !== 'sample') return;
      const distIR = stepKind.distIR;

      // Resolve scalar params from distIR.kwargs (same lookup resolveIRToValue uses).
      const params: Record<string, any> = {};
      if (distIR.kwargs && typeof distIR.kwargs === 'object') {
        for (const [k, expr] of Object.entries(distIR.kwargs)) {
          try {
            params[k] = orchestrator.resolveIRToValue(
              expr as any, bindings, fixedValues,
            );
          } catch (_) { /* leave unresolved */ }
        }
      }
      // Positional args (e.g. Normal(0, 1) without kwargs).
      if (distIR.args && Array.isArray(distIR.args) && distIR.args.length > 0
          && Object.keys(params).length === 0) {
        for (let i = 0; i < distIR.args.length; i++) {
          try {
            params[String(i)] = orchestrator.resolveIRToValue(
              distIR.args[i], bindings, fixedValues,
            );
          } catch (_) { /* leave unresolved */ }
        }
      }

      const discrete = DISCRETE_DISTRIBUTIONS && DISCRETE_DISTRIBUTIONS.has(distIR.op);
      let support: any = null;
      if (!discrete) {
        try {
          support = transforms.supportOf(distIR.op, params);
        } catch (_) {
          // Unknown distribution: mark discrete to force refusal downstream.
          latents.push({ name, distOp: distIR.op, params, support: null, discrete: true });
          return;
        }
      }
      latents.push({ name, distOp: distIR.op, params, support, discrete: !!discrete });
    }
  }

  visit(rootName);
  return latents;
}

// Build { latents, logLikelihood } from a classified 'bayesupdate' derivation
// and the materialiser ctx. Called by the matBayesupdate MH branch.
//
// Real accessor names confirmed against the codebase:
//   orchestrator.resolveIRToValue  — ir-shared.ts:214, re-exported at orchestrator.ts:523
//   binding.deps                   — orchestrator.ts:217 (confirmed field name)
//   classifyForChain               — orchestrator.ts:244, _internal export
//   d.bodyIR / d.bodyName          — derivations.ts:3758–3765 (classifyBayesupdate return)
//   d.paramKwargs                  — same (the kernel's parametric input names)
//   clm.lowerMeasure(body, ctx, {derivation:d}) — mat-density.ts:165 (IS path)
//   density.logDensityN(ir, value, refArrays, count, opts) — density.ts:2407
function buildPosteriorSpec(d: any, ctx: any): { latents: any[]; logLikelihood: (theta: any) => number } {
  const latents = collectLatents(d.from, ctx);
  for (const l of latents) {
    if (l.discrete) {
      throw new Error(
        `backend 'mh' (continuous): latent '${l.name}' is discrete (${l.distOp}); `
        + `use backend 'is' or a discrete MH path`,
      );
    }
  }

  // Resolve the observation value (same call as mat-density.ts:169).
  const observed = orchestrator.resolveIRToValue(d.obsIR, ctx.bindings, ctx.fixedValues || new Map());

  // Lower the likelihood body once (same as mat-density.ts:165, IS path).
  // lowerMeasure expands bodyIR/bodyName, inlines derived value bindings,
  // and declares the kernel's parametric inputs as boundary inputs (the
  // boundary-feeding fix, audit §3 / H1/H6). node.body is ready for
  // density.logDensityN with refArrays feeding the boundary inputs.
  const node = clm.lowerMeasure(d.bodyIR || d.bodyName, ctx, { derivation: d });
  if (!node) {
    throw new Error('buildPosteriorSpec: cannot expand the likelihood body into measure IR');
  }

  // The kernel's parametric input names (boundary names in node.body).
  // d.paramKwargs are the call-site names (confirmed: derivations.ts:3649–3651).
  const paramNames: string[] = d.paramKwargs || [];

  function logLikelihood(theta: any): number {
    // Build refArrays: each latent's current value as a Float64Array(1).
    // The density walker resolves boundary refs from refArrays per atom (count=1).
    const refArrays: Record<string, Float64Array> = {};
    for (const nm of paramNames) {
      if (!(nm in theta)) {
        throw new Error(
          `logLikelihood: required kernel kwarg '${nm}' has no corresponding `
          + `latent value in theta (available: ${Object.keys(theta).join(', ') || '(none)'}). `
          + `Kwarg names must match latent names in the model.`,
        );
      }
      const val = theta[nm];
      refArrays[nm] = Float64Array.from([typeof val === 'number' ? val : 0]);
    }
    try {
      const logps = density.logDensityN(node.body, observed, refArrays, 1, {});
      const lp = logps[0];
      return Number.isFinite(lp) ? lp : -Infinity;
    } catch (_) {
      return -Infinity;
    }
  }

  return { latents, logLikelihood };
}

// ---------------------------------------------------------------------------
// enumerateLatents(d, ctx) — new vector-aware latent enumerator for the
// async ModelView path.  Uses d.paramKwargs (the kernel's parametric input
// names — confirmed at derivations.ts:3649-3651) as the ordered latent list.
//
// Key insight: latent derivations may be alias chains (e.g. theta → __anon5
// (iid) → __anon4 (sample)); tau may be alias → normalize → truncate →
// sample.  We walk the chain to find the effective kind (iid or scalar) and
// to determine the support.
//
// For each param name:
//   • Effective kind is 'iid'  → vector with dims from the iid derivation
//   • Otherwise               → scalar
//
// Support: walkDerivChainForSupport follows alias/normalize links and reads
// the setDescr from a truncate node (positive/interval) or falls back to
// supportOf on the leaf distribution.
// ---------------------------------------------------------------------------

// Walk the derivation chain starting at `name` and return support info.
// Returns { isIid, dims, elementBaseName } for an iid latent, or a support
// object { kind: 'real'|'positive'|'interval', a?, b? } for a scalar.
function walkDerivChain(name: string, derivations: any): any {
  const visited = new Set<string>();
  let cur: string = name;
  while (cur && !visited.has(cur)) {
    visited.add(cur);
    const d = derivations[cur];
    if (!d) return { kind: 'real' };
    if (d.kind === 'alias') { cur = d.from; continue; }
    if (d.kind === 'normalize') { cur = d.from; continue; }
    if (d.kind === 'iid') {
      return { isIid: true, dims: d.dims as number[], elementBaseName: d.from as string };
    }
    if (d.kind === 'truncate') {
      const s = d.setDescr;
      if (s) {
        const lo = (s.lo != null && Number.isFinite(+s.lo)) ? +s.lo : -Infinity;
        const hi = (s.hi != null && Number.isFinite(+s.hi)) ? +s.hi : Infinity;
        if (lo === 0 && !Number.isFinite(hi)) return { kind: 'positive' };
        if (!Number.isFinite(lo) && hi === 0) return { kind: 'real' }; // negative half-line: no standard transform; use real as fallback
        if (Number.isFinite(lo) && Number.isFinite(hi)) return { kind: 'interval', a: lo, b: hi };
      }
      cur = d.from; continue;
    }
    if (d.kind === 'sample') {
      const distIR = d.distIR;
      if (!distIR) return { kind: 'real' };
      try {
        return transforms.supportOf(distIR.op, {});
      } catch (_) { return { kind: 'real' }; }
    }
    // Unrecognised kind: treat as scalar real
    return { kind: 'real' };
  }
  return { kind: 'real' };
}

// Infer support for an element of an iid latent from the iid's base name.
// The base derivation should be 'sample' with a distIR.
function elementSupportOf(elementBaseName: string, derivations: any): any {
  return walkDerivChain(elementBaseName, derivations);
}

function enumerateLatents(d: any, ctx: any): any[] {
  const paramNames: string[] = d.paramKwargs || [];
  const derivations = ctx.derivations || {};
  const result: any[] = [];

  for (const name of paramNames) {
    const info = walkDerivChain(name, derivations);

    if (info && info.isIid) {
      // Vector latent
      const dims: number[] = info.dims || [1];
      const totalDim = dims.reduce((a: number, b: number) => a * b, 1);
      const support = elementSupportOf(info.elementBaseName, derivations);
      result.push({ name, shape: { kind: 'vector', dims: [totalDim] }, support });
    } else {
      // Scalar latent — support from chain walk
      const support = info || { kind: 'real' };
      result.push({ name, shape: { kind: 'scalar' }, support });
    }
  }

  return result;
}

module.exports = { buildPosteriorSpec, collectLatents, enumerateLatents };
