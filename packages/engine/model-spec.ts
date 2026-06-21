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
const sampler       = require('./sampler.ts');

const { lowerExpr }             = require('./lower.ts');
const { DISCRETE_DISTRIBUTIONS } = orchestrator;
const bijReg                    = require('./bijection-registry.ts');

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
        // Uniform's bounds live in its support SET, which supportOf cannot
        // read from resolved scalar params — extract them from the IR.
        if (distIR.op === 'Uniform') {
          support = uniformSupportFromDistIR(distIR, bindings, fixedValues);
        }
        if (!support) {
          try {
            support = transforms.supportOf(distIR.op, params);
          } catch (_) {
            // Unknown distribution: mark discrete to force refusal downstream.
            latents.push({ name, distOp: distIR.op, params, support: null, discrete: true });
            return;
          }
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

// A `Uniform` draw's support is the SET passed as its `support` argument
// (spec 08-distributions "Uniform"): `Uniform(interval(0.1, 20))` is supported
// on [0.1, 20]. transforms.supportOf cannot recover it — resolveIRToValue
// throws on an `interval` call ("not evaluable"), so the bounds must be read
// from the distribution IR's support arg here. Resolves the interval's two
// endpoint exprs (which DO resolve — they are literals or fixed bindings).
// Returns { kind:'interval', a, b } for a finite interval, else null (caller
// falls back). `bindings`/`fixedValues` may be undefined when no resolution
// context is threaded through; literal endpoints still resolve.
function uniformSupportFromDistIR(distIR: any, bindings: any, fixedValues: any): any {
  if (!distIR) return null;
  const supIR = (distIR.kwargs && distIR.kwargs.support)
    || (Array.isArray(distIR.args) && distIR.args.length > 0 ? distIR.args[0] : null);
  if (!supIR) return null;
  const fv = fixedValues || new Map();
  let bounds: [number, number] | null = null;
  try {
    // Shared region parser (transforms.regionBounds) recognises interval(...)
    // and the region consts; endpoints resolve in the orchestrator's context.
    bounds = transforms.regionBounds(
      supIR, (a: any) => orchestrator.resolveIRToValue(a, bindings, fv));
  } catch (_) { return null; }       // unresolvable endpoint
  if (!bounds) return null;          // not a recognised region
  const [lo, hi] = bounds;
  // A Uniform needs a finite, non-degenerate interval (0 < λ(S) < ∞); a
  // half-line/real region (posreals, reals) is improper — fall back rather
  // than hand the transform infinite bounds.
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
    return { kind: 'interval', a: lo, b: hi };
  }
  return null;
}

// Image support of pushfwd(f, base) for a MONOTONE elementary f, derived by
// probing f over the base support's bounds. The samplers explore an
// unconstrained ℝ mapped onto the latent's support; if a pushfwd latent's
// support is left at the default `real` (e.g. the Pareto `pushfwd(0.1·exp,
// Exponential)` whose true support is [0.1, ∞)), the unconstraining transform
// is the identity and the chain proposes out-of-support points where the prior
// density is −∞ — chains stall and SMC's β=0 cloud loses all mass. Returns a
// support object, or null when the image can't be confidently derived
// (non-monotone, NaN, unknown base) → the caller falls back to `real` (never
// worse than the pre-existing default).
//
// Symbolic exact-bounds path (hardening): first attempts bijection-registry
// invertExpr on fnBody to confirm the function is a provably invertible
// elementary bijection. If successful, the bijectivity check is exact (no
// heuristic multi-point probe) and finite base bounds yield exact closed image
// bounds via direct f evaluation. The numeric magnitude-probe is kept as the
// fallback for infinite base ends and for functions invertExpr cannot handle.
function pushfwdImageSupport(fnBody: any, paramName: string, baseSupport: any): any {
  const f = (x: number): number => {
    try { const v = sampler.evaluateExpr(fnBody, { [paramName]: x }); return typeof v === 'number' ? v : NaN; }
    catch (_) { return NaN; }
  };
  let blo: number, bhi: number;
  if (baseSupport.kind === 'positive') { blo = 0; bhi = Infinity; }
  else if (baseSupport.kind === 'interval') { blo = baseSupport.a; bhi = baseSupport.b; }
  else if (baseSupport.kind === 'real') { blo = -Infinity; bhi = Infinity; }
  else return null;

  // --- Symbolic exact-bounds path ---
  // invertExpr with the %local namespace used for fn-body parameters.
  const freeRef = { name: paramName, ns: '%local' };
  const outputValue = { kind: 'lit', value: 0 };   // placeholder; only bijectivity matters here
  const invResult = bijReg.invertExpr({ outputExpr: fnBody, freeRef, outputValue });

  if (invResult != null) {
    // Function is a provably-invertible elementary bijection.
    // Determine monotonicity direction with two evaluation points.
    const xl = Number.isFinite(blo) ? blo : -1;
    const xh = Number.isFinite(bhi) ? bhi : 1;
    if (!(xh > xl)) return null;
    const fLo = f(xl), fHi = f(xh);
    if (!Number.isFinite(fLo) || !Number.isFinite(fHi)) return null;
    const inc = fHi > fLo;

    // Image bound at one base end. Finite base end → exact f(end). Infinite
    // base end → numeric magnitude-probe (same heuristic as the fallback path,
    // retained because determining the asymptotic limit symbolically would
    // require annotating the ELEMENTARY_BIJECTIONS table with limit behaviour).
    function imageEndSym(bound: number, dir: number): any {
      if (Number.isFinite(bound)) {
        const v = f(bound);
        return Number.isFinite(v) ? { v } : null;
      }
      const a = f(dir * 15), b = f(dir * 30), c = f(dir * 60);
      if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return { inf: dir };
      const d1 = Math.abs(b - a), d2 = Math.abs(c - b);
      if (d2 < d1 * 0.5 && d2 < 1e-6) return { v: c };   // finite asymptote
      return { inf: c >= a ? +1 : -1 };
    }

    const atLo = imageEndSym(blo, -1);
    const atHi = imageEndSym(bhi, +1);
    if (atLo == null || atHi == null) return null;

    const lowImg = inc ? atLo : atHi;
    const highImg = inc ? atHi : atLo;
    const lowInf = ('inf' in lowImg), highInf = ('inf' in highImg);
    if (!lowInf && !highInf) {
      return (highImg.v > lowImg.v) ? { kind: 'interval', a: lowImg.v, b: highImg.v } : null;
    }
    if (!lowInf && highInf) {
      return (Math.abs(lowImg.v) < 1e-12) ? { kind: 'positive' } : { kind: 'greaterThan', lo: lowImg.v };
    }
    if (lowInf && !highInf) {
      return (Math.abs(highImg.v) < 1e-12) ? { kind: 'real' } : { kind: 'lessThan', hi: highImg.v };
    }
    return { kind: 'real' };   // both ends infinite
  }

  // --- Numeric fallback path (for functions invertExpr cannot handle) ---
  // Monotonicity over the base support (finite proxies for ±∞).
  const xl = Number.isFinite(blo) ? blo : -30;
  const xh = Number.isFinite(bhi) ? bhi : 30;
  if (!(xh > xl)) return null;
  const vals = [0, 0.25, 0.5, 0.75, 1].map((t) => f(xl + (xh - xl) * t));
  if (!vals.every(Number.isFinite)) return null;
  let inc = true, dec = true;
  for (let i = 1; i < vals.length; i++) {
    if (!(vals[i] > vals[i - 1])) inc = false;
    if (!(vals[i] < vals[i - 1])) dec = false;
  }
  if (!inc && !dec) return null;   // non-monotone → no 1-D interval image

  // Image bound at one base end. Finite base end → closed f(end). Infinite base
  // end → converging (finite asymptote, e.g. exp(−∞)=0) vs diverging (±∞),
  // judged by whether |Δf| collapses across growing magnitudes.
  function imageEnd(bound: number, dir: number): any {
    if (Number.isFinite(bound)) return { v: f(bound) };
    const a = f(dir * 15), b = f(dir * 30), c = f(dir * 60);
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return { inf: dir };
    const d1 = Math.abs(b - a), d2 = Math.abs(c - b);
    if (d2 < d1 * 0.5 && d2 < 1e-6) return { v: c };   // finite asymptote
    return { inf: c >= a ? +1 : -1 };
  }
  const atLo = imageEnd(blo, -1), atHi = imageEnd(bhi, +1);
  const lowImg = inc ? atLo : atHi;   // image lower end ← base lower end iff increasing
  const highImg = inc ? atHi : atLo;
  const lowInf = ('inf' in lowImg), highInf = ('inf' in highImg);
  if (!lowInf && !highInf) {
    return (highImg.v > lowImg.v) ? { kind: 'interval', a: lowImg.v, b: highImg.v } : null;
  }
  if (!lowInf && highInf) {
    return (Math.abs(lowImg.v) < 1e-12) ? { kind: 'positive' } : { kind: 'greaterThan', lo: lowImg.v };
  }
  if (lowInf && !highInf) {
    return (Math.abs(highImg.v) < 1e-12) ? { kind: 'real' } : { kind: 'lessThan', hi: highImg.v };
  }
  return { kind: 'real' };   // both ends infinite
}

// Walk the derivation chain starting at `name` and return support info.
// Returns { isIid, dims, elementBaseName } for an iid latent, or a support
// object { kind: 'real'|'positive'|'interval'|'greaterThan'|'lessThan', … } for
// a scalar. `ctx` (optional) supplies bindings/fixedValues so a Uniform leaf's
// interval bounds resolve AND enables the pushfwd-image-support derivation.
function walkDerivChain(name: string, derivations: any, ctx?: any): any {
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
    if (d.kind === 'pushfwd') {
      // Latent support = image of the base support under f. Without bindings
      // (no fn body) or for a non-derivable image, fall back to real.
      const bindings = ctx && ctx.bindings;
      const fb = bindings && bindings.get(d.fnRef);
      if (fb && fb.ir && fb.ir.kind === 'call' && fb.ir.op === 'functionof'
          && Array.isArray(fb.ir.params) && fb.ir.params.length === 1 && fb.ir.body) {
        const baseSupport = walkDerivChain(d.from, derivations, ctx);
        if (baseSupport && baseSupport.kind && !baseSupport.isIid) {
          const img = pushfwdImageSupport(fb.ir.body, fb.ir.params[0], baseSupport);
          if (img) return img;
        }
      }
      return { kind: 'real' };
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
      if (distIR.op === 'Uniform') {
        const s = uniformSupportFromDistIR(
          distIR, ctx && ctx.bindings, ctx && ctx.fixedValues);
        if (s) return s;
      }
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
function elementSupportOf(elementBaseName: string, derivations: any, ctx?: any): any {
  return walkDerivChain(elementBaseName, derivations, ctx);
}

// Collect the STOCHASTIC `draw`-type bindings transitively feeding `rootName`,
// in dependency order. These — NOT the kernel's parametric inputs
// (d.paramKwargs) — are the model's free latents. The distinction matters when
// a kernel input is a DERIVED quantity: e.g. `sigma = sqrt(sigma2)` with
// `sigma2 ~ InverseGamma(...)` makes `sigma2` the latent and `sigma` a
// deterministic transform. Parameterising by draws (sigma2) rather than kernel
// inputs (sigma) is what lets the per-draw scorer evaluate the prior without
// densifying `sqrt` in a `lawof` record.
function collectDrawNames(rootName: string, ctx: any): string[] {
  const bindings = ctx.bindings;
  const seen = new Set<string>();
  const out: string[] = [];
  (function visit(n: string) {
    if (seen.has(n)) return;
    seen.add(n);
    const b = bindings.get(n);
    if (!b) return;
    for (const dep of (b.deps || [])) visit(dep);
    if (b.type === 'draw') out.push(n);
  })(rootName);
  return out;
}

// Resolve the prior's latent SOURCES as { latentName, measureName } pairs.
// `latentName` is the free-variate name the kernel and the point record key on;
// `measureName` is the binding whose measure scores that variate. They differ
// only for a joint-of-distributions prior (see below); on the canonical draw
// path the mapping is identity.
//
//   • Canonical draw path: prior = lawof(record-of-draws). Each stochastic
//     `~` draw transitively feeding the prior IS both the latent and its own
//     measure (e.g. sigma2 ~ InverseGamma → sigma2 scores sigma2).
//
//   • Joint-of-distributions prior: prior = joint(theta1 = D1, theta2 = D2, …)
//     built directly from distributions, with the variates supplied as
//     `elementof` boundary inputs (06-measure-algebra.md §joint — "the keyword
//     form names the component variates"; those named variates are what
//     bayesupdate combines with the like-named kernel inputs). This classifies
//     as a `record`-kind derivation whose fields map each variate LABEL to the
//     distribution binding. There are no `~` draws, so the latents are the field
//     labels and each label's measure is the binding it points at.
function priorLatentSources(d: any, ctx: any): Array<{ latentName: string; measureName: string }> {
  const draws = collectDrawNames(d.from, ctx);
  if (draws.length > 0) return draws.map((nm) => ({ latentName: nm, measureName: nm }));
  const der = (ctx.derivations || {})[d.from];
  if (der && der.kind === 'record' && der.fields && typeof der.fields === 'object') {
    return Object.keys(der.fields).map((label) => ({ latentName: label, measureName: der.fields[label] }));
  }
  return [];
}

function enumerateLatents(d: any, ctx: any): any[] {
  // Latents are the prior's free variates — the stochastic draws on the
  // canonical path, or the joint's named fields for a joint-of-distributions
  // prior. priorLatentSources resolves both to { latentName, measureName }.
  const sources = priorLatentSources(d, ctx);
  const derivations = ctx.derivations || {};
  const result: any[] = [];

  const { recognizeCompositeIidDraw } = require('./composite-prior.ts');

  for (const { latentName, measureName } of sources) {
    const info = walkDerivChain(measureName, derivations, ctx);

    if (info && info.isIid) {
      // Vector latent
      const dims: number[] = info.dims || [1];
      const totalDim = dims.reduce((a: number, b: number) => a * b, 1);
      const support = elementSupportOf(info.elementBaseName, derivations, ctx);
      result.push({ name: latentName, measureName, shape: { kind: 'vector', dims: [totalDim] }, support });
      continue;
    }

    // A draw whose measure is a kernel-broadcast over a user function returning
    // an iid block (e.g. p ~ beta_row_K.(a,b), a [G,N] matrix of Betas). The
    // elementwise support comes from the inner builtin distribution; the total
    // size is determined from the materialised prior pool in buildModelView.
    const comp = recognizeCompositeIidDraw(measureName, ctx);
    if (comp) {
      let support: any;
      try { support = transforms.supportOf(comp.distOp, {}); } catch (_) { support = { kind: 'real' }; }
      result.push({ name: latentName, measureName, shape: { kind: 'vector', dims: [0], fromPool: true }, support });
      continue;
    }

    // Scalar latent — support from chain walk
    const support = info || { kind: 'real' };
    result.push({ name: latentName, measureName, shape: { kind: 'scalar' }, support });
  }

  return result;
}

// Detect whether the UNCONSTRAINED prior is an independent Gaussian with
// constant parameters — i.e. every latent is Normal(μ,σ) with numeric-literal
// μ,σ and real support (so the identity transform leaves it Normal). Returns a
// per-latent { name → { mu, sigma } } map (a vector latent shares one μ,σ across
// its iid elements), or null if any latent disqualifies. Used by the elliptical
// slice sampler to pick its EXACT Gaussian reference; null ⇒ the fitted path.
function leafSampleDist(name: string, derivations: any): any {
  let cur = name; const seen = new Set<string>();
  while (cur && !seen.has(cur)) {
    seen.add(cur);
    const d = derivations[cur];
    if (!d) return null;
    if (d.kind === 'alias' || d.kind === 'normalize' || d.kind === 'iid') { cur = d.from; continue; }
    if (d.kind === 'sample') return d.distIR || null;
    return null;   // truncate / pushfwd / kernelbroadcast / … → not a plain Normal
  }
  return null;
}

function detectGaussianPrior(d: any, ctx: any): Record<string, { mu: number; sigma: number }> | null {
  const latents = enumerateLatents(d, ctx);
  const derivations = ctx.derivations || {};
  const out: Record<string, { mu: number; sigma: number }> = {};
  const lit = (ir: any) => (ir && ir.kind === 'lit' && typeof ir.value === 'number') ? ir.value : null;
  for (const l of latents) {
    if (l.discrete) return null;
    if (!(l.support && l.support.kind === 'real')) return null;   // Normal latents are real-support
    const dist = leafSampleDist(l.measureName || l.name, derivations);
    if (!dist || dist.op !== 'Normal') return null;
    let muIR: any, sigIR: any;
    const kw = dist.kwargs || dist.kwargIRs;
    if (kw) { muIR = kw.mu ?? kw.loc ?? kw.mean; sigIR = kw.sigma ?? kw.scale ?? kw.sd; }
    const args = dist.args || dist.argIRs;
    if (Array.isArray(args)) { if (muIR == null) muIR = args[0]; if (sigIR == null) sigIR = args[1]; }
    const mu = lit(muIR), sigma = lit(sigIR);
    if (mu == null || sigma == null || !(sigma > 0)) return null;
    out[l.name] = { mu, sigma };
  }
  return Object.keys(out).length > 0 ? out : null;
}

module.exports = { buildPosteriorSpec, collectLatents, enumerateLatents, collectDrawNames, priorLatentSources, detectGaussianPrior, uniformSupportFromDistIR };
