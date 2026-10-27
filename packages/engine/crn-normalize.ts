'use strict';
// Per-parameter normalizers for §06 weighted Lebesgue boxes. Exact moving
// windows, degree-checked tensor quadrature and exponential-affine integrals
// share one expression across sampling and density routes. Unresolved shapes
// refuse before inference; a pooled mass is not the conditional mass Z(θ).

const { mapIR } = require('./ir-walk.ts');

// Refuse rather than emit an expression bigger than this. A weight body large
// enough to blow the budget is a real feature request (a sum over the sample
// axis instead of M inlined copies), not something to silently shrink M for.
const CRN_NODE_BUDGET = 400000;

// =====================================================================
// Shape recognition
// =====================================================================

// Unwrap the reference-measure slot of a `weighted` to the "Lebesgue over a
// box" reading, returning its support set IR. Three spellings denote it, and
// all three have effective base density ≡ 1, so Z = ∫_S f dx with no extra
// factor:
//   • `Lebesgue(support = S)` — written directly.
//   • `logweighted(log(b−a), Uniform(interval(a,b)))` — the #307 shift that
//     scales Uniform's 1/(b−a) back to 1.
//   • `Uniform(interval(a, b))` with b − a = 1 — the lowering's unshifted form,
//     where Uniform's density is already 1.
// A NON-unit bare `Uniform` is declined: its density is 1/(b−a), so reading it
// as Lebesgue would drop that factor.
function lebesgueBoxSupport(refIR: any): any | null {
  let ref = refIR;
  let unwrappedShift = false;
  if (ref && ref.kind === 'call' && ref.op === 'logweighted'
      && Array.isArray(ref.args) && ref.args.length === 2) {
    ref = ref.args[1];
    unwrappedShift = true;
  }
  if (!ref || ref.kind !== 'call') return null;
  const supp = (ref.kwargs && ref.kwargs.support)
    || (Array.isArray(ref.args) ? ref.args[0] : null);
  if (!supp) return null;
  if (ref.op === 'Lebesgue') return supp;
  if (ref.op !== 'Uniform') return null;
  if (unwrappedShift) return supp;
  // Bare Uniform: only the unit interval reads as Lebesgue.
  const ax = _parseAxes(supp);
  if (!ax || ax.length !== 1 || ax[0].kind !== 'finite') return null;
  return Math.abs((ax[0].hi - ax[0].lo) - 1) < 1e-12 ? supp : null;
}

// Axis list for a support set IR, via mat-density's parseTruncationBox — one
// reader for the truncation-region and the CRN paths, so a shape one accepts
// cannot be a shape the other mis-parses. Required lazily: mat-density
// require()s this module.
function _parseAxes(setIR: any): any[] | null {
  const { parseTruncationBox } = require('./mat-density.ts');
  return parseTruncationBox(setIR);
}

type CrnShape = {
  weightParams: string[];
  weightBody: any;
  axes: Array<{ lo: number; hi: number; kind: string }>;
  // Is the base's variate an ARRAY (a positional cartprod/cartpow support) or a
  // scalar (a bare `interval`)? `axes` reports one axis for both, and a
  // one-parameter weight binds differently in each case — see `bindWeightAt`.
  arrayVariate: boolean;
};

// Recognise `normalize(weighted(<functionof weight>, <Lebesgue over a
// finite box>))` and return its weight function plus the box axes. null for
// every other shape, which leaves the caller's existing fallback in place.
function crnRecognize(node: any): CrnShape | null {
  if (!node || node.kind !== 'call' || node.op !== 'normalize'
      || !Array.isArray(node.args) || node.args.length !== 1) return null;
  const inner = node.args[0];
  if (!inner || inner.kind !== 'call' || inner.op !== 'weighted'
      || !Array.isArray(inner.args) || inner.args.length !== 2) return null;
  const fn = inner.args[0];
  if (!fn || fn.kind !== 'call' || fn.op !== 'functionof'
      || !Array.isArray(fn.params) || fn.params.length !== 1 || !fn.body) return null;
  const supp = lebesgueBoxSupport(inner.args[1]);
  if (!supp) return null;
  const axes = _parseAxes(supp);
  if (!axes || axes.length === 0) return null;
  // Every axis must be bounded: the integration rules cover only finite boxes.
  // Unbounded supports need a separate change of variables or analytic rule.
  for (const a of axes) if (a.kind !== 'finite') return null;
  const { supportIsArrayVariate } = require('./mat-density.ts');
  return {
    weightParams: fn.params.slice(), weightBody: fn.body, axes,
    arrayVariate: supportIsArrayVariate(supp),
  };
}

// Does the weight actually depend on a latent? Only then is a per-θ normalizer
// needed, and only then may this path take a node away from the deterministic
// integration for a fixed weight.
//
// The weight's OWN parameters must come out first. mat-density's
// `weightedFixedWeightEnv` does not remove them, which is harmless where it is
// used (a lambda's parameter is `%local`, so `collectSelfRefs` cannot see it)
// and wrong here: a REIFIED weight — `functionof(body, m = m, c = c)` — carries
// its parameters as the BOUNDARY BINDING names in `self`, so every reified
// weight looked latent-dependent and pulled a θ-independent 2-D box onto this
// path, changing numbers that were already right.
function crnWeightIsThetaDependent(shape: CrnShape, ctx: any): boolean {
  const orchestrator = require('./orchestrator.ts');
  const { isFunctionLikeBinding } = require('./materialiser-shared.ts');
  const params = new Set(shape.weightParams);
  for (const n of orchestrator.collectSelfRefs(shape.weightBody)) {
    if (params.has(n)) continue;                                  // the variate
    const b = ctx && ctx.bindings && ctx.bindings.get(n);
    if (isFunctionLikeBinding(b)) continue;                        // a called fn
    if (ctx && ctx.fixedValues && ctx.fixedValues.has(n)) continue; // a constant
    return true;
  }
  return false;
}

// =====================================================================
// The emitted expression
// =====================================================================

// Bind the weight function's parameters to one sample point, following the
// SAME convention density.ts's walkWeighted and mat-density's makeIntegrandND
// use — the single parameter takes the variate whole — so the normalizer integrates
// the function the density path scores. `arrayVariate` decides what "whole"
// means at ONE axis: a `cartprod(interval(...))` support gives a length-1
// VECTOR (§03/§07), a bare `interval(...)` support a scalar.
function bindWeightAt(params: string[], body: any, pt: number[], arrayVariate?: boolean): any {
  const bound: Record<string, any> = {};
  if (arrayVariate) {
    bound[params[0]] = {
      kind: 'call', op: 'vector',
      args: pt.map((c) => ({ kind: 'lit', value: c })),
    };
  } else {
    bound[params[0]] = { kind: 'lit', value: pt[0] };
  }
  return mapIR(body, (n: any) => (n && n.kind === 'ref' && n.name in bound ? bound[n.name] : n));
}

// Σ terms as a BALANCED binary tree of `add`, not a left-leaning chain. Depth
// log₂M rather than M: the engine's IR walkers (collectRefNames, mapIR, the
// worker's evaluator) all recurse, and a chain of M = 2048 terms overflowed the
// stack — which `likWith` then swallowed to −∞, so the model reported a constant
// chain rather than an error. Pairwise summation is also the more accurate
// reduction.
function balancedAdd(terms: any[]): any {
  let level = terms;
  while (level.length > 1) {
    const next: any[] = [];
    for (let i = 0; i < level.length; i += 2) {
      next.push(i + 1 < level.length
        ? { kind: 'call', op: 'add', args: [level[i], level[i + 1]] }
        : level[i]);
    }
    level = next;
  }
  return level[0];
}

function nodeCount(n: any): number {
  if (n == null || typeof n !== 'object') return 0;
  let c = 1;
  for (const k in n) { const v = n[k]; if (v && typeof v === 'object') c += nodeCount(v); }
  return c;
}

// Return log Z(θ). A recognized but unresolved latent-dependent weight must
// throw: null would let callers use a pooled divisor and change the posterior.
function boxIntegral(node: any): any | null {
  const shape = crnRecognize(node);
  if (!shape) return null;
  if (shape.axes.length > 3) throw unresolved(shape, 'more than three integration coordinates');
  if (shape.axes.some((a) => !Number.isFinite(a.hi - a.lo) || a.hi <= a.lo)) {
    throw unresolved(shape, 'the box has no finite positive representable width');
  }
  // A weight with a discontinuity: the fixed sample cannot see a window that
  // falls between its points, so split at the breakpoints instead
  // (piecewise-mass.ts), or refuse where that is not possible.
  const piecewise = require('./piecewise-mass.ts');
  const param = shape.weightParams[0];
  if (shape.axes.length > 1) {
    if (piecewise.jumpsInVariate(shape.weightBody, param)) {
      throw new Error('normalize(weighted(f, Lebesgue(box))): the weight depends on a latent '
        + 'and may jump in the ' + shape.axes.length + '-D variate, so its '
        + 'mass Z(θ) cannot be split at the region and a fixed sample could step over it; '
        + 'Z(θ) is unresolved (spec §06)');
    }
  } else {
    const bp = piecewise.weightBreakpoints(shape.weightBody, param, shape.arrayVariate);
    if (bp.refuse) {
      throw new Error('normalize(weighted(f, Lebesgue(box))): the weight depends on a latent '
        + 'and has ' + bp.refuse + ', which gives no breakpoint to split Z(θ) at, and a fixed '
        + 'sample could step over the region; Z(θ) is unresolved (spec §06).');
    }
    if (bp.exprs.length > 0) {
      const at = piecewise.variesBetweenCuts(shape.weightBody, param, shape.arrayVariate);
      if (at !== null) throw piecewise.variesError('Lebesgue(box)', at);
      const expr = piecewise.boxMassExpr(param, shape.weightBody, shape.axes[0].lo, shape.axes[0].hi,
        bp.exprs, shape.arrayVariate);
      _checkBudget(expr);
      return { logMass: { kind: 'call', op: 'log', args: [expr] } };
    }
  }
  const { boxLogMass } = require('./box-mass.ts');
  const expr = boxLogMass(shape.weightBody, param, shape.axes, shape.arrayVariate, CRN_NODE_BUDGET);
  if (expr === null) throw unresolved(shape, 'no exact box integral covers this weight within the expression budget');
  _checkBudget(expr);
  return expr;
}

function unresolved(shape: CrnShape, detail: string): Error {
  const { engineLimitation } = require('./limitations.ts');
  const loc = shape.weightBody.loc?.start;
  const at = loc ? ' at line ' + (loc.line + 1) + ', column ' + (loc.col + 1) : '';
  return engineLimitation('normalize(weighted(f, Lebesgue(box)))', 'parameter-dependent integration',
    detail + at + '; Z(θ) is unresolved (spec §06). Fixed rules can miss narrow peaks; refusing a pooled divisor');
}

function crnNormalizeLogMassExpr(node: any): any | null {
  return boxIntegral(node)?.logMass ?? null;
}

// Density can cancel an exponential offset before evaluation, retaining small
// normalization terms even when the original exponent is enormous.
function crnNormalizeDensity(node: any): any | null {
  const integral = boxIntegral(node);
  if (!integral) return null;
  const inner = node.args[0];
  if (!integral.logWeight) return { logMass: integral.logMass, inner };
  const fn = inner.args[0];
  return {
    logMass: integral.logShift,
    inner: { ...inner, op: 'logweighted', args: [{ ...fn, body: integral.logWeight }, inner.args[1]] },
  };
}

// Exact tensor rules inline the weight at quadrature points; piecewise-constant
// windows inline it once per piece. Both must fit the same expression budget.
function _checkBudget(expr: any) {
  const n = nodeCount(expr);
  if (n > CRN_NODE_BUDGET) {
    throw new Error('normalize(weighted(f, …)): the weight depends on a latent, so its mass Z(θ) '
      + 'inlines the weight body at integration points — ' + n + ' nodes, '
      + 'over the budget of ' + CRN_NODE_BUDGET + '. Refusing rather than falling back to an '
      + 'estimator that cannot see the weight\'s discontinuities (spec §06).');
  }
}

module.exports = {
  _checkBudget,
  crnNormalizeLogMassExpr,
  crnNormalizeDensity,
  crnRecognize,
  crnWeightIsThetaDependent,
  lebesgueBoxSupport,
  CRN_NODE_BUDGET,
  // Shared with leaf-mass-quad.ts, which emits the same kind of per-θ mass
  // expression over a probability-leaf base: one binder, one reduction and one
  // budget test, so the two builders cannot drift on the §06 weight arity, on
  // pairwise summation, or on what counts as too large an expression.
  bindWeightAt,
  nodeCount,
  balancedAdd,
};
