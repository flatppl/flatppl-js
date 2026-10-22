'use strict';
// totalMassExpr — IR expression for the total mass Z(θ) of a superpose of
// weighted probability measures (06-measure-algebra §normalize/§superpose).
// Shared by the MCMC scorer (mcmc-density) and the IS density path (mat-density)
// so a normalize(...) whose mass depends on a latent is scored per-θ rather than
// baked as a materialised constant.
const orchestrator = require('./orchestrator.ts');
const { engineLimitation } = require('./limitations.ts');

// The density walker evaluates these CDF normalizers at the current atom.
// Both standalone density and MCMC must defer them rather than freeze a
// captured parameter's pooled mass during setup.
function canDeferTruncateNormalizer(inner: any): boolean {
  if (!inner || inner.kind !== 'call' || inner.op !== 'truncate'
      || !Array.isArray(inner.args) || inner.args.length !== 2) return false;
  const base = inner.args[0];
  if (!base || base.kind !== 'call' || typeof base.op !== 'string') return false;
  const sampler = require('./sampler.ts');
  if (!sampler.isKnownDistribution(base.op)) return false;
  // A discrete base's mass is a sum over the atoms in S (density.ts).
  if (!require('./forward-cdf.ts').hasCdf(base.op) && !sampler.lookupDistribution(base).discrete) return false;
  const set = inner.args[1];
  return set?.kind === 'call' && set.op === 'interval' && set.args?.length === 2
    || (set?.kind === 'const' || set?.kind === 'ref' && set.ns === 'self')
      && ['reals', 'posreals', 'nonnegreals'].includes(set.name);
}

// Build an IR EXPRESSION for the total mass Z of a measure IR, as a function of
// the current point θ (the weights may reference latents). Returns null when
// the structure isn't a closed-form mass we can express symbolically.
//
// Recognised: a superpose of weighted probability measures — the
// (logweights-null) `select{branches}` lowering or a bare `superpose{args}`,
// each branch `weighted(w_i, M_i)` with M_i a SAMPLEABLE_DISTRIBUTIONS leaf
// (total mass 1) or a constant-parameter `truncate` of one (total mass the CDF
// difference over the interval). Then Z(θ) = Σ_i w_i·Z_i. This is the ONLY case where
// a normalize's mass genuinely depends on a latent yet is closed-form (the
// Gaussian-mixture `normalize(superpose(weighted(theta, …), weighted(1-theta,
// …)))`); baking a CONSTANT Z there (materialised once) silently mis-scores
// every θ ≠ the materialised point — the normalizer's −N·log Z(θ) term fails to
// cancel the θ the weights put in the numerator. Spec §06: normalize(M) density
// shifts by −log totalmass(M), evaluated at the scored θ.
function totalMassExpr(ir: any): any {
  if (!ir || ir.kind !== 'call') return null;
  const op = ir.op;
  // Pushforward changes coordinates, not total mass (§06). locscale may
  // remain scoped until its callback arguments are bound.
  if (op === 'locscale' && ir.args?.length === 3) return totalMassExpr(ir.args[0]);
  if (op === 'pushfwd' && ir.args?.length === 2) return totalMassExpr(ir.args[1]);
  if (op === 'iid') {
    const count = iidCount(ir);
    if (count == null) return null;
    if (count === 0) return { kind: 'lit', value: 1 };
    const mass = totalMassExpr(ir.args[0]);
    return mass == null ? null : { kind: 'call', op: 'pow',
      args: [mass, { kind: 'lit', value: count }] };
  }
  // superpose / logweights-null select: Σ component masses. A select with
  // explicit logweights or a retain-mode selectorName is NOT a raw additive
  // superpose — fall back to the constant bake rather than guess its mass.
  const comps = (op === 'select' && ir.logweights == null && ir.selectorName == null)
    ? ir.branches
    : (op === 'superpose' ? ir.args : null);
  if (Array.isArray(comps) && comps.length > 0) {
    let acc: any = null;
    for (const c of comps) {
      const m = totalMassExpr(c);
      if (m == null) return null;
      acc = (acc == null) ? m : { kind: 'call', op: 'add', args: [acc, m] };
    }
    return acc;
  }
  // joint(M₁, …) is the INDEPENDENT PRODUCT (§06 "Joint composition":
  // "(M1⊗M2)(A×B) = M1(A)·M2(B)"), so its mass is ∏ᵢ mass(Mᵢ) — 1 for a joint
  // of probability measures. Without this arm a `weighted(theta, joint(…))`
  // had no expression at all and fell to the θ-CONSTANT materialised bake,
  // which mis-scores every θ but the materialised one and (on the sampling
  // side) tilted E[θ] to 3.4464 against a prior mean of 3.0.
  if (op === 'joint') {
    // Keyword form carries `fields: [{name, value}]`, positional form `args`.
    const parts: any[] = Array.isArray(ir.fields)
      ? ir.fields.map((f: any) => f.value) : ir.args;
    /* c8 ignore next -- defensive: a `joint` node always carries one of the two */
    if (!Array.isArray(parts) || parts.length === 0) return null;
    let acc: any = null;
    for (const p of parts) {
      const m = totalMassExpr(p);
      if (m == null) return null;
      acc = (acc == null) ? m : { kind: 'call', op: 'mul', args: [acc, m] };
    }
    return acc;
  }
  if (op === 'weighted' && Array.isArray(ir.args) && ir.args.length === 2) {
    // A function-of-variate weight (`weighted(x -> w(x), M)`, spec §06 — the
    // §12 generic-density idiom; derivations.ts's expandMeasureIR rewraps a
    // function-of-variate weight into this exact `functionof` shape, #307)
    // is NOT a scalar mass factor: mass(weighted(w, M)) here is ∫ w(x) dM(x),
    // not a pointwise value multiplying mass(M). Treating `ir.args[0]` as a
    // per-θ scalar (the mixture-branch case this function targets) would
    // hand a bare `functionof` node to the worker's scalar evaluator — fail
    // closed (defer to the runtime massFrom/quadrature path) rather than
    // mis-scoring the generic-density normalizer as a mixture weight.
    if (ir.args[0] && ir.args[0].kind === 'call' && ir.args[0].op === 'functionof') {
      return null;
    }
    const inner = totalMassExpr(ir.args[1]);
    if (inner == null) return null;
    // mass(weighted(w, M)) = w · mass(M)
    return { kind: 'call', op: 'mul', args: [ir.args[0], inner] };
  }
  // A superpose branch with a LITERAL weight lowers to logweighted(logw, M)
  // (the log-space form) rather than weighted(w, M); a mixture with any
  // literal-weight component therefore presents as
  // select{ weighted(θ, ·), logweighted(0, ·) }. mass(logweighted(logw, M)) =
  // exp(logw)·mass(M). Without this branch totalMassExpr returned null for
  // such a mixture, dropping the whole normalize to the θ-constant materialised
  // Z bake.
  if (op === 'logweighted' && Array.isArray(ir.args) && ir.args.length === 2) {
    // A callable logweight contributes an integral, just like a callable
    // ordinary weight. It is not a scalar mass multiplier.
    if (ir.args[0] && ir.args[0].kind === 'call' && ir.args[0].op === 'functionof') {
      return null;
    }
    const inner = totalMassExpr(ir.args[1]);
    if (inner == null) return null;
    return { kind: 'call', op: 'mul', args: [{ kind: 'call', op: 'exp', args: [ir.args[0]] }, inner] };
  }
  // truncate(M, S) restricts the support: §06 "Support restriction" gives
  // "ν(A) = M(A ∩ S). Does not normalize automatically", so the truncated
  // measure's TOTAL mass is M(S) — for a scalar leaf over interval(a, b) the
  // CDF difference F(b) − F(a), not 1. Without this arm a mixture with a
  // truncate component had no Σ_i w_i expression at all: the sampling route
  // fell to the POOLED divisor and `normalize` left the residue Z(θ_i)/E[Z] on
  // the atom weights, while the density route baked the materialised superpose's
  // logTotalmass at one θ. On the §06 mixture spelling with a Beta(2, 5) mixing
  // weight that read E[y] = 7.8501 (pooled 7.8550) against the exact
  // 7.7783296397108580, with E[p] tilted to 0.27692 off the prior's 2/7.
  if (op === 'truncate' && Array.isArray(ir.args) && ir.args.length === 2) {
    const logMass = positiveCountLogMassExpr(ir);
    if (logMass != null) return { kind: 'call', op: 'exp', args: [logMass] };
    const m = truncateMassLit(ir.args[0], ir.args[1]);
    if (m == null) return null;
    return m;
  }
  // A probability-measure leaf has total mass 1.
  if (orchestrator.SAMPLEABLE_DISTRIBUTIONS && orchestrator.SAMPLEABLE_DISTRIBUTIONS.has(op)) {
    return { kind: 'lit', value: 1 };
  }
  return null;
}

// Keep scalar factors in log space: exp(±1000) is not representable, but
// normalize(logweighted(±1000, M)) is well-defined for a probability M.
function logTotalMassExpr(ir: any): any {
  if (!ir || ir.kind !== 'call') return null;
  if (ir.op === 'iid') {
    const count = iidCount(ir);
    if (count == null) return null;
    if (count === 0) return { kind: 'lit', value: 0 };
    const mass = logTotalMassExpr(ir.args[0]);
    return mass == null ? null : { kind: 'call', op: 'mul',
      args: [{ kind: 'lit', value: count }, mass] };
  }
  const countMass = positiveCountLogMassExpr(ir);
  if (countMass != null) return countMass;
  if (ir.op === 'locscale' && ir.args?.length === 3) return logTotalMassExpr(ir.args[0]);
  if (ir.op === 'pushfwd' && ir.args?.length === 2) return logTotalMassExpr(ir.args[1]);
  if ((ir.op === 'weighted' || ir.op === 'logweighted') && ir.args?.length === 2) {
    const weight = ir.args[0];
    if (weight?.kind === 'call' && weight.op === 'functionof') return null;
    const inner = logTotalMassExpr(ir.args[1]);
    if (inner == null) return null;
    const factor = ir.op === 'weighted'
      ? { kind: 'call', op: 'log', args: [weight] } : weight;
    return { kind: 'call', op: 'add', args: [factor, inner] };
  }
  // §06 superpose adds masses. Keep the sum in log space as well as each
  // factor: log(exp(theta) + exp(theta)) must remain theta + log(2),
  // including finite theta whose exponential overflows or underflows.
  const components = ir.op === 'superpose' ? ir.args
    : ir.op === 'select' && ir.logweights == null && ir.selectorName == null
      ? ir.branches : null;
  if (Array.isArray(components) && components.length > 0) {
    const terms = components.map(logTotalMassExpr);
    if (terms.some((term: any) => term == null)) return null;
    return { kind: 'call', op: 'logsumexp',
      args: [{ kind: 'call', op: 'vector', args: terms }] };
  }
  const mass = totalMassExpr(ir);
  return mass == null ? null : { kind: 'call', op: 'log', args: [mass] };
}

// Expanded iid nodes carry the resolved product shape. Leave unresolved sizes
// to the existing fallback rather than treating a shape vector as a scalar.
function iidCount(ir: any): number | null {
  if (ir.args?.length !== 2) return null;
  const size = ir.args[1];
  const axes = size?.kind === 'call' && size.op === 'vector' ? size.args : [size];
  if (!Array.isArray(axes) || !axes.every((a: any) =>
    a?.kind === 'lit' && Number.isSafeInteger(a.value) && a.value >= 0)) return null;
  const count = axes.reduce((n: number, a: any) => n * a.value, 1);
  return Number.isSafeInteger(count) ? count : null;
}

// For these nonnegative count laws, [1, inf] retains exactly 1 - P(0).
// Keep live parameters and expm1: small positive masses must not round to zero.
function positiveCountLogMassExpr(ir: any): any | null {
  if (ir.op !== 'truncate' || ir.args?.length !== 2) return null;
  const [base, set] = ir.args;
  if (!['Poisson', 'NegativeBinomial'].includes(base?.op)
      || set?.op !== 'interval' || set.args?.length !== 2) return null;
  const { resolveConstant } = require('./ir-shared.ts');
  const bounds = set.args.map((x: any) => resolveConstant(x, new Map(), new Set()));
  if (bounds[0] !== 1 || bounds[1] !== Infinity) return null;
  const bound = require('./leaf-params.ts').leafParamIRs(base);
  if (!bound || bound.paramIRs.some((x: any) => x == null)) return null;
  const call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });
  const lit = (value: number | boolean) => ({ kind: 'lit', value });
  const [first, second] = bound.paramIRs;
  const logP0 = base.op === 'Poisson' ? call('neg', first)
    : call('neg', call('mul', first, call('log1p', call('divide', lit(1), second))));
  let valid: any = lit(true);
  for (const parameter of bound.paramIRs) {
    const positive = call(base.op === 'Poisson' ? 'ge' : 'gt', parameter, lit(0));
    valid = call('land', valid, call('land', positive, call('isfinite', parameter)));
  }
  return call('ifelse', valid, call('log', call('neg', call('expm1', logP0))), lit(NaN));
}

// M(S) for `truncate(M, S)` as a rewrite-time LITERAL, or null when this arm
// cannot express it (the caller then keeps whatever route it had).
//
// WHY A LITERAL AND NOT AN EXPRESSION IN θ. The IR's scalar evaluator carries
// no CDF primitive, so F(b) − F(a) can only be emitted as a number. That fixes
// the accepted set: the base must be a scalar leaf whose parameters are all
// CONSTANTS, and the set a bounded-or-unbounded `interval` with constant
// bounds. A truncate whose base parameters or bounds move with a latent is
// declined here and keeps its existing route: the sampling route keeps the
// pooled divisor, which is what it already had, and a BARE
// `normalize(truncate(D(θ), S))` is re-resolved per density call by
// mcmc-density's `canDeferTruncateNormalizer` (that gate requires the truncate
// to sit directly under the `normalize`, so it does NOT cover a truncate
// component inside a mixture — that shape keeps the pooled divisor and is the
// narrowed open row in test/normalize-pooled-divisor.test.ts).
//
// The θ-DEPENDENCE this arm restores is the enclosing weight's: the mass of
// `superpose(weighted(p, truncate(B, S)), weighted(1 − p, C))` is
// `p·M(S) + (1 − p)`, which moves with p through a literal M(S).
function truncateMassLit(baseIR: any, setIR: any): any | null {
  // 1-D `interval` only. Gate on the op before calling `parseTruncationBox`:
  // that function THROWS for a record-form cartprod and above its dimension
  // cap, and a throw here would turn a shape that samples today into a
  // refusal. An N-D truncation is a Lebesgue-box mass, not a CDF difference.
  if (!setIR || setIR.kind !== 'call' || setIR.op !== 'interval') return null;
  // Required lazily: mat-density requires this module at load time.
  const { parseTruncationBox } = require('./mat-density.ts');
  const axes = parseTruncationBox(setIR);
  if (!axes || axes.length !== 1) return null;
  const base = constantLeafParams(baseIR);
  if (!base) return null;
  const forwardCdf = require('./forward-cdf.ts');
  // Eleven registered continuous kernels have no CDF row (GeneralizedNormal,
  // VonMises, the §09 HEP densities, Dirac), so this is a real decline and not
  // a formality.
  if (!forwardCdf.hasCdf(base.kernel)) return null;
  const { lo, hi } = axes[0];
  const Z = forwardCdf.intervalProbability(base.kernel, base.input, lo, hi);
  // §06 normalize leaves the result undefined at Z = 0, and the callers all
  // have their own loud refusal for a non-finite −log Z; decline rather than
  // emit a literal that makes the shift ±∞ from inside an algebraic sum, where
  // one zero-mass component would poison a mixture whose other components are
  // fine.
  if (!(Z > 0) || !Number.isFinite(Z)) return null;
  return { kind: 'lit', value: Z };
}

// A continuous scalar distribution leaf whose every parameter is a CONSTANT,
// as `{ kernel, input }` keyed by the SPEC parameter names — the keys
// `forward-cdf`'s CDF table reads. null for anything else.
//
// Parameters bind through `leaf-params.leafParamIRs`, the shared §04 spelling
// rule: reading `kwargs` alone declines the POSITIONAL spelling
// `Normal(0.0, 1.0)` and mis-keys an alias, so two spellings of one measure
// would take different mass routes and disagree — measured: the positional
// mixture kept the θ-constant bake (−5.7411 at θ = 0.9 against the exact
// −7.6256546896974614) while the keyword spelling of the same model was exact.
function constantLeafParams(ir: any): { kernel: string; input: Record<string, number> } | null {
  const { leafParamIRs } = require('./leaf-params.ts');
  const bound = leafParamIRs(ir);
  if (!bound) return null;
  const entry = bound.entry;
  // A DISCRETE base's restricted mass counts the atoms in S, which the
  // continuous CDF difference gets wrong (it drops an atom on a closed lower
  // end). Decline: the materialised mass route counts atoms exactly.
  if (entry.discrete) return null;
  const { resolveConstant } = require('./ir-shared.ts');
  const num = (x: any) => {
    const v = resolveConstant(x, new Map(), new Set());
    return typeof v === 'number' ? v : NaN;
  };
  // `Uniform` declares one parameter, `support`, and takes its bounds from an
  // `interval` inside it (sampler-registry's `customResolveParams`), while the
  // CDF row reads `lo`/`hi`. Map it here so a truncated Uniform is not the one
  // hole in this arm's accepted set.
  if (ir.op === 'Uniform') {
    const sup = bound.paramIRs[0];
    if (!sup || sup.kind !== 'call' || sup.op !== 'interval'
        || !Array.isArray(sup.args) || sup.args.length !== 2) return null;
    const lo = num(sup.args[0]);
    const hi = num(sup.args[1]);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
    return { kernel: 'Uniform', input: { lo, hi } };
  }
  const input: Record<string, number> = {};
  for (let i = 0; i < entry.params.length; i++) {
    const v = num(bound.paramIRs[i]);
    // A parameter that moves with a latent has no constant here, so the whole
    // arm declines and the caller keeps its existing route.
    if (!Number.isFinite(v)) return null;
    input[entry.params[i]] = v;
  }
  return { kernel: bound.kernel, input };
}

// A pooled ensemble mass can only serve as a constant density divisor when
// the unresolved measure has no parameter dependence. Run this after the
// supported algebraic and quadrature builders, which can retain that
// dependence explicitly. A miss here is a support limit, not proof that the
// integral varies: refuse conservatively instead of changing likelihood ratios.
//
// The refusal is an ENGINE LIMITATION, not a model error: §06 defines
// normalize(M) as M shifted by −log totalmass(M) at the SCORED θ, so a
// θ-dependent mass is perfectly valid FlatPPL. Only this engine's mass
// builders fall short of it, so the error carries the `ENGINE_LIMITATION`
// code (see `limitations.ts`) and names the route that hit the gap. Both
// callers reach the same construct by different paths, so `route` is theirs
// to supply.
function assertFixedMassFallback(inner: any, ctx: any, route: string): void {
  const { walkIRScoped } = require('./ir-walk.ts');
  const { isCallableLikeBindingType } = require('./ir-shared.ts');
  const seen = new Set<string>();
  const unresolved = new Set<string>();
  const visit = (ir: any, scope: Set<string>) => {
    walkIRScoped(ir, (n: any, shadowed: Set<string>) => {
      // Ref-headed calls hold their target outside the walker's IR children.
      const ref = n.kind === 'ref' ? n : n.kind === 'call' ? n.target : null;
      if (!ref || ref.ns !== 'self' || shadowed.has(ref.name)) return;
      if (ctx && ctx.fixedValues && ctx.fixedValues.has(ref.name)) return;
      const b = ctx && ctx.bindings && ctx.bindings.get(ref.name);
      if (b && isCallableLikeBindingType(b.type) && b.ir) {
        // A callable's free captures still belong to the enclosing measure.
        // Keep boundary shadowing when following the referenced definition.
        const key = ref.name + '\0' + [...shadowed].sort().join('\0');
        if (!seen.has(key)) {
          seen.add(key);
          visit(b.ir, shadowed);
        }
        return;
      }
      unresolved.add(ref.name);
    }, scope);
  };
  visit(inner, new Set());
  if (unresolved.size > 0) {
    throw engineLimitation(
      'normalize over a parameter-dependent total mass', route,
      'unresolved: ' + [...unresolved].sort().join(', ')
      + '; a pooled divisor would change likelihood ratios, spec §06');
  }
}

// `truncateMassLit` is exported for derivations.ts's `closedFormLogTotalmass`,
// which certifies a dropped structural-projection component's total mass and
// needs the same exact CDF difference. The two callers must not diverge: one
// arm computing M(S) means one accepted set and one refusal boundary.
// Materialise each fallback normalizer's inner measure once. A normalizer
// inside an applied kernel names a module binding that reads the kernel's
// boundary, so it carries its substituted inner IR instead (clm.ts).
function massFromMeasures(nodes: any[], ctx: any): Promise<any[]> {
  const byName = new Map<string, Promise<any>>();
  return Promise.all(nodes.map((node: any) => {
    if (node.massFrom.ir) return require('./materialiser.ts').materialiseMeasureIR(node.massFrom.ir, ctx);
    const name = node.massFrom.ref;
    if (!byName.has(name)) byName.set(name, Promise.resolve(ctx.getMeasure(name)));
    return byName.get(name);
  }));
}

module.exports = { totalMassExpr, logTotalMassExpr, positiveCountLogMassExpr,
  truncateMassLit, assertFixedMassFallback, canDeferTruncateNormalizer, massFromMeasures };
