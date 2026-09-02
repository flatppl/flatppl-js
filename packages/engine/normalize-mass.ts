'use strict';
// totalMassExpr — IR expression for the total mass Z(θ) of a superpose of
// weighted probability measures (06-measure-algebra §normalize/§superpose).
// Shared by the MCMC scorer (mcmc-density) and the IS density path (mat-density)
// so a normalize(...) whose mass depends on a latent is scored per-θ rather than
// baked as a materialised constant.
const orchestrator = require('./orchestrator.ts');

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
  // Z bake (Buffy #67).
  if (op === 'logweighted' && Array.isArray(ir.args) && ir.args.length === 2) {
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
  // a formality. `matTruncate` needs the same CDF to sample such a component,
  // and throws first in a whole model, which is why the unit test drives this
  // arm directly.
  if (!forwardCdf.hasCdf(base.kernel)) return null;
  const { lo, hi } = axes[0];
  // An infinite bound takes the CDF's limit rather than an evaluation at ±∞,
  // mirroring forward-cdf's own `truncatedQuantile`. One helper for both bounds
  // so the two cannot diverge, and so the infinite arm is exercised by either
  // one — a `-inf` LOWER bound never reaches here today, because it lowers to
  // `neg(const inf)` and `parseTruncationBox`'s bound resolver folds a `lit` or
  // a `const` but not a `neg` of one.
  const cdfAt = (x: number, limit: number) =>
    Number.isFinite(x) ? forwardCdf.cdf(base.kernel, x, base.input) : limit;
  const Z = cdfAt(hi, 1) - cdfAt(lo, 0);
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
// WHY NOT `mat-density.asScalarFactor`. That reads `kwargs` alone, so it
// declines the POSITIONAL spelling `Normal(0.0, 1.0)` and mis-keys an alias.
// Two spellings of one measure would then take different mass routes and
// disagree — measured: the positional mixture kept the θ-constant bake
// (−5.7411 at θ = 0.9 against the exact −7.6256546896974614) while the keyword
// spelling of the same model was exact. Resolution follows
// `sampler-registry.resolveParamsN`'s precedence — kwargs by name, then by
// alias, then positional by declared index — so the accepted set is the
// registry's own.
function constantLeafParams(ir: any): { kernel: string; input: Record<string, number> } | null {
  if (!ir || ir.kind !== 'call' || typeof ir.op !== 'string') return null;
  const samplerLib = require('./sampler.ts');
  if (!samplerLib.isKnownDistribution(ir.op)) return null;
  // `lookupDistribution` THROWS on a surplus keyword, which is §04's own static
  // error and belongs to the call site that recognises the distribution — not
  // to a mass builder whose contract is "an expression or null". Every consumer
  // looks the same IR up itself, so declining here changes only WHERE the error
  // surfaces, never whether it does.
  let entry: any;
  try {
    entry = samplerLib.lookupDistribution(ir);
  } catch {
    return null;
  }
  // A DISCRETE base's restricted mass is Σ pmf over S, which the continuous
  // CDF difference gets wrong (it drops the lower endpoint). §06 defines no
  // discrete-truncate normalizer, and mat-density already refuses
  // `normalize(truncate(<discrete>, S))` loudly; decline so that refusal keeps
  // firing rather than pre-empting it with a silently-wrong literal.
  if (entry.discrete) return null;
  const { resolveConstant } = require('./ir-shared.ts');
  const num = (x: any) => {
    const v = resolveConstant(x, new Map(), new Set());
    return typeof v === 'number' ? v : NaN;
  };
  const kwargs = ir.kwargs || {};
  const positional = Array.isArray(ir.args) ? ir.args : [];
  const paramIR = (name: string, i: number) => {
    if (name in kwargs) return kwargs[name];
    /* c8 ignore start -- no REGISTRY entry currently declares an alias, so this
       arm is unreachable today; it is kept because `resolveParamsN` consults
       aliases and the two resolvers must not drift apart */
    const alias = entry.aliases && entry.aliases[name];
    if (alias && alias in kwargs) return kwargs[alias];
    /* c8 ignore stop */
    return i < positional.length ? positional[i] : null;
  };
  // `Uniform` declares one parameter, `support`, and takes its bounds from an
  // `interval` inside it (sampler-registry's `customResolveParams`), while the
  // CDF row reads `lo`/`hi`. Map it here so a truncated Uniform is not the one
  // hole in this arm's accepted set.
  if (ir.op === 'Uniform') {
    const sup = paramIR('support', 0);
    if (!sup || sup.kind !== 'call' || sup.op !== 'interval'
        || !Array.isArray(sup.args) || sup.args.length !== 2) return null;
    const lo = num(sup.args[0]);
    const hi = num(sup.args[1]);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
    return { kernel: 'Uniform', input: { lo, hi } };
  }
  const input: Record<string, number> = {};
  for (let i = 0; i < entry.params.length; i++) {
    const name = entry.params[i];
    const v = num(paramIR(name, i));
    // A parameter that moves with a latent has no constant here, so the whole
    // arm declines and the caller keeps its existing route.
    if (!Number.isFinite(v)) return null;
    input[name] = v;
  }
  return { kernel: ir.op, input };
}

module.exports = { totalMassExpr };
