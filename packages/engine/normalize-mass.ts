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
// (total mass 1). Then Z(θ) = Σ_i w_i·1 = Σ_i w_i. This is the ONLY case where
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
  // A probability-measure leaf has total mass 1.
  if (orchestrator.SAMPLEABLE_DISTRIBUTIONS && orchestrator.SAMPLEABLE_DISTRIBUTIONS.has(op)) {
    return { kind: 'lit', value: 1 };
  }
  return null;
}

module.exports = { totalMassExpr };
