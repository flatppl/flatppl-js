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
  if (op === 'weighted' && Array.isArray(ir.args) && ir.args.length === 2) {
    const inner = totalMassExpr(ir.args[1]);
    if (inner == null) return null;
    // mass(weighted(w, M)) = w · mass(M)
    return { kind: 'call', op: 'mul', args: [ir.args[0], inner] };
  }
  // A probability-measure leaf has total mass 1.
  if (orchestrator.SAMPLEABLE_DISTRIBUTIONS && orchestrator.SAMPLEABLE_DISTRIBUTIONS.has(op)) {
    return { kind: 'lit', value: 1 };
  }
  return null;
}

module.exports = { totalMassExpr };
