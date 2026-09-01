'use strict';
// leaf-mass-quad — a θ-DEPENDENT normalizer for `normalize(weighted(w, B))`
// where `B` is a continuous scalar probability LEAF, emitted as an IR
// EXPRESSION in θ.
//
// WHY THIS EXISTS. Spec §06 "Density of composed measures" fixes the shift:
// "`normalize` (from M / Z): logdensityof(normalize(M), x) =
// logdensityof(M, x) − log Z, with Z = totalmass(M) finite and nonzero", and
// §06 `normalize` makes every θ-slice a probability measure. When `w` references
// a latent, Z is a function of θ, so a CONSTANT −log Z is wrong at every θ but
// the one it was computed at.
//
// Three routes already cover parts of that surface: `normalize-mass`'s
// `totalMassExpr` (an algebraic mass), `crn-normalize` (a Lebesgue-box base) and
// mat-density's `weightedLeafQuadLogZ` (a leaf base with a θ-INDEPENDENT
// weight). A θ-DEPENDENT weight over a leaf base fell through all three: the
// density route baked −log Ẑ from the inner measure's tracked `logTotalmass` (an
// importance-sampling estimate over the base's own atom ensemble, so it moved
// with the sample count) while the sampling route REFUSED the same measure. The
// two routes of one measure disagreed. Measured on `theta ~ Uniform(0.5, 2)`,
// `w = exp(theta * x)` over `Normal(0, 1)`: `likelihoodof` at θ = 1 gave −1.885
// at N = 1, −1.019 at N = 64 and −1.654 at N = 4000, against the exact
// −1.0439385332046727.
//
// THE ESTIMATOR. Substitute the base's own inverse CDF, exactly as the
// θ-independent arm does:
//
//     Z(θ) = ∫ w(x; θ) dB(x) = ∫₀¹ w(F_B⁻¹(u); θ) du
//
// and apply a FIXED composite Gauss-Legendre rule on (0,1) whose cells are
// graded geometrically towards both endpoints (`quadrature.gradedUnitCells`):
//
//     Ẑ(θ) = Σ_m c_m · w(x_m; θ),     x_m = F_B⁻¹(u_m)
//
// The nodes `x_m` and the coefficients `c_m` are computed at REWRITE time and
// baked into the IR as literals, so the emitted expression is arithmetic in θ
// alone. `F_B⁻¹` needs the base's parameters, so this arm requires them to be
// constants; a leaf whose parameters move with a latent is declined (see below).
//
// WHY THE INVERSE CDF. It cancels the base density exactly, leaving `w` alone as
// the integrand — flat for a constant weight over ANY base, including a
// heavy-tailed one. The alternative change-of-variables (an `atanh` stretch over
// an unbounded axis) outruns a heavy-tailed base's density, so a perfectly
// ordinary measure gets an unbounded integrand. Same reasoning as
// `weightedLeafQuadLogZ`, and the same accepted set: `inverse-cdf.hasQuantile`.
//
// WHY A FIXED RULE AND NOT ADAPTIVE. θ is unknown at rewrite time, so there is
// no integrand to refine against. A fixed rule also gives the property the
// route asymmetry above is about: the IS route (mat-density), the MH route
// (mcmc-density) and the sampling route (materialiser) all build the SAME
// expression for the same model, so they cannot disagree. §06
// "Reproducibility" is satisfied by construction — the value depends on the
// query alone.
//
// WHY NOT crn-normalize's MONTE-CARLO ESTIMATOR. crn-normalize reweights a
// fixed random sample because a k-D box admits no cheap graded rule. In one
// dimension the graded rule is available and is orders more accurate, which
// matters twice over. Measured on `w = exp(θx)` over `Normal(0, 1)`, relative
// error in Z over θ ∈ [0.5, 2]: this rule 8e-10 … 2e-8, a 128-point stratified
// sample 1e-2 … 1.7e-1. The second reason is internal agreement: a model with θ
// FIXED takes `weightedLeafQuadLogZ`'s adaptive quadrature at 1e-10, so a
// Monte-Carlo arm here would make two spellings of one measure disagree by
// ~0.1 nats. It also all but removes the Jensen bias crn-normalize documents:
// −log Ẑ is biased for −log Z by O(Var Ẑ), which at 1e-8 relative is nothing.
//
// THE ACCURACY GATE, AND WHY IT IS IN THE EXPRESSION. A weight can outgrow the
// base's tail — `exp(θ x²)` over `Normal(0, 1)` has no finite Z for θ ≥ 1/2 —
// and whether it does depends on θ, so no rewrite-time test can settle it. The
// emitted expression therefore carries its own probe: the OUTERMOST graded
// cells' contribution, which for an integrand this rule resolves is a negligible
// fraction of Ẑ and for one it does not is a large one. Over a measured spread
// of weights and bases the ratio tracks the relative error closely — every case
// with relative error ≤ 1e-4 sat at or below 1.1e-3, every case at or above
// 3.5e-3 error sat at or above 1.7e-2, and every divergent Z sat at or above
// 1.4e-1 — so the gate is set at 1e-2 and hands back +∞ when it trips. The
// callers mark the rewritten node `fromNormalize`, and density.ts's
// `addNormalizeShift` turns a non-finite −log Ẑ into a loud refusal naming §06's
// "If Z = 0 or Z = ∞, the result is undefined"; the sampling route has its own
// per-atom check on the same value.
//
// WHAT THIS ARM DOES NOT DO. A weight that goes negative or non-finite at a node
// is not clamped (the θ-independent arm clamps such a point to 0). Clamping in
// the IR needs the weight body twice per node, and §06's normalizer integrates a
// non-negative weight, so a Ẑ that comes out non-positive or NaN is left to the
// callers' loud refusals instead.

const { gradedUnitCells } = require('./quadrature.ts');
const {
  bindWeightAt, nodeCount, balancedAdd, crnWeightIsThetaDependent, CRN_NODE_BUDGET,
} = require('./crn-normalize.ts');

// Dyadic grading levels and the resulting node count (2·LEVELS cells × 5
// points). LEVELS = 40 puts the outermost cell at 2^-40 ≈ 9e-13, deep enough to
// resolve `exp(4x)` over a standard normal to 1e-4; deeper grading runs the
// inverse CDF into floating-point saturation instead of buying accuracy.
const LEAF_GRADE_LEVELS = 40;

// The outermost cell's share of Ẑ above which the rule is declared unresolved.
// Measured: see "THE ACCURACY GATE" above.
const LEAF_TAIL_GATE = 1e-2;

const _lit = (v: number) => ({ kind: 'lit', value: v });
const _call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });

type LeafShape = {
  weightParams: string[];
  weightBody: any;
  logSpace: boolean;
  kernel: string;
  input: Record<string, number>;
};

// Recognise `normalize(weighted(<functionof>, <continuous scalar leaf>))` and
// its log-space spelling `logweighted`. null for every other shape, which
// leaves the caller's existing route in place.
function leafRecognize(node: any, ctx: any): LeafShape | null {
  if (!node || node.kind !== 'call' || node.op !== 'normalize'
      || !Array.isArray(node.args) || node.args.length !== 1) return null;
  const inner = node.args[0];
  if (!inner || inner.kind !== 'call'
      || (inner.op !== 'weighted' && inner.op !== 'logweighted')
      || !Array.isArray(inner.args) || inner.args.length !== 2) return null;
  const fn = inner.args[0];
  // §06 weight arity: a scalar variate takes a ONE-parameter weight. A constant
  // weight (no `functionof`) is a scalar mass factor and belongs to
  // normalize-mass's algebraic arm, which is exact.
  if (!fn || fn.kind !== 'call' || fn.op !== 'functionof'
      || !Array.isArray(fn.params) || fn.params.length !== 1 || !fn.body) return null;
  // The base must be a closed scalar leaf whose parameters are all numbers:
  // `F_B⁻¹` is evaluated at rewrite time, so a parameter that moves with a
  // latent has no value here. `asScalarFactor` with no point resolves literals
  // only, so it declines exactly those. Required lazily — mat-density requires
  // this module.
  const { asScalarFactor } = require('./mat-density.ts');
  const base = asScalarFactor(inner.args[1], null);
  if (!base) return null;
  const invcdf = require('./inverse-cdf.ts');
  if (!invcdf.hasQuantile(base.kernel)) return null;
  const shape: LeafShape = {
    weightParams: fn.params.slice(), weightBody: fn.body,
    logSpace: inner.op === 'logweighted',
    kernel: base.kernel, input: base.input,
  };
  // Only a θ-DEPENDENT weight may be taken away from `weightedLeafQuadLogZ`'s
  // adaptive quadrature, which is the more accurate answer where it applies.
  // The test is crn-normalize's, so the two per-θ builders cannot drift on what
  // "θ-dependent" means — in particular on a REIFIED weight, whose boundary
  // binding names live in `self` and look latent until the parameters come out.
  if (!crnWeightIsThetaDependent(
    { weightParams: shape.weightParams, weightBody: shape.weightBody, axes: [] }, ctx)) {
    return null;
  }
  return shape;
}

// The baked quadrature: one (coefficient, base point) pair per node, grouped so
// the caller can address the outermost cell at each end.
function leafQuadCells(kernel: string, input: Record<string, number>) {
  const invcdf = require('./inverse-cdf.ts');
  return gradedUnitCells(LEAF_GRADE_LEVELS).map((cell: any) => ({
    cs: cell.ws as number[],
    xs: (cell.us as number[]).map((u: number) => invcdf.quantile(kernel, u, input)),
  }));
}

// Σ c_m · w(x_m; θ) over one cell, as a balanced `add` tree.
function cellSum(shape: LeafShape, cell: { cs: number[]; xs: number[] }): any {
  const terms = cell.xs.map((x, i) => {
    const bound = bindWeightAt(shape.weightParams, shape.weightBody, [x], false);
    const w = shape.logSpace ? _call('exp', bound) : bound;
    return _call('mul', _lit(cell.cs[i]), w);
  });
  return balancedAdd(terms);
}

// Build the IR expression for Ẑ(θ) of a `normalize` node, or null when the node
// is not the recognised shape (the caller keeps its existing route).
//
// THROWS when the shape IS recognised but the expression will not fit the node
// budget. Every caller's fallback there is the mass pooled over the atom
// ensemble — a baked −log Z on the two density routes, the pooled weight sum on
// the sampling route — which is E[Z] rather than Z(θ), so returning null would
// hand back the one number this arm exists to remove.
function leafMassExpr(node: any, ctx: any): any | null {
  const shape = leafRecognize(node, ctx);
  if (!shape) return null;
  const cells = leafQuadCells(shape.kernel, shape.input);
  // The value sum, the gate's own copy of it, and the two outermost cells: the
  // IR is a tree, so a subexpression used twice is built twice.
  const copies = 2 * cells.length * 5 + 2 * 5;
  const bodySize = nodeCount(shape.weightBody);
  if (bodySize * copies > CRN_NODE_BUDGET) {
    throw new Error('normalize(weighted(w, ' + shape.kernel + ')): the weight depends on a '
      + 'latent, so its mass Z(θ) = ∫ w dB is a fixed quadrature with the weight body '
      + 'inlined at each node — ' + bodySize + ' nodes × ' + copies + ' = '
      + (bodySize * copies) + ' nodes, over the budget of ' + CRN_NODE_BUDGET + '. Falling '
      + 'back to the pooled mass would score every θ against E[Z]; spec §06 normalize '
      + 'makes each θ-slice a probability measure, so refusing instead. Rewrite the '
      + 'measure so its mass is closed-form in the latent.');
  }
  const total = () => balancedAdd(cells.map((c: any) => cellSum(shape, c)));
  const outer = _call('add',
    _call('abs', cellSum(shape, cells[0])),
    _call('abs', cellSum(shape, cells[cells.length - 1])));
  const gate = _call('le', outer, _call('mul', _lit(LEAF_TAIL_GATE), _call('abs', total())));
  return _call('ifelse', gate, total(), _lit(Infinity));
}

module.exports = {
  leafMassExpr,
  leafRecognize,
  leafQuadCells,
  LEAF_GRADE_LEVELS,
  LEAF_TAIL_GATE,
};
