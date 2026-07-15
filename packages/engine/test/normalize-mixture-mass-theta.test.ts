'use strict';

// =====================================================================
// normalize-mixture-mass-theta.test.ts — Buffy #67
// =====================================================================
//
// A normalize(superpose(...)) mixture whose weights do NOT sum to 1 has a
// θ-DEPENDENT total mass Z(θ) = Σ_i w_i. The density must shift by
// −log Z(θ) at the scored θ (spec §06). The bug (#67): a superpose branch
// with a LITERAL weight lowers to `logweighted(logw, M)` rather than
// `weighted(w, M)`, so a mixture with any literal-weight component presents
// as `select{ weighted(θ, ·), logweighted(0, ·) }`. `totalMassExpr` had a
// `weighted` case but NO `logweighted` case, so it returned null for such a
// mixture — dropping the whole normalize to the θ-CONSTANT materialised-Z
// fallback (Z frozen at a single sampled θ). Result: correct only at the
// materialisation point, silently wrong at every other θ.
//
// This drives the exact IS/matScore density path (logdensityof(L, θ) →
// matLikelihoodDensity → matScore → resolveNormalizeMasses → totalMassExpr)
// at MULTIPLE θ with weights (θ, 1): Z(θ)=θ+1 and the normalized density
// both genuinely vary with θ (no cancellation), so a θ-constant Z mis-scores
// at every θ ≠ the (unknown) materialisation point.
//
// Oracle (Distributions.jl, a=0.5, b=-0.5, y=[0.1,-0.2,0.3,1.0,-1.0]):
//   density(y;θ) = (θ·N(y;a,1) + 1·N(y;b,1)) / (θ+1)
//   θ=0.2 → -6.205814457853139
//   θ=0.5 → -6.091040937398293
//   θ=0.9 → -6.037873947853902

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function scoreOf(src: string): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) {
    throw new Error('scoreOf: __score__ produced no data (measure shape unexpected)');
  }
  return s[0];
}

const YDATA = [0.1, -0.2, 0.3, 1.0, -1.0];
const MODEL = (theta: number) => `
theta ~ Beta(2.0, 2.0)
mixdir = normalize(superpose(weighted(theta, Normal(0.5, 1.0)), weighted(1.0, Normal(-0.5, 1.0))))
y ~ iid(mixdir, 5)
forward_kernel = kernelof(record(y = y), theta = theta)
L = likelihoodof(forward_kernel, record(y = [${YDATA.join(', ')}]))
__score__ = logdensityof(L, record(theta = ${theta.toFixed(6)}))
`;

const ORACLE: Array<[number, number]> = [
  [0.2, -6.205814457853139],
  [0.5, -6.091040937398293],
  [0.9, -6.037873947853902],
];

for (const [theta, oracle] of ORACLE) {
  test(`#67 θ-dependent mixture Z: normalize(superpose(weighted(θ,·),weighted(1,·))) at θ=${theta} matches Distributions.jl`, async () => {
    const got = await scoreOf(MODEL(theta));
    assert.ok(
      Math.abs(got - oracle) <= 1e-9,
      `θ=${theta}: engine ${got} vs oracle ${oracle} (Δ ${Math.abs(got - oracle)}) — `
      + `a θ-constant Z bake (the #67 bug) mis-scores here`
    );
  });
}

test('#67 the three θ scores are DISTINCT and correctly ordered (Z is not baked θ-constant)', async () => {
  const s: number[] = [];
  for (const [theta] of ORACLE) s.push(await scoreOf(MODEL(theta)));
  assert.ok(Math.abs(s[0] - s[1]) > 1e-3 && Math.abs(s[1] - s[2]) > 1e-3,
    `scores must vary with θ; got ${JSON.stringify(s)}`);
});
