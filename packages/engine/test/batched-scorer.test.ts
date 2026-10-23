'use strict';
// batched-scorer.test.ts — mv.logPosteriorBatch must equal the scalar
// mv.logPosterior point-by-point. The batched path scores the likelihood in one
// logDensityN(count=N) pass (per-atom latents via refArrays); this gate proves
// that batching changes performance only, never the numbers. Covers every
// bayesupdate baseline model (scalar + vector latents, derived bindings).

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const fs       = require('node:fs');
const path     = require('node:path');

const { ctxFor }    = require('./density/regression-baseline.test.ts');
const MV            = require('../model-view.ts');
const rng           = require('../rng.ts');
const sampler       = require('../sampler.ts');

const FIX = path.join(__dirname, 'fixtures/baseline');
const MODELS = ['linear-regression', 'eight-schools', 'partial-pooling', 'gamma-reparam',
                'hierarchical-logistic', 'rasch-1pl', 'poisson-glm-link'];

for (const model of MODELS) {
  test(`batched logπ == scalar logπ: ${model}`, async () => {
    const src = fs.readFileSync(path.join(FIX, model + '.flatppl'), 'utf8');
    const ctx = ctxFor(src, 2000).ctx;
    const dv = ctx.lookupDerivation ? ctx.lookupDerivation('posterior') : ctx.derivations.posterior;
    const mv = await MV.buildModelViewFromCtx(ctx, dv);
    assert.equal(typeof mv.logPosteriorBatch, 'function', 'mv exposes logPosteriorBatch');

    const k = rng.keyFromSeed(3);
    const prng = sampler.makePhiloxPrngAdapter(rng.stateFromKey(k[0], k[1]));
    // In-region points from the prior pool + jittered variants → finite logπ.
    const pts: Float64Array[] = mv.initFromPrior(24, prng);
    for (let i = 0; i < pts.length; i += 3) {
      const j = Float64Array.from(pts[i]);
      for (let d = 0; d < j.length; d++) j[d] += 0.3 * (prng() - 0.5);
      pts.push(j);
    }

    const scalar = pts.map((y: Float64Array) => mv.logPosterior(y));
    const batch  = mv.logPosteriorBatch(pts);
    assert.equal(batch.length, pts.length, 'batch length matches');

    // Contract: batched == scalar, point for point (including which points are
    // non-finite). Batching changes performance only, never the numbers.
    let maxDiff = 0, compared = 0;
    for (let i = 0; i < pts.length; i++) {
      const s = scalar[i], b = batch[i];
      if (!Number.isFinite(s)) { assert.ok(!Number.isFinite(b), `point ${i}: scalar=${s} batch=${b} (both should be non-finite)`); continue; }
      assert.ok(Number.isFinite(b), `point ${i}: scalar finite ${s} but batch ${b}`);
      maxDiff = Math.max(maxDiff, Math.abs(s - b)); compared++;
    }
    assert.ok(compared > 0, 'at least one finite point compared');
    assert.ok(maxDiff < 1e-9, `max |scalar - batch| = ${maxDiff} (want < 1e-9 over ${compared} points)`);
  });
}

// surgical-failures' matrix latent `p` (a [2,16] block of Betas from a kernel-
// broadcast over a user function) IS now enumerated + scored — but its prior
// `a_plus_b ~ iid(pushfwd(fn(0.1*exp(_)), Exp), G)` has no tractable density
// (spec §06: an unannotated pushforward is a static error). MCMC/AMIS must
// score the prior, so the ModelView refuses with an actionable message rather
// than running stuck chains on a silent −∞. (IS forward-samples the prior, so
// it is unaffected — its baseline golden still passes.)
test('MCMC ModelView refuses an intractable (genuinely non-invertible pushfwd) prior', async () => {
  // Spec §06 case 3: a pushforward through a function that is neither a
  // registry bijection nor a structural projection is a static error unless
  // wrapped in bijection(...). `_ * _` (x²) is non-injective on ℝ —
  // invertExpr refuses — so the prior is intractable for the score-the-prior
  // backends. (The litter Beta-Binomial `pushfwd(fn(0.1*exp(_)), …)` that this
  // test formerly used is NOT intractable — it's case 1 (exp ∘ positive
  // affine), now auto-derived; see pushfwd-elementary-density.test.ts.)
  // sq = z -> z*z : a single-arg, non-injective map (the free var appears in
  // both factors, so invertExpr refuses). Samples fine (z² ≥ 0) so the latent
  // pools and the tractability probe actually runs.
  const src = `
sq = z -> z * z
weird = pushfwd(sq, Exponential(1.5))
a_plus_b ~ iid(weird, 2)
k = (z) -> Normal(z, 1)
y ~ k.(a_plus_b)
prior = lawof(record(a_plus_b = a_plus_b))
forward_kernel = kernelof(record(y = y), a_plus_b = a_plus_b)
L = likelihoodof(forward_kernel, record(y = [1.0, 2.0]))
posterior = bayesupdate(L, prior)
`;
  const ctx = ctxFor(src, 2000).ctx;
  const dv = ctx.lookupDerivation ? ctx.lookupDerivation('posterior') : ctx.derivations.posterior;
  await assert.rejects(
    () => MV.buildModelViewFromCtx(ctx, dv),
    /tractable density|bijection/,
    'should reject with the pushforward/bijection guidance',
  );
});
