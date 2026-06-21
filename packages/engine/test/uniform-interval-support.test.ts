'use strict';
// Regression: a Uniform(interval(lo, hi)) latent must keep its DECLARED bounds
// in the MCMC constraining transform. The bug: transforms.supportOf could not
// read the bounds out of the `interval` support set, so every bounded-Uniform
// latent defaulted to [0, 1] — the constrain map a + w·sigmoid(y) became
// 0 + 1·sigmoid(y), capping scale parameters (and any bounded latent) at 1 with
// a hard edge at 0. Spec 08-distributions "Uniform": `Uniform(support)` is the
// uniform distribution ON `support`, so the support is the given set, not [0,1].

const { test }   = require('node:test');
const assert     = require('node:assert/strict');
const { ctxFor } = require('./density/regression-baseline.test.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');
const modelSpec  = require('../model-spec.ts');

// Scale parameter with a wide Uniform prior whose bounds are NOT [0,1]; data
// spread ≈ 5 forces the posterior scale well above 1.
const SRC = `
y_data = [5.0, -5.0, 6.0, -6.0, 4.0, -4.0, 5.5, -5.5]
mu ~ Normal(0.0, 10.0)
sigma ~ Uniform(interval(0.5, 10.0))
prior = lawof(record(mu = mu, sigma = sigma))
y ~ iid(Normal(mu, sigma), 8)
forward_kernel = kernelof(record(y = y), mu = mu, sigma = sigma)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`;

function posteriorDeriv(ctx: any): any {
  for (const n of Object.keys(ctx.derivations)) {
    const d = ctx.derivations[n];
    if (d && d.kind === 'bayesupdate') return d;
  }
  return null;
}

test('Uniform(interval) latent keeps its declared bounds, not [0,1]', async () => {
  const { ctx } = ctxFor(SRC, 100);
  const d = posteriorDeriv(ctx);
  assert.ok(d, 'posterior bayesupdate derivation present');

  const latents = modelSpec.enumerateLatents(d, ctx);
  const sig = latents.find((l: any) => l.name === 'sigma');
  assert.ok(sig, 'sigma latent enumerated');
  assert.deepEqual(sig.support, { kind: 'interval', a: 0.5, b: 10 });

  const mv = await buildModelViewFromCtx(ctx, d);
  const i = mv.names.indexOf('sigma');
  assert.ok(i >= 0, 'sigma is a ModelView coordinate');
  // constrain(0) = a + w/(1 + e^0) = 0.5 + 9.5/2 = 5.25 — well above 1, proving
  // the transform spans [0.5, 10] rather than the old [0,1] default.
  const y = new Float64Array(mv.dim);
  y[i] = 0;
  const sigmaAtZero = mv.constrainAll(y).sigma;
  assert.ok(sigmaAtZero > 1, `sigma can exceed 1 (got ${sigmaAtZero})`);
  assert.ok(Math.abs(sigmaAtZero - 5.25) < 1e-9, `midpoint 5.25 (got ${sigmaAtZero})`);
});

test('Uniform(interval) posterior explores above 1 (data scale ≈ 5)', async () => {
  const { ctx } = ctxFor(SRC, 4000);
  const m = await ctx.getMeasure('posterior');
  const fld = m.fields && m.fields.sigma;
  assert.ok(fld && fld.samples && fld.samples.length > 0, 'sigma samples present');
  let mx = -Infinity;
  for (const v of fld.samples) if (v > mx) mx = v;
  assert.ok(mx > 1.0, `posterior sigma should reach above 1, got max ${mx}`);
});
