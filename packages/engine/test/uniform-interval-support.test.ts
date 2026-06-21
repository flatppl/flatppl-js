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

test('Uniform(interval) posterior explores the full [0.5,10] support, not [0,1]', async () => {
  const { ctx } = ctxFor(SRC, 4000);
  const m = await ctx.getMeasure('posterior');
  const fld = m.fields && m.fields.sigma;
  assert.ok(fld && fld.samples && fld.samples.length > 0, 'sigma samples present');
  const s = fld.samples;

  let mx = -Infinity;
  let above1 = 0;
  for (const v of s) { if (v > mx) mx = v; if (v > 1) above1++; }

  // Importance-weighted posterior mean (the measure carries logWeights).
  const lw = m.logWeights || new Array(s.length).fill(0);
  let lwMax = -Infinity;
  for (const v of lw) if (v > lwMax) lwMax = v;
  let z = 0;
  for (const v of lw) z += Math.exp(v - lwMax);
  let mean = 0;
  for (let i = 0; i < s.length; i++) mean += (Math.exp(lw[i] - lwMax) / z) * s[i];

  // The old [0,1] default made all three of these impossible: every sample was
  // < 1, so mean < 1, the fraction above 1 was 0, and the max could not pass 1.
  // Data scale ≈ 5 ⇒ posterior concentrates well above 1 and reaches toward the
  // true upper bound 10 (thresholds are loose vs the seed values mean≈6.2,
  // frac>1≈0.95, max≈10 — they pin the regression, not exact draws).
  assert.ok(mean > 3.0, `posterior sigma mean should be well above 1, got ${mean}`);
  assert.ok(above1 / s.length > 0.7, `most samples should exceed 1, got ${above1 / s.length}`);
  assert.ok(mx > 8.0, `posterior sigma should reach toward the upper bound 10, got max ${mx}`);
});

// --- Unit coverage for the bounds extractor (model-spec.uniformSupportFromDistIR).
// Pins finding #3: a Uniform whose support is NOT a finite interval must return
// null (caller refuses / falls back) — never silently a [0,1] interval.
const lit = (v: number) => ({ kind: 'lit', value: v, numType: 'real' });
const interval = (lo: number, hi: number) => ({ kind: 'call', op: 'interval', args: [lit(lo), lit(hi)] });
const uni = (supIR: any) => ({ op: 'Uniform', args: [supIR] });
const supOf = (distIR: any) => modelSpec.uniformSupportFromDistIR(distIR, new Map(), new Map());

test('uniformSupportFromDistIR: finite interval → its bounds', () => {
  assert.deepEqual(supOf(uni(interval(0.1, 20))), { kind: 'interval', a: 0.1, b: 20 });
  // kwargs.support spelling is equivalent to the positional arg.
  assert.deepEqual(
    supOf({ op: 'Uniform', kwargs: { support: interval(0.1, 20) }, args: [] }),
    { kind: 'interval', a: 0.1, b: 20 });
});

test('uniformSupportFromDistIR: unitinterval const → [0,1]', () => {
  assert.deepEqual(supOf(uni({ kind: 'const', name: 'unitinterval' })), { kind: 'interval', a: 0, b: 1 });
});

test('uniformSupportFromDistIR: improper / degenerate / missing → null (no silent [0,1])', () => {
  assert.equal(supOf(uni({ kind: 'const', name: 'posreals' })), null); // half-line: infinite
  assert.equal(supOf(uni({ kind: 'const', name: 'reals' })), null);     // whole line: infinite
  assert.equal(supOf(uni(interval(20, 0.1))), null);                    // reversed (hi < lo)
  assert.equal(supOf(uni(interval(5, 5))), null);                       // degenerate (zero width)
  assert.equal(supOf({ op: 'Uniform', args: [] }), null);               // missing support
});
