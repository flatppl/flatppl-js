'use strict';
// A `Dirichlet` latent is constrained to `stdsimplex(K)`, so the MCMC/AMIS/SMC/
// nested backends must explore a K−1 dimensional unconstrained space through the
// stick-breaking bijection — not K independent real coordinates.
//
// Spec §08 "Dirichlet": "Domain/Support: cartpow(reals, n)/stdsimplex(n)", with
// the density stated only "for x ∈ {p ∈ ℝⁿ : Σᵢpᵢ = 1, pᵢ ≥ 0}" and the reference
// measure the coordinate measure dx₁⋯dx_{n−1} of Lebesgue(stdsimplex(n)). Off the
// simplex the density is undefined; the engine's unconstrained continuation
// Σᵢ(αᵢ−1)log pᵢ is unbounded above for αᵢ > 1, so a sampler let loose on ℝᴷ
// diverges instead of sampling the posterior.
//
// ORACLE — conjugacy, closed form, no sampler involved. Prior
// Dirichlet(α = (2,3,4)) with an iid Categorical(p) likelihood over class counts
// (n₁,n₂,n₃) = (14,8,8) gives posterior Dirichlet(16,11,12):
//   E[pᵢ]   = αᵢ / α₀,                       α₀ = 39
//   Var[pᵢ] = αᵢ(α₀−αᵢ) / (α₀²(α₀+1))
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');
const T = require('../transforms.ts');
const modelSpec = require('../model-spec.ts');
const MV = require('../model-view.ts');

// The coverage corpus's allele_freq model (corpora/coverage/allele_freq), whose
// frozen log-density vector is pinned in the testsuite at 1e-9.
const MODEL = `
counts_data = [1, 3, 2, 1, 1, 3, 2, 2, 1, 3, 3, 1, 2, 1, 3, 1, 1, 2, 3, 1, 2, 1, 1, 3, 2, 1, 3, 1, 2, 1]
p ~ Dirichlet(alpha = [2.0, 3.0, 4.0])
y ~ iid(Categorical(p = p), lengthof(counts_data))
prior = lawof(record(p = p))
forward_kernel = kernelof(record(y = y), p = p)
L = likelihoodof(forward_kernel, record(y = counts_data))
posterior = bayesupdate(L, prior)
`;

const A = [16, 11, 12];
const A0 = 39;
const CF_MEAN = A.map((a) => a / A0);
const CF_VAR  = A.map((a) => (a * (A0 - a)) / (A0 * A0 * (A0 + 1)));
const CF_SD   = CF_VAR.map(Math.sqrt);

// Per-coordinate draws from the returned measure. `p` is one vector field of
// per-atom width 3, stored row-major as [N, 3].
function coordsOf(m: any): Float64Array[] {
  const f = m.fields.p;
  const data: Float64Array = f.samples || (f.value && f.value.data);
  const D = (f.value && Array.isArray(f.value.shape)) ? f.value.shape[1] : 3;
  const N = data.length / D;
  const out: Float64Array[] = [];
  for (let j = 0; j < D; j++) {
    const c = new Float64Array(N);
    for (let a = 0; a < N; a++) c[a] = data[a * D + j];
    out.push(c);
  }
  return out;
}

function meanSd(a: Float64Array) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  const mu = s / a.length;
  let v = 0;
  for (let i = 0; i < a.length; i++) v += (a[i] - mu) ** 2;
  return { mean: mu, sd: Math.sqrt(v / (a.length - 1)) };
}

// Viewer defaults (viewer/src/main.ts ctx.inferenceOpts).
const VIEWER: any = {
  chains: 4, walkers: null, warmup: 1000, draws: 4000, seed: 1,
  amisIters: 30, amisSamples: 300,
  smcParticles: 2000, smcSteps: 12, smcCESS: 0.7,
  nLive: 400, dlogz: 0.5, regionMetric: 'off',
};

test('a Dirichlet latent reports simplex support and a K−1 unconstrained dimension', async () => {
  const { ctx } = ctxFor(MODEL, 200);
  const d = ctx.derivations['posterior'];
  const latents = modelSpec.enumerateLatents(d, ctx);
  assert.equal(latents.length, 1);
  assert.equal(latents[0].support.kind, 'simplex',
    `Dirichlet latent support should be simplex, got ${JSON.stringify(latents[0].support)}`);
  const mv = await MV.buildModelViewFromCtx(ctx, d);
  assert.equal(mv.dim, 2, 'stdsimplex(3) is a 2-dimensional manifold');
  assert.deepEqual(mv.names, ['p[0]', 'p[1]', 'p[2]'],
    'the CONSTRAINED output coordinates stay K wide');
});

test('the simplex transform round-trips and its Jacobian recovers the closed-form '
  + 'Dirichlet(16,11,12) moments (grid over the unconstrained space)', async () => {
  const st = T.simplexTransform(3);
  const p0 = Float64Array.from([0.41, 0.28, 0.31]);
  const back = st.constrain(st.unconstrain(p0));
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(back[i] - p0[i]) < 1e-12, `round-trip coord ${i}: ${back[i]} vs ${p0[i]}`);
  }
  // A boundary coordinate must unconstrain finite, not to −∞.
  const yEdge = st.unconstrain(Float64Array.from([0, 0.5, 0.5]));
  assert.ok(yEdge.every((v: number) => Number.isFinite(v)), `boundary draw gave ${Array.from(yEdge)}`);

  // ∫ exp(logPosterior(y)) dy over ℝ² reproduces the conjugate posterior.
  const { ctx } = ctxFor(MODEL, 200);
  const mv = await MV.buildModelViewFromCtx(ctx, ctx.derivations['posterior']);
  const M = 260, LO = -8, h = 16 / M;
  let Z = 0;
  const m1 = [0, 0, 0], m2 = [0, 0, 0];
  const y = new Float64Array(2);
  for (let i = 0; i < M; i++) {
    y[0] = LO + (i + 0.5) * h;
    for (let j = 0; j < M; j++) {
      y[1] = LO + (j + 0.5) * h;
      const lp = mv.logPosterior(y);
      if (!Number.isFinite(lp)) continue;
      const w = Math.exp(lp) * h * h;
      const flat = mv.constrainAll(y);
      Z += w;
      for (let k = 0; k < 3; k++) {
        const v = flat[`p[${k}]`];
        m1[k] += w * v; m2[k] += w * v * v;
      }
    }
  }
  assert.ok(Z > 0, 'grid integral collapsed to zero mass');
  for (let k = 0; k < 3; k++) {
    const mu = m1[k] / Z, va = m2[k] / Z - mu * mu;
    // Midpoint-rule error on a 260² grid, not sampling error.
    assert.ok(Math.abs(mu - CF_MEAN[k]) < 1e-6, `E[p[${k}]] ${mu} vs ${CF_MEAN[k]}`);
    assert.ok(Math.abs(va / CF_VAR[k] - 1) < 1e-3, `Var[p[${k}]] ${va} vs ${CF_VAR[k]}`);
  }
});

// Margins are absolute on the mean and relative on the sd. The tightest
// per-sampler numbers observed at these settings sit an order of magnitude
// inside them; the slack covers the seed-to-seed spread of the low-ESS backends
// (mh at ~200 ESS gives a mean standard error of ~0.006).
const MEAN_TOL = 0.02;
const SD_REL_TOL = 0.15;

for (const backend of ['mh', 'ram', 'slice', 'emcee', 'demcz', 'amis', 'smc', 'nested',
  'elliptical-slice-sampler']) {
  test(`backend:${backend} recovers the conjugate Dirichlet(16,11,12) posterior`, async () => {
    const { ctx } = ctxFor(MODEL, 4000);
    const opts = Object.assign({}, VIEWER, { backend });
    const m = await materialiser.materialiseMeasure('posterior', ctx, opts);
    const cs = coordsOf(m);
    assert.equal(cs.length, 3, `expected 3 coordinates, got ${cs.length}`);
    let sumMean = 0;
    for (let k = 0; k < 3; k++) {
      const { mean, sd } = meanSd(cs[k]);
      assert.ok(Number.isFinite(mean) && Number.isFinite(sd),
        `p[${k}] non-finite: mean ${mean} sd ${sd}`);
      // Every draw stays on the simplex.
      let lo = Infinity, hi = -Infinity;
      for (const v of cs[k]) { if (v < lo) lo = v; if (v > hi) hi = v; }
      assert.ok(lo >= 0 && hi <= 1, `p[${k}] left [0,1]: [${lo}, ${hi}]`);
      assert.ok(Math.abs(mean - CF_MEAN[k]) < MEAN_TOL,
        `E[p[${k}]] ${mean.toFixed(5)} vs closed form ${CF_MEAN[k].toFixed(5)}`);
      assert.ok(Math.abs(sd / CF_SD[k] - 1) < SD_REL_TOL,
        `sd[p[${k}]] ${sd.toFixed(5)} vs closed form ${CF_SD[k].toFixed(5)}`);
      sumMean += mean;
    }
    assert.ok(Math.abs(sumMean - 1) < 1e-9, `Σ E[pᵢ] = ${sumMean}, want 1`);
  });
}

test('unconstrainAll reads a simplex latent given as a plain array or a shaped Value', async () => {
  // The scorer hands a vector latent a Float64Array, but `constrainAll`'s
  // record and a matrix latent's `{shape, data}` Value both reach the same
  // reader, so all three spellings must round-trip to the same coordinates.
  const { ctx } = ctxFor(MODEL, 200);
  const mv = await MV.buildModelViewFromCtx(ctx, ctx.derivations['posterior']);
  const p = [0.5, 0.2, 0.3];
  const fromTyped = mv.unconstrainAll({ p: Float64Array.from(p) });
  const fromArray = mv.unconstrainAll({ p });
  const fromValue = mv.unconstrainAll({ p: { shape: [3], data: Float64Array.from(p) } });
  for (let i = 0; i < mv.dim; i++) {
    assert.ok(Math.abs(fromArray[i] - fromTyped[i]) < 1e-12, `array coord ${i}`);
    assert.ok(Math.abs(fromValue[i] - fromTyped[i]) < 1e-12, `Value coord ${i}`);
  }
});

test('a simplex latent whose prior measure will not materialise is refused, not flattened', async () => {
  // model-view reconciles a Dirichlet latent's width from its materialised
  // prior pool; `enumerateLatents` reports it as scalar until then. With the
  // pool unavailable the width is unknown, and silently treating it as one
  // scalar coordinate would drop the simplex constraint again.
  const { ctx } = ctxFor(MODEL, 200);
  const inner = ctx.getMeasure;
  const blind = Object.assign({}, ctx, {
    getMeasure: (n: string) => {
      if (n === 'p') throw new Error('prior pool unavailable (test)');
      return inner(n);
    },
  });
  await assert.rejects(
    () => MV.buildModelViewFromCtx(blind, ctx.derivations['posterior']),
    (e: any) => {
      assert.match(e.message, /simplex latent 'p' needs a vector shape/, `unexpected: ${e.message}`);
      return true;
    },
  );
});

test('a latent whose support has no unconstraining transform is refused, not sampled', async () => {
  // LKJ classifies to its own derivation kind, so before this it fell through to
  // `real` support and the sampler explored the ambient matrix space.
  const SRC = `
R ~ LKJ(n = 3, eta = 2.0)
prior = lawof(record(R = R))
obs_dist = joint(z = Normal(mu = 0.0, sigma = 1.0))
K = functionof(obs_dist, R = R)
L = likelihoodof(K, record(z = 0.0))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(SRC, 200);
  await assert.rejects(
    () => materialiser.materialiseMeasure('posterior', ctx,
      Object.assign({}, VIEWER, { backend: 'mh', warmup: 10, draws: 10 })),
    (e: any) => {
      assert.match(e.message, /no unconstraining transform|not yet supported|LKJ/,
        `unexpected message: ${e.message}`);
      return true;
    },
  );
});
