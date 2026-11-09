'use strict';
// §06 truncate(M, S): ν(A) = M(A ∩ S), not normalized. A vector-valued parent
// truncated to a vector set — ordered(n), cartpow(S, n), a positional
// cartprod — refused at materialisation ("unsupported set kind 'vectorset'"),
// and a truncation to cartpow(posreals, 2) lost its power. Found through
// flatppl-examples/posteriordb/hmm_example.flatppl, whose mu prior is
// truncate(truncate(joint(Normal(3, 1), Normal(10, 1)), ordered(2)), cartpow(posreals, 2)).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');
const ms = require('../model-spec.ts');

const N = 20000;
const P = 'P = joint(Normal(0.0, 1.0), Normal(0.0, 1.0))\n';

async function scalar(ctx: any, name: string): Promise<number> {
  const m = await ctx.getMeasure(name);
  return (m.value ? m.value.data : m.samples)[0];
}
// A Monte-Carlo mass p from N atoms: allow 6 binomial standard errors.
function closeMass(got: number, p: number, label: string) {
  const tol = 6 * Math.sqrt(p * (1 - p) / N);
  assert.ok(Math.abs(got - p) < tol, `${label}: ${got} vs ${p} ± ${tol}`);
}

test('a truncation to ordered(2) keeps the ordered atoms and has mass 1/2', async () => {
  // Two iid standard normals are ordered with probability 1/2.
  const { ctx } = ctxFor(P + 'T = truncate(P, ordered(2))\ntm = totalmass(T)\n', N);
  closeMass(await scalar(ctx, 'tm'), 0.5, 'totalmass');
  const t = await ctx.getMeasure('T');
  const [a, b] = [t.elems[0].samples, t.elems[1].samples];
  let live = 0;
  for (let i = 0; i < N; i++) {
    if (t.logWeights[i] === -Infinity) continue;
    live++;
    assert.ok(a[i] < b[i], `atom ${i}: ${a[i]} ≥ ${b[i]}`);
  }
  assert.ok(live > 0.45 * N && live < 0.55 * N, `live ${live}`);
});

test('a nested truncation to cartpow(posreals, 2) has mass 1/8 and positive ordered draws', async () => {
  // P(0 < x1 < x2) = P(both positive) · P(ordered | both positive) = 1/4 · 1/2.
  const { ctx } = ctxFor(P + 'T = truncate(P, ordered(2))\nT2 = truncate(T, cartpow(posreals, 2))\ntm = totalmass(T2)\n', N);
  closeMass(await scalar(ctx, 'tm'), 1 / 8, 'totalmass');
  const t = await ctx.getMeasure('T2');
  const [a, b] = [t.elems[0].samples, t.elems[1].samples];
  for (let i = 0; i < N; i++) {
    if (t.logWeights[i] === -Infinity) continue;
    assert.ok(a[i] > 0 && a[i] < b[i], `atom ${i}: (${a[i]}, ${b[i]})`);
  }
});

test('a vector set of the wrong width refuses accurately', async () => {
  const { ctx } = ctxFor(P + 'T = truncate(P, ordered(3))\n', 8);
  await assert.rejects(() => ctx.getMeasure('T'),
    /truncate\(M, S\): the set has 3 coordinates but the variates of 'P' have 2 \(spec §06/);
});

// flatppl-examples/posteriordb/hmm_example.flatppl, with five inline data points.
const HMM = `
theta1 ~ Dirichlet([1.0, 1.0])
theta2 ~ Dirichlet([1.0, 1.0])
mu = elementof(ordered(2))
mu_prior = truncate(truncate(joint(Normal(3, 1), Normal(10, 1)), ordered(2)), cartpow(posreals, 2))
data = record(y = [2.9, 3.4, 9.8, 10.1, 3.0])
log_ratio = 0.5 .* ((data.y .- mu[2]).^2 .- (data.y .- mu[1]).^2)
predict(q, r, p) = p.theta2[1] + (p.theta1[1] - p.theta2[1]) * invlogit(logit(q) + r)
step(state, r, p) = record(current = state.next, next = predict(state.next, r, p))
states = scan(step, record(current = 0.5, next = 0.5), log_ratio,
    record(theta1 = theta1, theta2 = theta2))
state_weight(state) = state.current
weight = state_weight.(states)
emission(w, mu1, mu2) = normalize(superpose(
    weighted(w, Normal(mu1, 1)), weighted(1 - w, Normal(mu2, 1))))
y ~ emission.(weight, mu[1], mu[2])
Pi_prior = joint(lawof(record(theta1 = theta1, theta2 = theta2)), joint(mu = mu_prior))
K_model = kernelof(record(y = y), theta1 = theta1, theta2 = theta2, mu = mu)
L = likelihoodof(K_model, record(y = data.y))
Pi_post = bayesupdate(L, Pi_prior)
`;

function hmmSource(muPrior?: string): string {
  return muPrior ? HMM.replace(/mu_prior = [^\n]*/, 'mu_prior = ' + muPrior) : HMM;
}

test('the hmm latents get their charts and widths from the model alone', () => {
  const { ctx } = ctxFor(hmmSource(), 8);
  const latents = ms.enumerateLatents(ctx.derivations.Pi_post, ctx);
  const byName = Object.fromEntries(latents.map((l: any) => [l.name, l]));
  // A Dirichlet's simplex width is its concentration's length.
  assert.deepEqual(byName.theta1.support, { kind: 'simplex', n: 2 });
  // Nested full-dimensional truncations take the outermost chart.
  assert.deepEqual(byName.mu.support, { kind: 'positive', n: 2 });
  assert.deepEqual(byName.mu.shape, { kind: 'vector', dims: [2] });
});

test('a lower-dimensional set in a truncation chain fixes the chart', () => {
  const { ctx } = ctxFor('d = draw(truncate(truncate(Dirichlet([1.0, 1.0, 1.0]), stdsimplex(3)), cartpow(posreals, 3)))\n'
    + 'P = lawof(record(d = d))\nL = likelihoodof(functionof(Normal(d[1], 1.0), d = d), 0.5)\npost = bayesupdate(L, P)\n', 8);
  const latents = ms.enumerateLatents(ctx.derivations.post, ctx);
  assert.deepEqual(latents.find((l: any) => l.name === 'd').support, { kind: 'simplex', n: 3 });
});

test('the hmm model view builds for importance sampling and MH', async () => {
  for (const backend of ['is', 'mh']) {
    const { ctx } = ctxFor(hmmSource(), 200);
    ctx.inferenceOpts = { backend };
    const v = await buildModelViewFromCtx(ctx, ctx.derivations.Pi_post);
    assert.deepEqual(v.names, ['theta1[0]', 'theta1[1]', 'theta2[0]', 'theta2[1]', 'mu[0]', 'mu[1]'], backend);
    assert.equal(v.dim, 4, backend);
  }
});

test('a joint-of-distributions vector field is a vector latent', async () => {
  const { ctx } = ctxFor(hmmSource('joint(Normal(3, 1), Normal(10, 1))'), 200);
  ctx.inferenceOpts = { backend: 'mh' };
  const v = await buildModelViewFromCtx(ctx, ctx.derivations.Pi_post);
  assert.deepEqual(v.names.slice(-2), ['mu[0]', 'mu[1]']);
});

test('importance sampling feeds a tuple-form vector field of the prior', async () => {
  const { ctx } = ctxFor(hmmSource(), 400);
  ctx.inferenceOpts = { backend: 'is' };
  const m = await ctx.getMeasure('Pi_post');
  const mu = m.fields.mu;
  const a = mu.elems ? mu.elems[0].samples : mu.value.data;
  assert.ok(a.length > 0 && Array.from(m.logWeights).some(Number.isFinite));
});
