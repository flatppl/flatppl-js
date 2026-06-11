'use strict';

// Boundary-feeding fix (audit §3 / H1, H6): bayesupdate must feed the kernel's
// parametric input from the PRIOR's atoms (spec lowering
// bayesupdate(L,prior)=logweighted(fn(logdensityof(L,_)),prior)), NOT
// re-materialise a like-named module binding via getMeasure. The bug was
// silent precisely off the canonical prior==lawof(draws) idiom.
//
// Decisive counterexample: the kernel's boundary `theta` ORIGINATES as a tight
// spike draw (Normal(0, 0.01)), but the prior given to bayesupdate is a WIDE,
// separate measure (Normal(0, 10)) and the data sit far from the spike (~2).
//   - OLD (buggy): kernel scored at the spike draw (~0) → posterior ≈ the spike
//     mean ~0, IGNORING the data.
//   - FIXED: kernel scored at the PRIOR's atoms → IS posterior concentrates
//     near the data MLE (~2).

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const der = require(ENG + 'derivations.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function materialisePosterior(src: string, N: number) {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  for (const nm of ['posterior', 'L']) {
    if (built.bindings.has(nm) && !built.derivations[nm]) {
      const c = der.classifyDerivation(built.bindings.get(nm), built.bindings);
      if (c) built.derivations[nm] = c;
    }
  }
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 4242 });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N, rootKey: 4242,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n); const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p; },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx.getMeasure('posterior');
}

// Self-normalized weighted mean of a scalar reweighted ensemble.
function weightedMean(m: any) {
  const x = m.samples, lw = m.logWeights;
  const N = x.length;
  if (!lw) { let s = 0; for (let i = 0; i < N; i++) s += x[i]; return s / N; }
  let mx = -Infinity;
  for (let i = 0; i < N; i++) if (lw[i] > mx) mx = lw[i];
  let sw = 0, sxw = 0;
  for (let i = 0; i < N; i++) { const wgt = Math.exp(lw[i] - mx); sw += wgt; sxw += x[i] * wgt; }
  return sxw / sw;
}

test('bayesupdate: scores at the PRIOR atoms, not the boundary draw (prior != lawof(draw))', async () => {
  const src = `
theta ~ Normal(0.0, 0.01)
obs ~ iid(Normal(mu = theta, sigma = 1.0), 5)
fwd = kernelof(record(obs = obs), theta = theta)
obs_data = [2.0, 2.1, 1.9, 2.2, 1.8]
L = likelihoodof(fwd, record(obs = obs_data))
prior = Normal(0.0, 10.0)
posterior = bayesupdate(L, prior)
`;
  const post = await materialisePosterior(src, 8000);
  const mean = weightedMean(post);
  process.stderr.write(`  posterior mean(theta) = ${mean.toFixed(3)} (data ~2.0; spike-draw bug would give ~0)\n`);
  // FIXED: IS posterior over the wide prior, concentrated by the likelihood
  // near the data MLE ~2. The old spike-draw bug gave ~0.
  assert.ok(mean > 1.4,
    `posterior mean ${mean.toFixed(3)} not near the data (~2) — scoring at the boundary draw, not the prior (the H1 bug)`);
});

test('bayesupdate: aligned idiom (prior == lawof(draws)) still correct', async () => {
  // Regression: the canonical conjugate Normal shape must be unchanged.
  // theta ~ Normal(0, 2); obs ~ Normal(theta, 1) x5; prior = lawof(theta).
  // Posterior over theta given data ~3 concentrates between prior (0) and data.
  const src = `
theta ~ Normal(0.0, 2.0)
obs ~ iid(Normal(mu = theta, sigma = 1.0), 5)
fwd = kernelof(record(obs = obs), theta = theta)
obs_data = [3.0, 3.1, 2.9, 3.2, 2.8]
L = likelihoodof(fwd, record(obs = obs_data))
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
`;
  const post = await materialisePosterior(src, 8000);
  // Record posterior: the reweighting is at the record level (post.logWeights);
  // the field carries the (unweighted) atoms.
  const fld = post.fields
    ? { samples: post.fields.theta.samples, logWeights: post.logWeights }
    : post;
  const mean = weightedMean(fld);
  process.stderr.write(`  aligned posterior mean(theta) = ${mean.toFixed(3)} (between prior 0 and data 3)\n`);
  // Conjugate posterior mean = (data_sum/sigma^2) / (n/sigma^2 + 1/tau^2) with
  // tau=2,sigma=1,n=5,mean_data~3 → ~2.78. Just assert it moved toward the data.
  assert.ok(mean > 2.0 && mean < 3.0,
    `aligned posterior mean ${mean.toFixed(3)} off the conjugate range (~2.6-2.9)`);
});

// ─────────────────────────────────────────────────────────────────────────
// Cascade-prune regression: a bayesupdate over a likelihood of a USER-KERNEL
// whose body is a LIFT-INTRODUCED generative composite must KEEP its
// derivation (so the viewer plots it). liftInlineSubexpressions runs after
// computePhases, so a hoisted composite (`ys = post.(xs)` dot-broadcast)
// carries phase == null; the cascade-prune used to treat that null-phase
// body-internal ref as an unresolvable fixed dep and prune the WHOLE posterior
// → "Not plottable", even though it materialises correctly. The prune now
// skips lift-introduced (null-phase) body internals, same as parameterized /
// stochastic ones (the materialiser resolves them via the expand path).
// ─────────────────────────────────────────────────────────────────────────

test('bayesupdate over a generative-composite likelihood keeps its derivation (not cascade-pruned)', () => {
  const src = `
mu = elementof(reals)
x ~ Normal(mu, 1.0)
y = x * 2.0
post = z -> z + 1.0
k_inner = kernelof(y, mu = mu)
n = elementof(posintegers)
xs ~ iid(k_inner(mu), n)
ys = post.(xs)
k = kernelof(ys, mu = mu, n = n)
data = [1.0, 2.0, 3.0]
km = mu -> k(lengthof(data), mu)
L = likelihoodof(km, data)
prior = joint(mu = Normal(0.0, 1.0))
posterior = bayesupdate(L, prior)
`;
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const d = built.derivations.posterior;
  assert.ok(d, 'posterior must keep a derivation (not cascade-pruned)');
  assert.strictEqual(d.kind, 'bayesupdate');
});
