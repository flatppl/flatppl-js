'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const errs = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) return { errs, ctx: null, built: null, lifted };
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler(); worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    moduleRegistry: lifted.loweredModule.moduleRegistry || null,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n); const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p; },
    sendWorker: (m: any) => { const r = worker.handle(m); return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r); },
    sampleCount: N, rootSeed: 42,
  };
  return { errs: [], ctx, built, lifted };
}
const DATA = '[1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]';
const ORACLE = -37.99959196868393;
async function score(src: string) {
  const { errs, ctx, lifted } = setupCtx(src, 1);
  if (errs.length) throw new Error('DIAG: ' + JSON.stringify((lifted.diagnostics || []).map((d: any) => d.message)));
  const m = await ctx!.getMeasure('__score__');
  return m.value ? m.value.data[0] : m.samples[0];
}

test('C1: joint(relabel(...)) prior classifies + materialises to a record variate', async () => {
  const { errs, ctx, built } = setupCtx(`
flatppl_compat = "0.1"
prior = joint(relabel(Normal(0, 1), ["theta1"]), relabel(Exponential(1), ["theta2"]))
`, 20);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.prior, 'prior classifies to a derivation');
  const m = await ctx.getMeasure('prior');
  assert.ok(m && m.fields && m.fields.theta1 && m.fields.theta2,
    `prior is a record variate with fields theta1, theta2 (got keys ${m && Object.keys(m)})`);
});

test('C1: relabel(joint(...)) prior classifies + materialises to a record variate', async () => {
  const { errs, ctx, built } = setupCtx(`
flatppl_compat = "0.1"
prior = relabel(joint(Normal(0, 1), Exponential(1)), ["theta1", "theta2"])
`, 20);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.prior, 'prior classifies to a derivation');
  const m = await ctx.getMeasure('prior');
  assert.ok(m && m.fields && m.fields.theta1 && m.fields.theta2,
    `prior is a record variate with fields theta1, theta2 (got keys ${m && Object.keys(m)})`);
});

// The EXPLICIT-boundary path: logdensityof(L, θ) with a literal vector point.
// matLikelihoodDensity feeds θ as an atom-independent `explicit` input, so this
// exercises the vector-point scoring path — NOT the C2 bindOne/byFrom feed
// (a literal point never materialises the prior nor flows through feedInputs'
// boundary link). Kept as a companion regression guard for that path.
test('explicit-boundary: logdensityof over a lone vector-point kernel input', async () => {
  const v = await score(`
flatppl_compat = "0.1"
prior = joint(Normal(0, 1), Exponential(1))
theta = elementof(cartpow(reals, 2))
a = theta[1]
b = abs(theta[2])
obs ~ iid(Normal(mu = a, sigma = b), 10)
forward_kernel = kernelof(record(obs = obs), theta = theta)
L = likelihoodof(forward_kernel, record(obs = ${DATA}))
theta_pt = [0.5, 1.5]
__score__ = logdensityof(L, theta_pt)
`);
  assert.ok(Math.abs(v - ORACLE) < 1e-9, `got ${v}`);
});

// The C2 fix proper: the kchain MC-marginal density feeds the whole positional
// (non-record) `prior` variate into the kernel's lone REFERENCED boundary input
// through feedInputs/bindOne's byFrom link. Pre-fix this threw
// `matScore: … no fed column covers it`; the whole-non-record-variate bind
// (clm.ts bindOne) closes the gap. Normal–Normal conjugate so the marginal is
// closed form: ∫ N(1; mu, 1) N(mu; 0, 1) dmu = N(1; 0, sqrt2). sigma is FIXED
// (not theta[2]) to keep the integral analytic; theta stays a positional
// 2-vector so the prior is non-record and drives the arity-1 whole-variate feed.
test('C2: kchain feeds a whole positional-joint prior into a lone boundary input', async () => {
  const N = 20000;
  const { errs, ctx, lifted } = setupCtx(`
flatppl_compat = "0.1"
prior = joint(Normal(0, 1), Normal(0, 1))
theta = elementof(cartpow(reals, 2))
a = theta[1]
K = functionof(Normal(mu = a, sigma = 1), theta = theta)
ch = kchain(prior, K)
__score__ = logdensityof(ch, 1.0)
`, N);
  assert.equal(errs.length, 0,
    'DIAG: ' + JSON.stringify((lifted.diagnostics || []).map((d: any) => d.message)));
  const m = await ctx!.getMeasure('__score__');
  const v = m.value ? m.value.data[0] : m.samples[0];
  // ∫ N(1; mu, 1) N(mu; 0, 1) dmu = N(1; 0, sqrt2); scipy norm.logpdf(1, 0, sqrt2).
  const CONJ_ORACLE = -1.5155121234846454;
  assert.ok(Math.abs(v - CONJ_ORACLE) < 0.05,
    `kchain MC marginal ${v} should match the closed-form ${CONJ_ORACLE}`);
});
