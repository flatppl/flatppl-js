'use strict';

// =====================================================================
// chain-variate-domain.test.ts — a positional chain's variate is an ARRAY
// =====================================================================
//
// §06 dependent composition, on `jointchain(M, K1, K2, ...)`: "In contrast
// to `kchain`, the output variate is the `cat` of the variates of all the
// components, as with `joint`." The same shape as `joint`, by the same
// operator.
//
// §06's own density rule writes that variate as an ARRAY literal:
// "`jointchain` (the product of the constituent conditional densities):
// logdensityof(jointchain(M, K), [a, b]) = logdensityof(M, a) +
// logdensityof(K(a), b)".
//
// §04 sec:tuples forbids the alternative outright: "Tuples are objects,
// not values. They have no `valueset`, are not drawn from measures, and
// are not part of the measure algebra ... Measures, kernels, and
// likelihoods never use tuples as their domain."
//
// The engine typed the positional chain's variate as a tuple, so §06's
// array-literal rule was a static error while the tuple literal the spec
// never writes type-checked. The runtime density path always consumed the
// array correctly — only the type was wrong — so this pins the array as
// the accepted literal, the tuple as the refusal, and the iid-over-chain
// density path the array domain unlocks.
//
// Every number is hand-derived from the factorisation, never read off an
// engine output.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function errorsOf(source: string): any[] {
  const lifted = processSource(source);
  const built = orchestrator.buildDerivations(lifted.bindings);
  return (lifted.diagnostics || []).concat(built.diagnostics || [])
    .filter((d: any) => d.severity === 'error');
}

function setupCtx(source: string, N: number, seed: number) {
  const lifted = processSource(source);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => {
      const reply = worker.handle(m);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: N, rootSeed: seed, rootKey: [seed, 0],
    marginalizationCount: 32,
  };
  return ctx;
}

/** Normal log-pdf, closed form — the density oracle. */
function normLogpdf(x: number, mu: number, sigma: number) {
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma)
    - 0.5 * ((x - mu) / sigma) * ((x - mu) / sigma);
}

const CHAIN3 = `
flatppl_compat = "0.1"
k1 = fn(Normal(mu = _, sigma = 1.0))
k2 = fn(Normal(mu = sum(_), sigma = 1.0))
ch = jointchain(Normal(mu = 0.0, sigma = 1.0), k1, k2)
`;

const CHAIN2 = `
flatppl_compat = "0.1"
k1 = fn(Normal(mu = _, sigma = 1.0))
ch = jointchain(Normal(mu = 0.0, sigma = 1.0), k1)
`;

// =====================================================================
// The array literal §06 writes
// =====================================================================

test('§06 array-literal density rule type-checks and scores', async () => {
  const src = CHAIN3 + `
ld = logdensityof(ch, [0.3, 0.7, 1.1])
`;
  assert.deepEqual(errorsOf(src).map((d: any) => d.message), []);
  const want = normLogpdf(0.3, 0.0, 1.0)
             + normLogpdf(0.7, 0.3, 1.0)
             + normLogpdf(1.1, 0.3 + 0.7, 1.0);
  const m = await setupCtx(src, 4, 3).getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `logdensityof ${m.samples[0]} vs closed form ${want}`);
});

test('the 2-component chain scores at an array literal', async () => {
  const src = CHAIN2 + `
ld = logdensityof(ch, [0.3, 0.7])
`;
  assert.deepEqual(errorsOf(src).map((d: any) => d.message), []);
  const want = normLogpdf(0.3, 0.0, 1.0) + normLogpdf(0.7, 0.3, 1.0);
  const m = await setupCtx(src, 4, 3).getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `logdensityof ${m.samples[0]} vs closed form ${want}`);
});

test('kchain keeps its last variate alone, unaffected', () => {
  // The unification touches the retain arm only. kchain marginalises, so its
  // variate stays the last step's scalar and an array literal is refused.
  const src = `
flatppl_compat = "0.1"
k1 = fn(Normal(mu = _, sigma = 1.0))
ch = kchain(Normal(mu = 0.0, sigma = 1.0), k1)
ld = logdensityof(ch, 0.7)
`;
  assert.deepEqual(errorsOf(src).map((d: any) => d.message), []);
  const arrErrs = errorsOf(`
flatppl_compat = "0.1"
k1 = fn(Normal(mu = _, sigma = 1.0))
ch = kchain(Normal(mu = 0.0, sigma = 1.0), k1)
ld = logdensityof(ch, [0.3, 0.7])
`);
  assert.ok(arrErrs.length >= 1, 'kchain has a scalar variate here');
  assert.match(arrErrs[0].message, /expects real, got array of real/);
});

// =====================================================================
// The tuple literal §04 forbids
// =====================================================================

test('a tuple literal is refused — a measure never has a tuple domain', () => {
  const errs = errorsOf(CHAIN3 + `
ld = logdensityof(ch, tuple(0.3, 0.7, 1.1))
`);
  assert.ok(errs.length >= 1, 'a tuple variate must be a static error');
  assert.match(errs[0].message, /expects array of real \(length 3\)/);
  assert.match(errs[0].message, /got tuple/);
});

test('joint and jointchain accept the same literal', () => {
  // The divergence this closes: each construct used to reject the
  // other's literal, though §06 gives them one variate operator.
  const jointErrs = errorsOf(`
flatppl_compat = "0.1"
j = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))
ld = logdensityof(j, [0.3, 0.7])
`);
  const chainErrs = errorsOf(CHAIN2 + `
ld = logdensityof(ch, [0.3, 0.7])
`);
  assert.deepEqual(jointErrs.map((d: any) => d.message), []);
  assert.deepEqual(chainErrs.map((d: any) => d.message), []);
});

// =====================================================================
// iid over a positional chain — the density path the array domain unlocks
// =====================================================================

test('iid over a jointchain scores as the product over rows', async () => {
  // §06 iid: the rows are independent, each scored by the chain's own
  // factorisation. x_0 ~ N(0,1), x_1 | x_0 ~ N(x_0, 1).
  const src = CHAIN2 + `
m = iid(ch, 2)
ld = logdensityof(m, [[0.3, 0.7], [0.4, 0.8]])
`;
  assert.deepEqual(errorsOf(src).map((d: any) => d.message), []);
  const want = normLogpdf(0.3, 0.0, 1.0) + normLogpdf(0.7, 0.3, 1.0)
             + normLogpdf(0.4, 0.0, 1.0) + normLogpdf(0.8, 0.4, 1.0);
  const m = await setupCtx(src, 4, 3).getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `iid-over-chain logdensityof ${m.samples[0]} vs closed form ${want}`);
});

test('iid over a 3-step jointchain scores the cat-fed third term', async () => {
  const src = CHAIN3 + `
m = iid(ch, 2)
ld = logdensityof(m, [[0.3, 0.7, 1.1], [0.4, 0.8, 1.2]])
`;
  assert.deepEqual(errorsOf(src).map((d: any) => d.message), []);
  const row = (a: number, b: number, c: number) =>
    normLogpdf(a, 0.0, 1.0) + normLogpdf(b, a, 1.0) + normLogpdf(c, a + b, 1.0);
  const want = row(0.3, 0.7, 1.1) + row(0.4, 0.8, 1.2);
  const m = await setupCtx(src, 4, 3).getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `iid-over-chain3 logdensityof ${m.samples[0]} vs closed form ${want}`);
});

test('iid over a positional joint keeps scoring the same way', async () => {
  const src = `
flatppl_compat = "0.1"
j = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))
m = iid(j, 2)
ld = logdensityof(m, [[0.3, 0.7], [0.4, 0.8]])
`;
  assert.deepEqual(errorsOf(src).map((d: any) => d.message), []);
  const want = normLogpdf(0.3, 0.0, 1.0) + normLogpdf(0.7, 1.0, 2.0)
             + normLogpdf(0.4, 0.0, 1.0) + normLogpdf(0.8, 1.0, 2.0);
  const m = await setupCtx(src, 4, 3).getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `iid-over-joint logdensityof ${m.samples[0]} vs closed form ${want}`);
});
