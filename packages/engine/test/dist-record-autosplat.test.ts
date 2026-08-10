'use strict';

// Auto-splatting a record into a distribution constructor — spec §04
// "Calling conventions": `Dist(record(a = x, b = y))` is equivalent to
// `Dist(a = x, b = y)`, and so is any expression that evaluates to such a
// record (e.g. a value-function call returning `record(...)`). The
// type-checker already accepted these forms, but resolveParams left the
// lone positional record unsplit, so both the sampler and the density
// path threw `'<Dist>' missing parameter '<p>'`. resolveParams now splats
// a single positional record whose field names cover the parameters.
//
// INDEPENDENT ORACLE — Distributions.jl (NOT any engine output):
//   Gamma(shape = 4, rate = 2)  ≡  Distributions.Gamma(4, 1/2)
//     logpdf@1.0 = -1.0191707469882738
//     logpdf@2.0 = -0.9397292053084376
//     mean = 2.0,  var = 1.0
//   Normal(mu = 0, sigma = 1)
//     logpdf@0.0 = -0.9189385332046727   (= -log √(2π))

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function buildCtx(src: string, N: number, seed: number) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: seed, rootSeed: seed, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

// The reported repro: a value function returns the shape/rate record and is
// passed positionally to Gamma. gamma_shape_rate(2, 1) → shape=4, rate=2.
const FN_RETURN = `gamma_shape_rate(mu, sd) = record(shape = mu^2 / sd^2, rate = mu / sd^2)
ld = logdensityof(Gamma(gamma_shape_rate(2.0, 1.0)), 1.0)`;

test('dist auto-splat: Gamma(fn → record) density matches the Distributions.jl oracle', async () => {
  const ORACLE = -1.0191707469882738;   // Distributions.jl; NOT an engine output
  const ctx = buildCtx(FN_RETURN, 1, 1);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - ORACLE) < 1e-9,
    `score ${m.samples[0]} ≈ oracle ${ORACLE} (Δ ${Math.abs(m.samples[0] - ORACLE)})`);
});

test('dist auto-splat: Gamma(record literal) density matches the oracle at two points', async () => {
  for (const [x, ORACLE] of [[1.0, -1.0191707469882738], [2.0, -0.9397292053084376]]) {
    const ctx = buildCtx(`ld = logdensityof(Gamma(record(shape = 4.0, rate = 2.0)), ${x})`, 1, 1);
    const m = await ctx.getMeasure('ld');
    assert.ok(Math.abs(m.samples[0] - ORACLE) < 1e-9,
      `Gamma logpdf@${x}: ${m.samples[0]} ≈ oracle ${ORACLE} (Δ ${Math.abs(m.samples[0] - ORACLE)})`);
  }
});

test('dist auto-splat: documented Normal(record(mu, sigma)) form scores (spec §02)', async () => {
  const ORACLE = -0.9189385332046727;   // -log √(2π)
  const ctx = buildCtx(`ld = logdensityof(Normal(record(mu = 0.0, sigma = 1.0)), 0.0)`, 1, 1);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - ORACLE) < 1e-9,
    `Normal logpdf@0: ${m.samples[0]} ≈ oracle ${ORACLE} (Δ ${Math.abs(m.samples[0] - ORACLE)})`);
});

test('dist auto-splat: sampling the splatted Gamma agrees with the density moments', async () => {
  // Scar-zone guard: the sample path and the density path of the SAME measure
  // must agree. Gamma(shape=4, rate=2) has mean 2, variance 1.
  const ctx = buildCtx(`sigma ~ Gamma(record(shape = 4.0, rate = 2.0))`, 60000, 17);
  const m = await ctx.getMeasure('sigma');
  const s = m.samples;
  let mean = 0; for (const v of s) mean += v; mean /= s.length;
  let varr = 0; for (const v of s) varr += (v - mean) * (v - mean); varr /= s.length;
  assert.ok(Math.abs(mean - 2.0) < 0.05, `sample mean ${mean} ≈ 2.0`);
  assert.ok(Math.abs(varr - 1.0) < 0.05, `sample var ${varr} ≈ 1.0`);
});

test('dist auto-splat: a field/parameter name mismatch still errors (spec static error)', async () => {
  // `rate` misspelled `rat` — the splat fires regardless (§04 always-splat),
  // and the misspelling is reported as the unbindable name it is rather than
  // as the missing `rate` it leaves behind.
  const ctx = buildCtx(`ld = logdensityof(Gamma(record(shape = 4.0, rat = 2.0)), 1.0)`, 1, 1);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('ld')),
    /'Gamma' has no parameter 'rat'/);
});

test('dist auto-splat: a SURPLUS field is an error, not silently dropped', async () => {
  // §04's name rule runs both directions. The per-parameter loop reads out
  // of the splatted kwargs, so it only ever caught a parameter with nothing
  // bound; `zz` was ignored and the model scored as if it were not written.
  const ctx = buildCtx(
    `ld = logdensityof(Normal(record(mu = 1.1, sigma = 0.2, zz = 9.0)), 1.1)`, 1, 1);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('ld')),
    /'Normal' has no parameter 'zz' \(parameters: mu, sigma\)/);
});

test('dist auto-splat: a surplus KEYWORD argument is an error too', async () => {
  // Same rule, written without a record — the check sits after the splat so
  // it covers both spellings.
  const ctx = buildCtx(
    `ld = logdensityof(Normal(mu = 1.1, sigma = 0.2, zz = 9.0), 1.1)`, 1, 1);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('ld')),
    /'Normal' has no parameter 'zz'/);
});

test('dist auto-splat: a surplus field on the SAMPLING path reports legibly', async () => {
  // The sampling path had no worker-error guard, so every worker-side
  // parameter failure surfaced as "Cannot read properties of undefined
  // (reading 'length')" — including a plain missing parameter, before this
  // check existed.
  const ctx = buildCtx(`s ~ Normal(record(mu = 1.1, sigma = 0.2, zz = 9.0))`, 4096, 17);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('s')),
    /worker failed sampling.*'Normal' has no parameter 'zz'/);
});

test('dist auto-splat: a missing parameter on the SAMPLING path reports legibly', async () => {
  const ctx = buildCtx(`s ~ Normal(mu = 1.1)`, 4096, 17);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('s')),
    /worker failed sampling.*'Normal' missing parameter 'sigma'/);
});

test('dist auto-splat: the BATCHED resolver defers a surplus name to the canonical error', async () => {
  // `mu = t` references an upstream draw, which selects the batched per-atom
  // resolver (resolveParamsN) over the scalar one. That resolver carried the
  // missing-parameter throw but not the surplus check, so this drew exactly
  // the same 4096 atoms as the spelling without `zz`. It now returns null,
  // the caller falls back to the scalar path, and resolveParams raises the one
  // canonical message — the resolver's never-throw contract is preserved.
  const ctx = buildCtx(`t ~ Normal(0.0, 1.0)
s ~ Normal(mu = t, sigma = 0.2, zz = 9.0)`, 4096, 17);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('s')),
    /'Normal' has no parameter 'zz' \(parameters: mu, sigma\)/);
});

test('dist auto-splat: the batched resolver still draws the valid hierarchical shape', async () => {
  // Control for the test above — the same model without `zz` must still batch.
  const ctx = buildCtx(`t ~ Normal(0.0, 1.0)
s ~ Normal(mu = t, sigma = 0.2)`, 4096, 17);
  const s = (await ctx.getMeasure('s')).samples;
  assert.equal(s.length, 4096);
  // Var(s) = Var(t) + 0.2^2 = 1.04 for the marginal of this two-level model.
  let mean = 0; for (const v of s) mean += v; mean /= s.length;
  let varr = 0; for (const v of s) varr += (v - mean) * (v - mean); varr /= s.length;
  assert.ok(Math.abs(mean) < 0.06, `sample mean ${mean} ≈ 0`);
  assert.ok(Math.abs(varr - 1.04) < 0.08, `sample var ${varr} ≈ 1.04`);
});

test('dist auto-splat: a ONE-parameter constructor also errors on a bad field name', async () => {
  // §04 always-splat. Gating the splat on the fields covering the params
  // left this record bound to `rate` itself, and `Poisson(<record>)` then
  // scored NaN with no error — a single-parameter constructor was the one
  // shape where the name rule went unenforced entirely.
  const ctx = buildCtx(`ld = logdensityof(Poisson(record(zzz = 0.5)), 2)`, 1, 1);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('ld')),
    /'Poisson' has no parameter 'zzz' \(parameters: rate\)/);
});
