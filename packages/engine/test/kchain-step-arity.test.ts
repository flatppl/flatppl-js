'use strict';
// Chain step boundaries carry the `cat` of every variate to the step's left,
// not just the previous step's variate. Spec §06 "Dependent composition":
//
//   model = kchain(M1, K2, K3)
//
// "is equivalent to
//
//   a ~ M1
//   b ~ K2(a)
//   c ~ K3([a, b])
//   model = lawof(c)"
//
// — so step 3 consumes `[a, b]`. §06 then rules how that value binds: "A
// non-record variate — for example the `cat`'d variate of a positional
// `joint` — carries no field names, so it feeds a kernel only when the kernel
// has a single input, to which the whole value is bound; feeding one to a
// kernel with two or more inputs is a static error".
//
// Typing the boundary as step 2's variate alone put loud and silent exactly
// backwards. A third step spelled `b -> Normal(mu = b, sigma = 1)` type-checked
// `mu` against a real while the materialiser fed it the whole `[a, b]` pair, so
// the chain SAMPLED NaN with no diagnostic; the spelling that indexes the pair
// was fine but its 2-input sibling reported the wrong fed type.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function errorsOf(src: string) {
  const lifted = processSource(src);
  return (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
}

function setupCtx(src: string, N: number, seed: number) {
  const lifted = processSource(src);
  const errs = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'model must type cleanly');
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    moduleRegistry: lifted.loweredModule.moduleRegistry || null,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
    sampleCount: N, rootSeed: seed,
  };
  return ctx;
}

// The oracle model, a linear-Gaussian chain whose marginal is closed form:
//   a ~ N(0, 1);  b | a ~ N(2a, 1);  c | [a, b] ~ N(b/2, 3)
// var(b) = 4·1 + 1 = 5;  var(c) = ¼·5 + 9 = 10.25;  mean 0 throughout.
// Step 2 reads `ab[2]` — the second slot of the fed `[a, b]` — so the model
// only holds its oracle if the whole pair really arrives.
const ORACLE_MODEL = `
flatppl_compat = "0.1"
K1 = a -> Normal(mu = 2 * a, sigma = 1)
K2 = ab -> Normal(mu = 0.5 * ab[2], sigma = 3)
ch = kchain(Normal(0, 1), K1, K2)
`;
const ORACLE_VAR = 10.25;
// scipy.stats.norm.logpdf(x, 0, sqrt(10.25)), matching the hand derivation
// -½·log(2π·10.25) − x²/(2·10.25).
const ORACLE_LOGPDF: Record<string, number> = {
  '0.0': -2.0825773859968812,
  '1.0': -2.1313578738017593,
};

test('well-formed: a single-input third step consumes the whole [a, b] pair', async () => {
  const N = 40000;
  const ctx = setupCtx(ORACLE_MODEL, N, 7);
  const m = await ctx.getMeasure('ch');
  const s: number[] = Array.from(m.value ? m.value.data : m.samples);
  assert.equal(s.length, N);
  assert.equal(s.filter((x) => Number.isNaN(x)).length, 0,
    'a well-formed chain must not sample NaN');
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  const varr = s.reduce((a, b) => a + (b - mean) ** 2, 0) / (s.length - 1);
  // se(var) ≈ var·sqrt(2/N) ≈ 0.07 at N = 40000; 5 se of headroom.
  assert.ok(Math.abs(varr - ORACLE_VAR) < 0.35,
    `marginal variance ${varr} should match the closed-form ${ORACLE_VAR}`);
  assert.ok(Math.abs(mean) < 0.1, `marginal mean ${mean} should be 0`);
});

for (const x of Object.keys(ORACLE_LOGPDF)) {
  test(`well-formed: logdensityof matches the closed-form marginal at ${x}`, async () => {
    const ctx = setupCtx(ORACLE_MODEL + `__score__ = logdensityof(ch, ${x})\n`,
      20000, 7);
    const m = await ctx.getMeasure('__score__');
    const v = m.value ? m.value.data[0] : m.samples[0];
    // The kchain marginal is a Monte-Carlo estimate over the prior atoms, so
    // the tolerance is the estimator's, not float noise.
    assert.ok(Math.abs(v - ORACLE_LOGPDF[x]) < 0.01,
      `logdensityof(ch, ${x}) = ${v} should match ${ORACLE_LOGPDF[x]}`);
  });
}

test('malformed: a scalar-named third step is a located error, never NaN', () => {
  const errs = errorsOf(`
flatppl_compat = "0.1"
K1 = a -> Normal(mu = a, sigma = 1)
K2 = b -> Normal(mu = b, sigma = 1)
ch = kchain(Normal(0, 1), K1, K2)
`);
  assert.equal(errs.length, 1, JSON.stringify(errs.map((d: any) => d.message)));
  assert.match(errs[0].message,
    /Normal: kwarg "mu" expects real, got array of real \(length 2\)/);
  // The error must point at the offending body, not at the chain call.
  assert.ok(errs[0].loc && errs[0].loc.start, 'diagnostic must carry a location');
  assert.equal(errs[0].loc.start.line, 3, 'located on the K2 line');
});

test('malformed: the `fn` placeholder spelling errors the same way', () => {
  // The shape the defect was reported under: a 3-ary kchain of `fn`
  // placeholder kernels, which lower to the same param-less `functionof`.
  for (const base of ['Normal(mu = 0, sigma = 1)', 'lawof(z)']) {
    const errs = errorsOf(`
flatppl_compat = "0.1"
z ~ Normal(0, 1)
C = kchain(${base}, fn(Normal(mu = _, sigma = 1.0)), fn(Normal(mu = _, sigma = 1.0)))
b ~ C
`);
    assert.equal(errs.length, 1, base + ': '
      + JSON.stringify(errs.map((d: any) => d.message)));
    assert.match(errs[0].message,
      /Normal: kwarg "mu" expects real, got array of real \(length 2\)/);
    assert.ok(errs[0].loc && errs[0].loc.start, 'diagnostic must carry a location');
  }
});

test('left variates that do not cat are a located error', () => {
  // §06 builds the fed value with `cat`, and `cat` admits all-scalar,
  // all-vector or all-record components only. A scalar base followed by a
  // record-variate step has no cat, so the third step has nothing to consume.
  const errs = errorsOf(`
flatppl_compat = "0.1"
K1 = a -> joint(b = Normal(mu = a, sigma = 1))
K2 = c -> Normal(mu = 0, sigma = 1)
ch = kchain(Normal(0, 1), K1, K2)
`);
  assert.equal(errs.length, 1, JSON.stringify(errs.map((d: any) => d.message)));
  assert.match(errs[0].message, /do not `cat`/);
  assert.match(errs[0].message, /all scalar, all vector, or all record/);
  assert.ok(errs[0].loc && errs[0].loc.start, 'diagnostic must carry a location');
});

// The RETAIN chain keeps the previous-variate boundary. §06 gives kchain and
// jointchain the same `c ~ K3([a, b])` lowering, but the materialiser threads
// only the previous variate through a positional jointchain, which is what
// `fixtures/hierarchical-state-space.flatppl` (a 4-step AR-1 walk of one-input
// kernels) is calibrated on in `hierarchical-models.test.ts`. Typing that
// boundary as the cat would reject a model the engine samples correctly, so
// the cat rule stops at the marginal chain until the feed itself is settled.
// The divergence is recorded in flatppl-dev/TODO-flatppl-js.md.
test('scope: the retain chain keeps the previous-variate boundary', () => {
  const errs = errorsOf(`
flatppl_compat = "0.1"
K1 = a -> Normal(mu = a, sigma = 1)
K2 = b -> Normal(mu = b, sigma = 1)
ch = jointchain(Normal(0, 1), K1, K2)
`);
  assert.deepEqual(errs.map((d: any) => d.message), []);
});

test('a 2-input third step names the cat as the fed type', () => {
  const errs = errorsOf(`
flatppl_compat = "0.1"
K1 = a -> Normal(mu = a, sigma = 1)
K2 = (a, b) -> Normal(mu = b, sigma = 1)
ch = kchain(Normal(0, 1), K1, K2)
`);
  assert.equal(errs.length, 1, JSON.stringify(errs.map((d: any) => d.message)));
  // §06: a non-record cat feeds two or more inputs never — and the reported
  // fed type is the cat `[a, b]`, not step 1's lone variate.
  assert.match(errs[0].message, /multi-input step boundary requires a record/);
  assert.match(errs[0].message, /got array of real \(length 2\)/);
  assert.ok(errs[0].loc && errs[0].loc.start, 'diagnostic must carry a location');
});

// A record variate feeding a step's LONE input ought to splat by field name
// (§04 sec:calling-convention: "A sole positional record or table therefore
// always splats"). The chain materialiser does not implement that feed — it
// binds the record whole, which sampled NaN. Until it does, the whole-value
// bind is what typeinfer models, so the mismatch surfaces as this located
// error. Pinned so the day the feed lands, this test fails and says so.
// Recorded in flatppl-dev/TODO-flatppl-js.md.
test('gap: a record variate into a lone step input errors rather than NaN', () => {
  const errs = errorsOf(`
flatppl_compat = "0.1"
M0 = joint(a = Normal(0, 1))
K1 = a -> Normal(mu = a, sigma = 1)
ch = kchain(M0, K1)
`);
  assert.equal(errs.length, 1, JSON.stringify(errs.map((d: any) => d.message)));
  assert.match(errs[0].message, /expects real, got record with fields a: real/);
});

test('no false positive: a 2-step chain still types and samples', async () => {
  const N = 20000;
  const ctx = setupCtx(`
flatppl_compat = "0.1"
K1 = a -> Normal(mu = a, sigma = 1)
ch = kchain(Normal(0, 1), K1)
`, N, 11);
  const m = await ctx.getMeasure('ch');
  const s: number[] = Array.from(m.value ? m.value.data : m.samples);
  assert.equal(s.filter((x) => Number.isNaN(x)).length, 0);
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  const varr = s.reduce((a, b) => a + (b - mean) ** 2, 0) / (s.length - 1);
  // a ~ N(0,1), b | a ~ N(a, 1) ⇒ var(b) = 2.
  assert.ok(Math.abs(varr - 2) < 0.15, `2-step marginal variance ${varr} ≈ 2`);
});
