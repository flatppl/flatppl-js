'use strict';

// jointchain / kchain over REIFIED kernels and disintegrate-produced priors
// (audit Phase 9 + the H6(b) tail).
//
// Two coupled gaps closed together:
//   1. `lawof` was missing from MEASURE_PRODUCING (builtins.ts), so a
//      disintegrate-prior — whose destructured binding's IR head is `lawof` —
//      failed classifyJointchain's measure check and the whole
//      `jointchain(<disintegrate-prior>, …)` got NO derivation.
//   2. expandMeasure's jointchain `kernelExpand` only expanded REF bodies; an
//      inline `lawof(record(obs = <draw ref>))` body (what kernelof /
//      disintegrate reify) shipped verbatim with bare draw refs, dead-ending
//      the walkers (legacy: "'lawof' is not a known distribution"; CLM:
//      "non-call IR (kind 'ref')"). Now the inline body is expanded via
//      _expandStructural — identity on already self-contained bodies.
//
// Model: theta ~ N(0,1); phi ~ N(0,1); obs | theta, phi ~ N(theta + phi, 1). The kernel has two
// inputs, so the prior's record variate splats onto them by field name (§04 "Auto-splatting").
// Joint marginals: theta ~ N(0,1); obs ~ N(0, sqrt(3)); corr(theta, obs) = 1/sqrt(3).

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

const SRC_DISINTEGRATE = `
theta ~ Normal(0, 1)
phi ~ Normal(0, 1)
obs ~ Normal(theta + phi, 1)
joint_model = lawof(record(theta = theta, phi = phi, obs = obs))
forward_kernel, prior = disintegrate(["obs"], joint_model)
chained = jointchain(prior, forward_kernel)
marg = kchain(prior, forward_kernel)
`;

const SRC_KERNELOF = `
theta ~ Normal(0, 1)
phi ~ Normal(0, 1)
obs ~ Normal(theta + phi, 1)
fk = kernelof(record(obs = obs), theta = theta, phi = phi)
prior2 = lawof(record(theta = theta, phi = phi))
chained2 = jointchain(prior2, fk)
`;

function buildCtx(src: string, N: number) {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 11 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: 11, rootSeed: 11, marginalizationCount: 32,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

function stats(f: any) {
  let s = 0, s2 = 0; const n = f.length;
  for (let i = 0; i < n; i++) { s += f[i]; s2 += f[i] * f[i]; }
  const mean = s / n;
  return { mean, sd: Math.sqrt(s2 / n - mean * mean) };
}

function corr(a: any, b: any) {
  const sa = stats(a), sb = stats(b);
  let sab = 0; const n = a.length;
  for (let i = 0; i < n; i++) sab += a[i] * b[i];
  return (sab / n - sa.mean * sb.mean) / (sa.sd * sb.sd);
}

test('jointchain over a disintegrate prior + kernel classifies and samples the joint law', async () => {
  const ctx = buildCtx(SRC_DISINTEGRATE, 8000);
  // The Phase 9 gap: this derivation used to be silently absent.
  assert.ok(ctx.derivations['chained'], 'jointchain(disintegrate-prior, …) gets a derivation');
  assert.strictEqual(ctx.derivations['chained'].kind, 'jointchain');

  const m = await ctx.getMeasure('chained');
  assert.strictEqual(m.shape, 'record');
  // Positional jointchain: the variate is the merged record of the prior and the kernel.
  assert.deepStrictEqual(Object.keys(m.fields).sort(), ['obs', 'phi', 'theta']);
  const st = stats(m.fields.theta.samples);
  const so = stats(m.fields.obs.samples);
  assert.ok(Math.abs(st.mean) < 0.1 && Math.abs(st.sd - 1) < 0.1,
    `theta marginal ~ N(0,1): mean ${st.mean.toFixed(3)} sd ${st.sd.toFixed(3)}`);
  assert.ok(Math.abs(so.mean) < 0.15 && Math.abs(so.sd - Math.sqrt(3)) < 0.15,
    `obs marginal ~ N(0,sqrt3): mean ${so.mean.toFixed(3)} sd ${so.sd.toFixed(3)}`);
  // The joint dependence must survive (env-threading of the kernel's
  // theta ref to the per-atom prior draw): corr = 1/sqrt(2).
  const c = corr(m.fields.theta.samples, m.fields.obs.samples);
  assert.ok(Math.abs(c - 1 / Math.sqrt(3)) < 0.08,
    `corr(theta, obs) ${c.toFixed(3)} ≈ 1/sqrt(3)`);
});

test('kchain over a disintegrate prior + kernel samples the marginal of the kernel variate', async () => {
  const ctx = buildCtx(SRC_DISINTEGRATE, 8000);
  assert.ok(ctx.derivations['marg'], 'kchain(disintegrate-prior, …) gets a derivation');
  const m = await ctx.getMeasure('marg');
  // Marginal of the record-bodied kernel's variate: record{obs}, prior
  // integrated out — obs ~ N(0, sqrt(3)).
  assert.strictEqual(m.shape, 'record');
  assert.deepStrictEqual(Object.keys(m.fields), ['obs']);
  const so = stats(m.fields.obs.samples);
  assert.ok(Math.abs(so.mean) < 0.15 && Math.abs(so.sd - Math.sqrt(3)) < 0.15,
    `kchain marginal ~ N(0,sqrt3): mean ${so.mean.toFixed(3)} sd ${so.sd.toFixed(3)}`);
});

test('jointchain over a hand-written kernelof(record(…)) kernel samples the joint law (H6(b) tail)', async () => {
  const ctx = buildCtx(SRC_KERNELOF, 8000);
  assert.ok(ctx.derivations['chained2'], 'jointchain(prior, kernelof(record(…))) gets a derivation');
  const m = await ctx.getMeasure('chained2');
  assert.strictEqual(m.shape, 'record');
  assert.deepStrictEqual(Object.keys(m.fields).sort(), ['obs', 'phi', 'theta']);
  const so = stats(m.fields.obs.samples);
  assert.ok(Math.abs(so.mean) < 0.15 && Math.abs(so.sd - Math.sqrt(3)) < 0.15,
    `obs marginal ~ N(0,sqrt3): mean ${so.mean.toFixed(3)} sd ${so.sd.toFixed(3)}`);
  const c = corr(m.fields.theta.samples, m.fields.obs.samples);
  assert.ok(Math.abs(c - 1 / Math.sqrt(3)) < 0.08,
    `corr(theta, obs) ${c.toFixed(3)} ≈ 1/sqrt(3)`);
});

// One-field complement. Spec §06 `disintegrate`: "For a record-shaped complement, `kernel` takes its
// fields as keyword-only inputs", so §04 "Auto-splatting" splats the prior's record variate into the
// kernel even though it has a single input. A labelled chain nests the variate under its label
// (`record(p = record(theta = ...))`), which names no `theta`, so it is a located static error.
const SRC_ONE_FIELD = `
theta ~ Normal(0, 1)
obs ~ Normal(theta, 1)
joint_model = lawof(record(theta = theta, obs = obs))
forward_kernel, prior = disintegrate(["obs"], joint_model)
chained = jointchain(prior, forward_kernel)
marg = kchain(prior, forward_kernel)
lp = logdensityof(chained, record(theta = 0.3, obs = 0.5))
`;

test('a one-field record prior splats into the keyword-only disintegrate kernel', async () => {
  const ctx = buildCtx(SRC_ONE_FIELD, 8000);
  assert.deepStrictEqual(processSource(SRC_ONE_FIELD).diagnostics.filter((d: any) => d.severity === 'error'), []);
  // Closed form: logN(0.3; 0, 1) + logN(0.5; 0.3, 1).
  const logN = (x: number, m: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - m) ** 2;
  const lp = (await ctx.getMeasure('lp')).samples[0];
  assert.ok(Math.abs(lp - (logN(0.3, 0) + logN(0.5, 0.3))) < 1e-12, `density ${lp}`);
  const m = await ctx.getMeasure('chained');
  assert.deepStrictEqual(Object.keys(m.fields).sort(), ['obs', 'theta']);
  const so = stats(m.fields.obs.samples);
  assert.ok(Math.abs(so.mean) < 0.15 && Math.abs(so.sd - Math.SQRT2) < 0.15,
    `obs marginal ~ N(0,sqrt2): mean ${so.mean.toFixed(3)} sd ${so.sd.toFixed(3)}`);
  const c = corr(m.fields.theta.samples, m.fields.obs.samples);
  assert.ok(Math.abs(c - Math.SQRT1_2) < 0.08, `corr(theta, obs) ${c.toFixed(3)} ≈ 1/sqrt(2)`);
  const marg = await ctx.getMeasure('marg');
  const sm = stats(marg.fields.obs.samples);
  assert.ok(Math.abs(sm.mean) < 0.15 && Math.abs(sm.sd - Math.SQRT2) < 0.15,
    `kchain marginal ~ N(0,sqrt2): mean ${sm.mean.toFixed(3)} sd ${sm.sd.toFixed(3)}`);
});

test('a labelled chain nests the one-field variate, so the keyword-only kernel does not name it', () => {
  const errs = processSource(SRC_ONE_FIELD.replace('chained = jointchain(prior, forward_kernel)',
    'chained = jointchain(p = prior, o = forward_kernel)').replace(/lp = .*\n/, '')).diagnostics
    .filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1, JSON.stringify(errs.map((e: any) => e.message)));
  assert.match(errs[0].message, /record with fields p: record with fields theta: real/);
  assert.equal(errs[0].loc.start.line, 5);
});

test('an explicit one-input kernel takes the prior record whole, a located static error', () => {
  const src = `
theta ~ Normal(0, 1)
obs ~ Normal(theta, 1)
fk = kernelof(record(obs = obs), theta = theta)
prior2 = lawof(record(theta = theta))
chained2 = jointchain(prior2, fk)
`;
  const errs = processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1, JSON.stringify(errs.map((e: any) => e.message)));
  assert.match(errs[0].message, /record with fields theta: real/);
  assert.equal(errs[0].loc.start.line, 5);
});

// Spec §06 `joint` of kernels: "the result only supports keyword arguments", so a sole record splats
// into it, however many inputs it has, and a positional argument is refused.
test('the joint of kernels is keyword-only: a sole record splats, a positional argument is an error', async () => {
  const MODEL = `
t = elementof(reals)
K1 = functionof(joint(a = Normal(t, 1)), theta = t)
K2 = functionof(joint(b = Normal(2 * t, 0.5)), theta = t)
Kj = joint(K1, K2)
`;
  const src = MODEL + 'M = Kj(record(theta = 0.3))\nlp = logdensityof(M, record(a = 0.1, b = 0.9))\n';
  assert.deepStrictEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error'), []);
  // Closed form: logN(0.1; 0.3, 1) + logN(0.9; 0.6, 0.5).
  const logN = (x: number, m: number, s: number) =>
    -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
  const lp = (await buildCtx(src, 1).getMeasure('lp')).samples[0];
  assert.ok(Math.abs(lp - (logN(0.1, 0.3, 1) + logN(0.9, 0.6, 0.5))) < 1e-12, `density ${lp}`);
  const errs = processSource(MODEL + 'N = Kj(0.3)\n').diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /call to "Kj" does not accept 1 positional arguments/);
});

// A `joint` of kernels is a jointchain step: one input fans out to every component, so the prior's
// record variate splats into the union of the components' inputs by name (spec §06 `joint`).
const JOINT_STEP = (inputs: 1 | 2) => `
t = elementof(reals)
u = elementof(reals)
K1 = functionof(joint(a = Normal(t, 1)), theta = t)
K2 = functionof(joint(b = ${inputs === 1 ? 'Normal(2 * t, 0.5)' : 'Normal(u, 0.5)'}), ${inputs === 1 ? 'theta = t' : 'phi = u'})
Kj = joint(K1, K2)
th = draw(Normal(0, 1))
ph = draw(Normal(0, 1))
prior = lawof(record(${inputs === 1 ? 'theta = th' : 'theta = th, phi = ph'}))
chained = jointchain(prior, Kj)
marg = kchain(prior, Kj)
lp = logdensityof(chained, record(${inputs === 1 ? 'theta = 0.3' : 'theta = 0.3, phi = 0.2'}, a = 0.1, b = 0.9))
`;

for (const inputs of [1, 2] as const) {
  test(`a joint of kernels as a jointchain step scores its closed form (${inputs === 1 ? 'one shared input' : 'two inputs'})`, async () => {
    const logN = (x: number, m: number, s = 1) =>
      -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
    const src = JOINT_STEP(inputs);
    assert.deepStrictEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error'), []);
    const ctx = buildCtx(src, 20000);
    const want = inputs === 1
      ? logN(0.3, 0) + logN(0.1, 0.3) + logN(0.9, 0.6, 0.5)
      : logN(0.3, 0) + logN(0.2, 0) + logN(0.1, 0.3) + logN(0.9, 0.2, 0.5);
    const lp = (await ctx.getMeasure('lp')).samples[0];
    assert.ok(Math.abs(lp - want) < 1e-12, `density ${lp} vs ${want}`);
    // Sampling: a | theta ~ N(theta, 1), so Var(a) = 2 and corr(theta, a) = 1/sqrt(2).
    const m = await ctx.getMeasure('chained');
    assert.deepStrictEqual(Object.keys(m.fields).sort(), inputs === 1 ? ['a', 'b', 'theta'] : ['a', 'b', 'phi', 'theta']);
    const sa = stats(m.fields.a.samples);
    assert.ok(Math.abs(sa.mean) < 0.1 && Math.abs(sa.sd - Math.SQRT2) < 0.1, `a mean ${sa.mean} sd ${sa.sd}`);
    assert.ok(Math.abs(corr(m.fields.theta.samples, m.fields.a.samples) - Math.SQRT1_2) < 0.05);
    // b | theta ~ N(2 theta, 0.5) (one input) or N(phi, 0.5) (two): Var(b) = 4.25 or 1.25.
    const sb = stats(m.fields.b.samples);
    const sdB = Math.sqrt(inputs === 1 ? 4.25 : 1.25);
    assert.ok(Math.abs(sb.sd - sdB) < 0.12, `b sd ${sb.sd} vs ${sdB}`);
    // The kchain marginal keeps only the step's variates.
    const k = await ctx.getMeasure('marg');
    assert.deepStrictEqual(Object.keys(k.fields).sort(), ['a', 'b']);
  });
}

test('a measure component of a joint of kernels ignores the input', async () => {
  const logN = (x: number, m: number, s = 1) =>
    -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
  const src = `
t = elementof(reals)
K1 = functionof(joint(a = Normal(t, 1)), theta = t)
M = joint(c = Normal(0, 2))
Kj = joint(K1, M)
prior = lawof(record(theta = draw(Normal(0, 1))))
chained = jointchain(prior, Kj)
lp = logdensityof(chained, record(theta = 0.3, a = 0.1, c = 0.7))
`;
  assert.deepStrictEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error'), []);
  const lp = (await buildCtx(src, 1).getMeasure('lp')).samples[0];
  const want = logN(0.3, 0) + logN(0.1, 0.3) + logN(0.7, 0, 2);
  assert.ok(Math.abs(lp - want) < 1e-12, `density ${lp} vs ${want}`);
});
