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
// Model: theta ~ N(0,1); obs | theta ~ N(theta, 1).
// Joint marginals: theta ~ N(0,1); obs ~ N(0, sqrt(2)); corr = 1/sqrt(2).

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

const SRC_DISINTEGRATE = `
theta ~ Normal(0, 1)
obs ~ Normal(theta, 1)
joint_model = lawof(record(theta = theta, obs = obs))
forward_kernel, prior = disintegrate(["obs"], joint_model)
chained = jointchain(p = prior, o = forward_kernel)
marg = kchain(prior, forward_kernel)
`;

const SRC_KERNELOF = `
theta ~ Normal(0, 1)
obs ~ Normal(theta, 1)
fk = kernelof(record(obs = obs), theta = theta)
prior2 = lawof(record(theta = theta))
chained2 = jointchain(p = prior2, o = fk)
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
  // Spec §06 keyword form ≡ relabel(component, [label]): the labelled
  // single-field components are RENAMED to the chain labels (p, o).
  assert.deepStrictEqual(Object.keys(m.fields).sort(), ['o', 'p']);
  const st = stats(m.fields.p.samples);
  const so = stats(m.fields.o.samples);
  assert.ok(Math.abs(st.mean) < 0.1 && Math.abs(st.sd - 1) < 0.1,
    `theta marginal ~ N(0,1): mean ${st.mean.toFixed(3)} sd ${st.sd.toFixed(3)}`);
  assert.ok(Math.abs(so.mean) < 0.15 && Math.abs(so.sd - Math.SQRT2) < 0.15,
    `obs marginal ~ N(0,sqrt2): mean ${so.mean.toFixed(3)} sd ${so.sd.toFixed(3)}`);
  // The joint dependence must survive (env-threading of the kernel's
  // theta ref to the per-atom prior draw): corr = 1/sqrt(2).
  const c = corr(m.fields.p.samples, m.fields.o.samples);
  assert.ok(Math.abs(c - Math.SQRT1_2) < 0.08,
    `corr(p, o) ${c.toFixed(3)} ≈ 1/sqrt(2)`);
});

test('kchain over a disintegrate prior + kernel samples the marginal of the kernel variate', async () => {
  const ctx = buildCtx(SRC_DISINTEGRATE, 8000);
  assert.ok(ctx.derivations['marg'], 'kchain(disintegrate-prior, …) gets a derivation');
  const m = await ctx.getMeasure('marg');
  // Marginal of the record-bodied kernel's variate: record{obs}, prior
  // integrated out — obs ~ N(0, sqrt(2)).
  assert.strictEqual(m.shape, 'record');
  assert.deepStrictEqual(Object.keys(m.fields), ['obs']);
  const so = stats(m.fields.obs.samples);
  assert.ok(Math.abs(so.mean) < 0.15 && Math.abs(so.sd - Math.SQRT2) < 0.15,
    `kchain marginal ~ N(0,sqrt2): mean ${so.mean.toFixed(3)} sd ${so.sd.toFixed(3)}`);
});

test('jointchain over a hand-written kernelof(record(…)) kernel samples the joint law (H6(b) tail)', async () => {
  const ctx = buildCtx(SRC_KERNELOF, 8000);
  assert.ok(ctx.derivations['chained2'], 'jointchain(prior, kernelof(record(…))) gets a derivation');
  const m = await ctx.getMeasure('chained2');
  assert.strictEqual(m.shape, 'record');
  assert.deepStrictEqual(Object.keys(m.fields).sort(), ['o', 'p']);
  const so = stats(m.fields.o.samples);
  assert.ok(Math.abs(so.mean) < 0.15 && Math.abs(so.sd - Math.SQRT2) < 0.15,
    `obs marginal ~ N(0,sqrt2): mean ${so.mean.toFixed(3)} sd ${so.sd.toFixed(3)}`);
  const c = corr(m.fields.p.samples, m.fields.o.samples);
  assert.ok(Math.abs(c - Math.SQRT1_2) < 0.08,
    `corr(p, o) ${c.toFixed(3)} ≈ 1/sqrt(2)`);
});
