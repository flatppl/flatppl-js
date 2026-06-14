'use strict';

// PoissonProcess end-to-end (spec §08; engine-concepts §2.3) — the ragged
// per-atom point-set distribution. Pins the materialiser pipeline
// (classifier → matPoissonProcess → ragged EmpiricalMeasure) and the MVP
// boundary rejections. The load-bearing sampling assembly + density MATH is
// pinned separately, oracle-against-Julia, in mat-poisson.test.ts; here we
// check the wiring and the sampling STATISTICS (counts ~ Poisson(M), points
// ~ shape) plus the structural invariants of the ragged measure.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const R = require('../ragged.ts');
const density = require('../density.ts');

const lit = (v: any) => ({ kind: 'lit', value: v });
const ref = (name: any) => ({ kind: 'ref', ns: 'self', name });
const Normal = (mu: any, sigma: any) =>
  ({ kind: 'call', op: 'Normal', kwargs: { mu, sigma } });
const Exponential = (rate: any) =>
  ({ kind: 'call', op: 'Exponential', kwargs: { rate } });
const weighted = (M: any, shape: any) =>
  ({ kind: 'call', op: 'weighted', args: [M, shape] });
const superpose = (...comps: any[]) =>
  ({ kind: 'call', op: 'superpose', args: comps });
const PP = (intensity: any) =>
  ({ kind: 'call', op: 'PoissonProcess', kwargs: { intensity } });
const normlogpdf = (t: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * t * t;

const ROOT_SEED = 0xCAFEBEEF;

function makeCtx(source: any, sampleCount: any) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: sampleCount || 4096,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

// =====================================================================
// Classification + ragged measure structure
// =====================================================================

test('PoissonProcess: classifier recognises and materialises a ragged measure', async () => {
  const ctx = makeCtx(`
shape = Normal(mu = 0.0, sigma = 1.0)
m ~ PoissonProcess(intensity = weighted(5.0, shape))
`, 256);
  const mm = await ctx.getMeasure('m');
  assert.equal(mm.shape, 'ragged');
  assert.ok(R.isRagged(mm.ragged), 'carries a ragged value');
  assert.equal(R.raggedCount(mm.ragged), 256, 'one atom per sample');
  assert.equal(mm.samples, mm.ragged.data, '.samples aliases the pooled flat points');
  assert.deepEqual(mm.dims, [], 'scalar points → empty kernelShape');
  // offsets frame the data: monotone, run 0 … data.length.
  const off = mm.ragged.offsets;
  assert.equal(off[0], 0);
  assert.equal(off[off.length - 1], mm.ragged.data.length);
  for (let i = 1; i < off.length; i++) assert.ok(off[i] >= off[i - 1]);
});

test('PoissonProcess: inline shape (no intervening binding) also works', async () => {
  const ctx = makeCtx(`
m ~ PoissonProcess(intensity = weighted(3.0, Normal(mu = 10.0, sigma = 2.0)))
`, 256);
  const mm = await ctx.getMeasure('m');
  assert.equal(mm.shape, 'ragged');
  assert.equal(R.raggedCount(mm.ragged), 256);
});

// =====================================================================
// Sampling statistics: counts ~ Poisson(M), points ~ shape
// =====================================================================

test('PoissonProcess: per-atom counts ~ Poisson(M), pooled points ~ shape', async () => {
  const M = 5.0;
  const ctx = makeCtx(`
shape = Normal(mu = 2.0, sigma = 3.0)
m ~ PoissonProcess(intensity = weighted(${M}, shape))
`, 40000);
  const mm = await ctx.getMeasure('m');
  const N = R.raggedCount(mm.ragged);
  // Mean count ≈ M (Poisson mean). Variance of the mean ≈ M/N ⇒ tol generous.
  const meanCount = mm.ragged.data.length / N;
  assert.ok(Math.abs(meanCount - M) < 0.1, 'meanCount=' + meanCount);
  // Pooled points ≈ shape moments (mu=2, sigma=3).
  let sum = 0, sum2 = 0;
  const d = mm.ragged.data;
  for (let i = 0; i < d.length; i++) { sum += d[i]; sum2 += d[i] * d[i]; }
  const mean = sum / d.length;
  const sd = Math.sqrt(sum2 / d.length - mean * mean);
  assert.ok(Math.abs(mean - 2.0) < 0.1, 'pooled mean=' + mean);
  assert.ok(Math.abs(sd - 3.0) < 0.1, 'pooled sd=' + sd);
});

test('PoissonProcess: M = 0 ⇒ every atom is empty (all-zero offsets)', async () => {
  const ctx = makeCtx(`
m ~ PoissonProcess(intensity = weighted(0.0, Normal(mu = 0.0, sigma = 1.0)))
`, 64);
  const mm = await ctx.getMeasure('m');
  assert.equal(mm.shape, 'ragged');
  assert.equal(R.raggedCount(mm.ragged), 64);
  assert.equal(mm.ragged.data.length, 0, 'no points');
  for (let i = 0; i < mm.ragged.offsets.length; i++) {
    assert.equal(mm.ragged.offsets[i], 0);
  }
});

// =====================================================================
// MVP boundary — documented rejections (red-for-the-right-reason)
// =====================================================================

test('PoissonProcess: a superpose intensity materialises (superposition union)', async () => {
  // λ = 3·Normal(0,1) + 2·Exponential(rate=0.5). M = 5; mixture mean =
  // (3·0 + 2·2)/5 = 0.8 (Exp(rate=0.5) mean = 2).
  const ctx = makeCtx(`
sig = Normal(mu = 0.0, sigma = 1.0)
bkg = Exponential(rate = 0.5)
m ~ PoissonProcess(intensity = superpose(weighted(3.0, sig), weighted(2.0, bkg)))
`, 40000);
  const mm = await ctx.getMeasure('m');
  assert.equal(mm.shape, 'ragged');
  const N = R.raggedCount(mm.ragged);
  const meanCount = mm.ragged.data.length / N;
  assert.ok(Math.abs(meanCount - 5.0) < 0.1, 'meanCount=' + meanCount);
  let sum = 0;
  for (let i = 0; i < mm.ragged.data.length; i++) sum += mm.ragged.data[i];
  const mean = sum / mm.ragged.data.length;
  assert.ok(Math.abs(mean - 0.8) < 0.1, 'pooled mixture mean=' + mean);
});

test('PoissonProcess: a draw-dependent (per-atom) shape param is refused for sampling', async () => {
  const ctx = makeCtx(`
raw ~ Normal(mu = 0.0, sigma = 1.0)
res = 2.5 + 0.3 * raw
m ~ PoissonProcess(intensity = weighted(5.0, Normal(mu = 0.0, sigma = res)))
`, 64);
  await assert.rejects(() => ctx.getMeasure('m'), /per-atom|cannot resolve/);
});

// =====================================================================
// Density walker (walkPoissonProcess) — Julia-oracle-pinned (the same
// reference as mat-poisson.test.ts: spec density (∏λ(tᵢ))·exp(−M)).
// =====================================================================

test('density: weighted(5, Normal(0,1)) at points [0,1] = Julia oracle −4.119001', () => {
  const ir = PP(weighted(lit(5.0), Normal(lit(0.0), lit(1.0))));
  const logp = density.logDensity(ir, [0.0, 1.0], {});
  assert.ok(Math.abs(logp - (-4.119001)) < 1e-5, 'got ' + logp);
});

test('density: empty observation scores −M (no points)', () => {
  const ir = PP(weighted(lit(4.0), Normal(lit(0.0), lit(1.0))));
  const logp = density.logDensity(ir, [], {});
  assert.ok(Math.abs(logp - (-4.0)) < 1e-12, 'got ' + logp);
});

test('density: logweighted intensity form (expandMeasure-expanded) scores identically', () => {
  // A constant-weight binding expands to logweighted(log M, shape); M = exp(logM).
  const ir = PP({ kind: 'call', op: 'logweighted',
                  args: [lit(Math.log(5.0)), Normal(lit(0.0), lit(1.0))] });
  const logp = density.logDensity(ir, [0.0, 1.0], {});
  assert.ok(Math.abs(logp - (-4.119001)) < 1e-5, 'got ' + logp);
});

test('density: per-atom M (M as a θ-ref) scores each atom with its own count rate', () => {
  // M is a per-atom ref; one shared observed point set [0,1].
  const ir = PP(weighted(ref('n'), Normal(lit(0.0), lit(1.0))));
  const out = density.logDensityN(ir, [0.0, 1.0],
    { n: Float64Array.from([5.0, 4.0]) }, 2, {});
  const S = normlogpdf(0.0) + normlogpdf(1.0);
  const a0 = 2 * Math.log(5.0) - 5.0 + S;   // = −4.119001 (the oracle)
  const a1 = 2 * Math.log(4.0) - 4.0 + S;
  assert.ok(Math.abs(out[0] - a0) < 1e-6, 'atom0 ' + out[0]);
  assert.ok(Math.abs(out[1] - a1) < 1e-6, 'atom1 ' + out[1]);
  assert.ok(Math.abs(out[0] - (-4.119001)) < 1e-5);
});

test('density: per-atom SHAPE param (the spec flagship pattern) resolves per θ', () => {
  // shape sigma varies per atom — the case generative sampling defers, but
  // the density/likelihood path handles. atom0 σ=1 ⇒ the −4.119001 oracle.
  const ir = PP(weighted(lit(5.0), Normal(lit(0.0), ref('sg'))));
  const out = density.logDensityN(ir, [0.0, 1.0],
    { sg: Float64Array.from([1.0, 2.0]) }, 2, {});
  const nlp = (t: number, s: number) =>
    -Math.log(s) - 0.5 * Math.log(2 * Math.PI) - 0.5 * (t / s) * (t / s);
  const a0 = 2 * Math.log(5.0) - 5.0 + (nlp(0, 1) + nlp(1, 1));
  const a1 = 2 * Math.log(5.0) - 5.0 + (nlp(0, 2) + nlp(1, 2));
  assert.ok(Math.abs(out[0] - a0) < 1e-6, 'atom0 ' + out[0]);
  assert.ok(Math.abs(out[1] - a1) < 1e-6, 'atom1 ' + out[1]);
  assert.ok(Math.abs(out[0] - (-4.119001)) < 1e-5, 'atom0 = oracle');
});

// =====================================================================
// General intensity — superpose density (Julia-oracle-pinned; closed-form
// λ(t)=3·Normal(0,1)+2·Exp(rate=0.5), logp = Σ log λ(t_j) − M, M=5).
// =====================================================================

test('density superpose: @ [0.5,1.5] = Julia oracle −4.54271174', () => {
  const ir = PP(superpose(weighted(lit(3.0), Normal(lit(0.0), lit(1.0))),
                          weighted(lit(2.0), Exponential(lit(0.5)))));
  const logp = density.logDensity(ir, [0.5, 1.5], {});
  assert.ok(Math.abs(logp - (-4.54271174)) < 1e-6, 'got ' + logp);
});

test('density superpose: empty obs scores −M = −5', () => {
  const ir = PP(superpose(weighted(lit(3.0), Normal(lit(0.0), lit(1.0))),
                          weighted(lit(2.0), Exponential(lit(0.5)))));
  assert.ok(Math.abs(density.logDensity(ir, [], {}) - (-5.0)) < 1e-12);
});

test('density superpose: @ [0,1,2] = Julia oracle −4.56112897', () => {
  const ir = PP(superpose(weighted(lit(3.0), Normal(lit(0.0), lit(1.0))),
                          weighted(lit(2.0), Exponential(lit(0.5)))));
  const logp = density.logDensity(ir, [0.0, 1.0, 2.0], {});
  assert.ok(Math.abs(logp - (-4.56112897)) < 1e-6, 'got ' + logp);
});

test('density superpose: one-component superpose ≡ the single weighted form', () => {
  const a = density.logDensity(PP(superpose(weighted(lit(5.0), Normal(lit(0.0), lit(1.0))))),
    [0.0, 1.0], {});
  const b = density.logDensity(PP(weighted(lit(5.0), Normal(lit(0.0), lit(1.0)))),
    [0.0, 1.0], {});
  assert.ok(Math.abs(a - b) < 1e-12 && Math.abs(a - (-4.119001)) < 1e-5);
});

// =====================================================================
// End-to-end: from source through matLogdensityof (the production path
// the worker / viewer use), scoring a PoissonProcess measure against an
// observed point set — reaching the same Julia oracle.
// =====================================================================

test('e2e: matLogdensityof(events @ observed) through the full pipeline = −4.119001', () => {
  const { matLogdensityof } = require('../mat-density.ts');
  const lifted = processSource(`
shape = Normal(mu = 0.0, sigma = 1.0)
events ~ PoissonProcess(intensity = weighted(5.0, shape))
observed = [0.0, 1.0]
`);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: 1, rootKey: ROOT_SEED,
    // The density path expands the intensity (so the shape leaf is inline) and
    // must NOT re-materialise any measure binding — assert no getMeasure leak.
    getMeasure: (name: any) => Promise.reject(new Error('getMeasure leaked for ' + name)),
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
  };
  const d = { kind: 'logdensityof', measureName: 'events',
    obsIR: { kind: 'ref', ns: 'self', name: 'observed' } };
  return matLogdensityof(d as any, ctx).then((m: any) => {
    assert.ok(Math.abs(m.samples[0] - (-4.119001)) < 1e-5, 'got ' + m.samples[0]);
  });
});

test('e2e: the spec-flagship superpose (separate intensity binding → select path) = −4.54271174', () => {
  // intensity bound separately ⇒ a ref ⇒ expandMeasure renders it as `select`
  // (the by-name superpose→select rewrite). Exercises that density path e2e.
  const { matLogdensityof } = require('../mat-density.ts');
  const lifted = processSource(`
sig = Normal(mu = 0.0, sigma = 1.0)
bkg = Exponential(rate = 0.5)
intensity = superpose(weighted(3.0, sig), weighted(2.0, bkg))
events ~ PoissonProcess(intensity = intensity)
observed = [0.5, 1.5]
`);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: 1, rootKey: ROOT_SEED,
    getMeasure: (name: any) => Promise.reject(new Error('getMeasure leaked for ' + name)),
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
  };
  const d = { kind: 'logdensityof', measureName: 'events',
    obsIR: { kind: 'ref', ns: 'self', name: 'observed' } };
  return matLogdensityof(d as any, ctx).then((m: any) => {
    assert.ok(Math.abs(m.samples[0] - (-4.54271174)) < 1e-6, 'got ' + m.samples[0]);
  });
});
