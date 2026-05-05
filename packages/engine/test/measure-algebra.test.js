'use strict';

// Spec-identity tests for the main-thread measure-algebra pipeline.
//
// These tests exercise the same path the visualizer takes — parse →
// analyze → buildDerivations → recursively materialise into an
// EmpiricalMeasure — and assert spec-level identities at the
// EmpiricalMeasure level (samples + logWeights).
//
// The materialise() helper below mirrors visualPanel.js's getMeasure
// recursion in-process, swapping the worker postMessage roundtrip for
// a direct call to createWorkerHandler. That means:
//   - sample / evaluate run via the real worker handler (real stdlib
//     distributions, real Philox RNG)
//   - alias / weighted / normalize / superpose / array run via the
//     same logic visualPanel uses (mass-faithful, materialiseUniform,
//     systematicResample for superpose)
//
// What we assert (spec §sec:measure-algebra, §sec:additive-superposition,
// §sec:disintegrate):
//   - lawof(draw(m)) ≡ m                         identity law
//   - weighted(1, m) ≡ m                         no-op weighting
//   - weighted(a, weighted(b, m)) ≡ weighted(a*b, m)   composition
//   - normalize(weighted(c, m)) ≡ normalize(m)   scalar absorbed
//   - normalize(normalize(m)) ≡ normalize(m)     idempotence
//   - totalLogMass(superpose(m, m)) = log(2) + totalLogMass(m)   additivity
//   - normalize(superpose(M, M)) ≡ normalize(M)  statistical equivalence
//
// Equality at the empirical level is reference-level when aliases
// share an array; otherwise we check (samples bit-equal under the
// same per-binding seed) AND (logWeights elementwise close).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, empirical } = require('..');
const { createWorkerHandler } = require('../worker');

const SAMPLE_COUNT = 2048;

// Per-binding seed: same FNV-1a hash mix as visualPanel.nameSeed,
// so test results are deterministic and match what the extension
// would compute for the same source.
function nameSeed(name, rootSeed) {
  let h = 2166136261;
  for (let i = 0; i < name.length; i++) {
    h = h ^ name.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h ^ rootSeed) >>> 0;
}

// Deterministic main-thread PRNG for systematic resampling — wraps
// the engine's Philox in a U(0,1) callback. Mirrors visualPanel's
// makeMainThreadPrng but uses Math.random as a stable fallback if
// rng.nextUniform isn't accessible. We use rng so superpose draws are
// reproducible across test runs.
function makeMainThreadPrng(seed) {
  const rng = require('../rng');
  let state = rng.stateFromKey(seed);
  return () => {
    const pair = rng.nextUniform(state);
    state = pair[1];
    return pair[0];
  };
}

/**
 * Materialise an EmpiricalMeasure for `name` by walking the
 * derivation graph from `bindings`. Mirrors visualPanel.getMeasure
 * but runs synchronously in-process, using a single shared
 * worker-handler instance for sample / evaluate.
 *
 * Returns { samples, logWeights } — logWeights is null for unweighted
 * measures (the spec's "uniform 1/N" convention).
 *
 * @param {string} name
 * @param {Map}    bindings
 * @param {{ rootSeed?: number, sampleCount?: number, cache?: Map }} [opts]
 */
function materialise(name, bindings, opts) {
  opts = opts || {};
  const rootSeed     = opts.rootSeed    != null ? opts.rootSeed    : 12345;
  const sampleCount  = opts.sampleCount != null ? opts.sampleCount : SAMPLE_COUNT;
  const cache        = opts.cache       || new Map();
  const worker       = opts.worker      || createWorkerHandler();
  if (!opts.worker) worker.handle({ type: 'init', seed: rootSeed });

  const { derivations } = orchestrator.buildDerivations(bindings);
  return go(name);

  function go(name) {
    if (cache.has(name)) return cache.get(name);
    const d = derivations[name];
    if (!d) throw new Error(`no derivation for '${name}'`);

    let m;
    switch (d.kind) {
      case 'alias': {
        // Alias: same EmpiricalMeasure object as the parent. Reference
        // equality is the *point* — variates and their measures share.
        m = go(d.from);
        break;
      }
      case 'sample': {
        const refArrays = collectRefArrays(d.distIR);
        const reply = worker.handle({
          type: 'drawN',
          ir: d.distIR,
          count: sampleCount,
          refArrays,
          seed: nameSeed(name, rootSeed),
        });
        if (reply.type === 'error') throw new Error(reply.message);
        m = { samples: reply.samples, logWeights: reply.logWeights || null };
        break;
      }
      case 'evaluate': {
        const refArrays = collectRefArrays(d.ir);
        const reply = worker.handle({
          type: 'evaluateN',
          ir: d.ir,
          count: sampleCount,
          refArrays,
        });
        if (reply.type === 'error') throw new Error(reply.message);
        m = { samples: reply.samples, logWeights: reply.logWeights || null };
        break;
      }
      case 'array': {
        m = { samples: Float64Array.from(d.values), logWeights: null };
        break;
      }
      case 'weighted': {
        const parent = go(d.from);
        const lifted = empirical.materialiseUniform(parent);
        const w = new Float64Array(lifted.logWeights.length);
        for (let i = 0; i < w.length; i++) w[i] = lifted.logWeights[i] + d.logShift;
        m = { samples: lifted.samples, logWeights: w };
        break;
      }
      case 'normalize': {
        const parent = go(d.from);
        const lifted = empirical.materialiseUniform(parent);
        const lse = empirical.logSumExp(lifted.logWeights);
        const w = new Float64Array(lifted.logWeights.length);
        for (let i = 0; i < w.length; i++) w[i] = lifted.logWeights[i] - lse;
        m = { samples: lifted.samples, logWeights: w };
        break;
      }
      case 'superpose': {
        const parents = d.fromNames.map(go);
        let totalN = 0;
        for (const p of parents) totalN += p.samples.length;
        if (totalN === 0) { m = { samples: new Float64Array(0), logWeights: null }; break; }
        const combinedSamples    = new Float64Array(totalN);
        const combinedLogWeights = new Float64Array(totalN);
        let offset = 0;
        for (const p of parents) {
          const lifted = empirical.materialiseUniform(p);
          combinedSamples.set(lifted.samples, offset);
          combinedLogWeights.set(lifted.logWeights, offset);
          offset += lifted.samples.length;
        }
        const prng = makeMainThreadPrng(nameSeed(name, rootSeed));
        const idx = empirical.systematicResample(combinedLogWeights, sampleCount, prng);
        const out = new Float64Array(sampleCount);
        for (let i = 0; i < sampleCount; i++) out[i] = combinedSamples[idx[i]];
        const totalLogMass = empirical.logSumExp(combinedLogWeights);
        const perAtom = totalLogMass - Math.log(sampleCount);
        const outW = new Float64Array(sampleCount);
        outW.fill(perAtom);
        m = { samples: out, logWeights: outW };
        break;
      }
      default:
        throw new Error(`unsupported derivation kind '${d.kind}' in materialise()`);
    }
    cache.set(name, m);
    return m;
  }

  function collectRefArrays(ir) {
    const refs = orchestrator.collectSelfRefs(ir);
    const out = {};
    refs.forEach(n => { out[n] = go(n).samples; });
    return out;
  }
}

// =====================================================================
// Helpers for measure-equality assertions
// =====================================================================

function arraysClose(a, b, tol) {
  tol = tol == null ? 1e-12 : tol;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!(Math.abs(a[i] - b[i]) <= tol)) return false;
  }
  return true;
}

function assertSameSamples(a, b, msg) {
  assert.equal(a.samples.length, b.samples.length, (msg || '') + ' (samples length)');
  assert.ok(arraysClose(a.samples, b.samples, 0),
    (msg || '') + ' (samples not bit-equal)');
}

function assertSameLogWeights(a, b, tol, msg) {
  const la = a.logWeights, lb = b.logWeights;
  if (la == null && lb == null) return;
  // null = uniform: replace with explicit -log(N) for the comparison.
  const A = la == null ? empirical.materialiseUniform(a).logWeights : la;
  const B = lb == null ? empirical.materialiseUniform(b).logWeights : lb;
  assert.equal(A.length, B.length, (msg || '') + ' (logWeights length)');
  assert.ok(arraysClose(A, B, tol == null ? 1e-10 : tol),
    (msg || '') + ' (logWeights not close)');
}

// =====================================================================
// Identity tests
// =====================================================================

test('identity: lawof(draw(m)) ≡ m — both lawof and draw alias share the parent', () => {
  // Per spec: drawing from a measure and taking the law of that variate
  // returns the original measure. In our derivation graph this is
  // expressed as a chain of aliases all sharing one EmpiricalMeasure.
  const src = `
    m_dist = Normal(mu=0, sigma=1)
    x = draw(m_dist)
    m_again = lawof(x)
  `;
  const { bindings, diagnostics } = processSource(src);
  assert.equal(diagnostics.filter(d => d.severity === 'error').length, 0);
  const cache = new Map();
  const m       = materialise('m_dist',  bindings, { cache });
  const x       = materialise('x',       bindings, { cache });
  const mAgain  = materialise('m_again', bindings, { cache });
  // Reference equality: variate and law-of-variate share the SAME
  // EmpiricalMeasure object — no extra draws, no extra allocation.
  assert.equal(x,      m, 'draw(m) should alias m');
  assert.equal(mAgain, m, 'lawof(draw(m)) should alias m');
});

test('identity: weighted(1, m) ≡ m up to a uniform log-weight shift of 0', () => {
  // weighted(c, m) shifts every log-weight by log(c). For c=1 the
  // shift is 0, so the result has the same samples and (uniform-equiv)
  // weights as m.
  const src = `
    m = Normal(mu=0, sigma=1)
    w = weighted(1, m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const M = materialise('m', bindings, { cache });
  const W = materialise('w', bindings, { cache });
  assertSameSamples(M, W, 'weighted(1, m) samples');
  assertSameLogWeights(M, W, 1e-12, 'weighted(1, m) logWeights');
  // Total mass matches: log(1) = 0 added to log(N * 1/N) = 0 → 0.
  assert.ok(Math.abs(empirical.totalLogMass(W) - empirical.totalLogMass(M)) < 1e-10);
});

test('identity: weighted(a, weighted(b, m)) ≡ weighted(a*b, m) — log-shifts compose', () => {
  // Composition: nested constant reweights collapse to a single
  // shift of log(a) + log(b) = log(a*b). Samples stay identical, only
  // logWeights differ.
  const src = `
    m = Normal(mu=0, sigma=1)
    w_inner = weighted(2, m)
    w_outer = weighted(3, w_inner)
    w_combined = weighted(6, m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const lhs = materialise('w_outer',    bindings, { cache });
  const rhs = materialise('w_combined', bindings, { cache });
  assertSameSamples(lhs, rhs, 'composed weighted samples');
  assertSameLogWeights(lhs, rhs, 1e-10, 'composed weighted logWeights');
  // Total mass = log(6) (start at 0 for unit-mass m, add log 6).
  assert.ok(Math.abs(empirical.totalLogMass(lhs) - Math.log(6)) < 1e-10);
});

test('identity: normalize(weighted(c, m)) ≡ normalize(m) — scalar absorbed by normalisation', () => {
  // Multiplying every weight by a positive constant shifts logSumExp
  // by the same constant, so subtracting it back leaves the same
  // probability measure. Samples must match (alias chain) and final
  // logWeights must agree.
  const src = `
    m = Normal(mu=0, sigma=1)
    n_direct = normalize(m)
    w = weighted(7, m)
    n_via_weighted = normalize(w)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const a = materialise('n_direct',       bindings, { cache });
  const b = materialise('n_via_weighted', bindings, { cache });
  assertSameSamples(a, b, 'normalize(weighted) samples');
  assertSameLogWeights(a, b, 1e-10, 'normalize(weighted) logWeights');
  // totalLogMass = 0 (probability measure).
  assert.ok(Math.abs(empirical.totalLogMass(a)) < 1e-10);
});

test('identity: normalize(normalize(m)) ≡ normalize(m) — idempotence', () => {
  // After one normalize, total log-mass = 0; the second normalize
  // subtracts logSumExp = 0 and is a no-op modulo float noise.
  const src = `
    m = Normal(mu=0, sigma=1)
    n1 = normalize(m)
    n2 = normalize(n1)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const once  = materialise('n1', bindings, { cache });
  const twice = materialise('n2', bindings, { cache });
  assertSameSamples(once, twice, 'normalize idempotent samples');
  assertSameLogWeights(once, twice, 1e-10, 'normalize idempotent logWeights');
});

test('identity: totalLogMass(superpose(m, m)) = log(2) + totalLogMass(m) — additivity', () => {
  // Spec §sec:additive-superposition: superpose adds masses, never
  // rescales. Two copies of a unit-mass m give a measure of total
  // mass 2.
  const src = `
    m = Normal(mu=0, sigma=1)
    s = superpose(m, m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const M = materialise('m', bindings, { cache });
  const S = materialise('s', bindings, { cache });
  const mass_m = empirical.totalLogMass(M);
  const mass_s = empirical.totalLogMass(S);
  assert.ok(Math.abs(mass_s - (Math.log(2) + mass_m)) < 1e-10,
    `total mass: superpose=${mass_s}, expected=${Math.log(2) + mass_m}`);
});

test('identity: superpose with weighted summands tracks per-branch mass', () => {
  // superpose(weighted(2, m), weighted(3, m)) has total mass 2 + 3 = 5.
  // Confirms the mass-faithful behaviour of weighted *and* superpose
  // composes correctly.
  const src = `
    m = Normal(mu=0, sigma=1)
    a = weighted(2, m)
    b = weighted(3, m)
    s = superpose(a, b)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const S = materialise('s', bindings, { cache });
  // Total mass = log(5).
  assert.ok(Math.abs(empirical.totalLogMass(S) - Math.log(5)) < 1e-9,
    `expected log(5), got ${empirical.totalLogMass(S)}`);
});

test('identity: normalize(superpose(m, m)) ≡ normalize(m) — statistical equivalence', () => {
  // Two copies of m, normalised, is statistically the same probability
  // measure as m itself: the per-bin density of a histogram should
  // agree to within Monte-Carlo noise. We test this on 1-D Normal
  // samples by comparing means and standard deviations.
  const src = `
    m = Normal(mu=2, sigma=0.5)
    s = superpose(m, m)
    pn = normalize(m)
    ps = normalize(s)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const PN = materialise('pn', bindings, { cache, sampleCount: 8192 });
  const PS = materialise('ps', bindings, { cache, sampleCount: 8192 });
  // Both are now probability measures (totalLogMass ≈ 0).
  assert.ok(Math.abs(empirical.totalLogMass(PN)) < 1e-10);
  assert.ok(Math.abs(empirical.totalLogMass(PS)) < 1e-10);
  // Compare weighted means and SDs — they should match the underlying
  // distribution (mu=2, sigma=0.5) up to Monte-Carlo noise.
  function weightedMeanSd(meas) {
    const lifted = empirical.materialiseUniform(meas);
    let totW = 0, mean = 0, m2 = 0;
    for (let i = 0; i < lifted.samples.length; i++) {
      const w = Math.exp(lifted.logWeights[i]);
      const x = lifted.samples[i];
      const newW = totW + w;
      const delta = x - mean;
      mean += (w / newW) * delta;
      m2 += w * delta * (x - mean);
      totW = newW;
    }
    return { mean, sd: Math.sqrt(m2 / totW) };
  }
  const a = weightedMeanSd(PN);
  const b = weightedMeanSd(PS);
  // Both should be near (mu=2, sigma=0.5) within 3*SE; cross-equality
  // is the spec claim — check directly.
  assert.ok(Math.abs(a.mean - b.mean) < 0.05, `means differ: ${a.mean} vs ${b.mean}`);
  assert.ok(Math.abs(a.sd   - b.sd)   < 0.05, `sds differ: ${a.sd} vs ${b.sd}`);
});

test('identity: weighted preserves base samples reference (no extra draws)', () => {
  // weighted should be implemented as a *re-weighting* of the parent's
  // sample array, not a fresh draw. Asserting reference equality of
  // the .samples Float64Array makes this contract testable.
  const src = `
    m = Normal(mu=0, sigma=1)
    w = weighted(2, m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const M = materialise('m', bindings, { cache });
  const W = materialise('w', bindings, { cache });
  // After materialiseUniform inside the weighted handler, samples
  // ref is preserved (materialiseUniform only allocates logWeights).
  assert.equal(W.samples, M.samples, 'weighted should share parent samples');
  // logWeights is fresh, of course.
  assert.notEqual(W.logWeights, M.logWeights);
});
