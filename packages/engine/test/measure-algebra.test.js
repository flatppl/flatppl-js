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
          type: 'sampleN',
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
        if (d.weightIR) {
          const refArrays = collectRefArrays(d.weightIR);
          const reply = worker.handle({
            type: 'evaluateN', ir: d.weightIR, count: sampleCount, refArrays,
          });
          if (reply.type === 'error') throw new Error(reply.message);
          const weights = reply.samples;
          if (d.isLog) {
            for (let i = 0; i < w.length; i++) w[i] = lifted.logWeights[i] + weights[i];
          } else {
            for (let i = 0; i < w.length; i++) {
              const v = weights[i];
              w[i] = (v > 0) ? lifted.logWeights[i] + Math.log(v) : -Infinity;
            }
          }
        } else {
          for (let i = 0; i < w.length; i++) w[i] = lifted.logWeights[i] + d.logShift;
        }
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
      case 'iid': {
        // iid(M, n, …): draw count*prod(dims) scalars from the inner
        // measure's distIR, layout atom-major (atom i occupies indices
        // [i*k, (i+1)*k) where k = prod(dims)). The worker's sampleN
        // takes a `repeat: k` shortcut that does this in one pass.
        const distIR = orchestrator.leafSampleIR(d.from, derivations);
        if (!distIR) throw new Error("iid: can't resolve leaf sample IR for " + d.from);
        const k = d.dims.reduce((p, n) => p * n, 1);
        const reply = worker.handle({
          type: 'sampleN', ir: distIR, count: sampleCount, repeat: k,
          seed: nameSeed(name, rootSeed),
        });
        if (reply.type === 'error') throw new Error(reply.message);
        m = empirical.arrayMeasure(reply.samples, d.dims, null);
        break;
      }
      case 'record': {
        // Multivariate: materialise each field's source binding,
        // assemble into a record-shaped EmpiricalMeasure (SoA).
        // logWeights at the top level is the join of all fields'
        // weights — we materialise each component (unifying their
        // weight arrays via materialiseUniform) and sum log-weights
        // index-aligned. Uniform components contribute -log(N) each.
        const fields = {};
        const componentArrays = [];
        for (const fname in d.fields) {
          const sub = go(d.fields[fname]);
          fields[fname] = sub;
          if (sub.logWeights) componentArrays.push(sub.logWeights);
        }
        let logWeights = null;
        if (componentArrays.length > 0) {
          const N = componentArrays[0].length;
          logWeights = new Float64Array(N);
          for (let i = 0; i < N; i++) {
            let s = 0;
            for (const arr of componentArrays) s += arr[i];
            logWeights[i] = s;
          }
        }
        m = empirical.recordMeasure(fields, logWeights);
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

// =====================================================================
// Function-of-variate weights: weighted(<expr>, m) and
// logweighted(<expr>, m) where the weight depends on per-atom values
// rather than being a constant. The orchestrator stores weightIR; the
// materialiser evaluates it per-i and adds the (log-)result to the
// parent's logWeights, atom-aligned through the shared sample axis.
// =====================================================================

test('per-atom weighted: log-weights track log(weight_i) atom-wise', () => {
  // theta is Exponential(1) — strictly positive, so log(theta_i) is
  // finite and varies per atom. weighted(theta, m) must therefore
  // produce non-uniform logWeights matching log(theta.samples) — not
  // their mean, not a single constant shift.
  const src = `
    theta_dist = Exponential(rate=1)
    theta = draw(theta_dist)
    m = Normal(mu=0, sigma=1)
    w = weighted(theta, m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const theta = materialise('theta', bindings, { cache });
  const W     = materialise('w',     bindings, { cache });
  // Same samples (alias chain through materialiseUniform).
  const M     = materialise('m',     bindings, { cache });
  assert.equal(W.samples, M.samples, 'weighted should share parent samples');
  // logWeights[i] = log(theta_i) + log(1/N).  Subtract the uniform
  // baseline and we should recover log(theta_i).
  const N = W.logWeights.length;
  const baseline = -Math.log(N);
  for (let i = 0; i < N; i++) {
    const expected = Math.log(theta.samples[i]);
    const got = W.logWeights[i] - baseline;
    assert.ok(Math.abs(got - expected) < 1e-12,
      `atom ${i}: log(theta=${theta.samples[i]}) expected ${expected}, got ${got}`);
  }
  // Sanity: per-atom variation, not a flat shift.
  let minLW = Infinity, maxLW = -Infinity;
  for (let i = 0; i < N; i++) {
    if (W.logWeights[i] < minLW) minLW = W.logWeights[i];
    if (W.logWeights[i] > maxLW) maxLW = W.logWeights[i];
  }
  assert.ok(maxLW - minLW > 0.5, 'expected meaningful per-atom variation');
});

test('per-atom logweighted: log-weights track lw_i directly (no log call)', () => {
  // logweighted(theta, m) where theta is on the log scale already —
  // logWeights[i] = theta_i + log(1/N), no log() applied.
  const src = `
    theta_dist = Normal(mu=0, sigma=1)
    theta = draw(theta_dist)
    m = Normal(mu=0, sigma=1)
    w = logweighted(theta, m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const theta = materialise('theta', bindings, { cache });
  const W     = materialise('w',     bindings, { cache });
  const N = W.logWeights.length;
  const baseline = -Math.log(N);
  for (let i = 0; i < N; i++) {
    const got = W.logWeights[i] - baseline;
    assert.ok(Math.abs(got - theta.samples[i]) < 1e-12,
      `atom ${i}: expected lw=${theta.samples[i]}, got ${got}`);
  }
});

test('per-atom weighted: arithmetic expressions in the weight slot', () => {
  // Weight is `2 * theta` — exercises the evaluable-expression path,
  // not just a bare ref. Confirms the orchestrator routes evaluateN
  // refs and the materialiser threads them through correctly.
  const src = `
    theta_dist = Exponential(rate=1)
    theta = draw(theta_dist)
    m = Normal(mu=0, sigma=1)
    w = weighted(2 * theta, m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const theta = materialise('theta', bindings, { cache });
  const W     = materialise('w',     bindings, { cache });
  const N = W.logWeights.length;
  const baseline = -Math.log(N);
  for (let i = 0; i < N; i++) {
    const expected = Math.log(2 * theta.samples[i]);
    const got = W.logWeights[i] - baseline;
    assert.ok(Math.abs(got - expected) < 1e-10,
      `atom ${i}: expected ${expected}, got ${got}`);
  }
});

test('per-atom weighted: zero-valued atoms become -Infinity (zero mass)', () => {
  // weighted(0, m) is a valid edge case: the atom carries no mass.
  // The materialiser turns log(0) into -Infinity rather than NaN.
  // Build a weight expression that produces zero at every atom.
  const src = `
    theta_dist = Normal(mu=0, sigma=1)
    theta = draw(theta_dist)
    m = Normal(mu=0, sigma=1)
    w = weighted(0 * theta, m)
  `;
  const { bindings } = processSource(src);
  const W = materialise('w', bindings);
  for (let i = 0; i < W.logWeights.length; i++) {
    assert.equal(W.logWeights[i], -Infinity, `atom ${i} should have -Inf log-weight`);
  }
  // Total mass = log(0) = -Infinity (the measure is the zero measure).
  assert.equal(empirical.totalLogMass(W), -Infinity);
});

test('per-atom weighted: inline draw(<measure-ref>) in the weight slot', () => {
  // weighted(draw(theta_dist), m) — an inline draw in the weight
  // slot. The orchestrator unwraps it to a ref to theta_dist (variates
  // and measures share samples in our cache), so the per-atom path
  // works just as if the user had written the named-variate version.
  const src = `
    theta_dist = Exponential(rate=1)
    m = Normal(mu=0, sigma=1)
    w_inline = weighted(draw(theta_dist), m)
    theta = draw(theta_dist)
    w_named  = weighted(theta, m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const inline = materialise('w_inline', bindings, { cache });
  const named  = materialise('w_named',  bindings, { cache });
  // The inline form aliases through theta_dist; the named form
  // aliases through theta which itself aliases to theta_dist. Both
  // end up reading theta_dist's samples → bit-equal logWeights.
  assertSameSamples(inline, named, 'inline-draw vs named-variate samples');
  assertSameLogWeights(inline, named, 1e-12, 'inline-draw vs named-variate logWeights');
});

test('per-atom weighted: inline draw nested inside an arithmetic expression', () => {
  // weighted(2 * draw(theta_dist), m) — exercises the recursive walk
  // in unwrapInlineDraws. After unwrap the IR is `2 * theta_dist`,
  // which is evaluable; logWeights should track log(2 * theta_i).
  const src = `
    theta_dist = Exponential(rate=1)
    m = Normal(mu=0, sigma=1)
    w = weighted(2 * draw(theta_dist), m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const theta_dist = materialise('theta_dist', bindings, { cache });
  const W          = materialise('w',          bindings, { cache });
  const N = W.logWeights.length;
  const baseline = -Math.log(N);
  for (let i = 0; i < N; i++) {
    const expected = Math.log(2 * theta_dist.samples[i]);
    const got = W.logWeights[i] - baseline;
    assert.ok(Math.abs(got - expected) < 1e-10,
      `atom ${i}: expected ${expected}, got ${got}`);
  }
});

test('lifting: deeply nested inline measure expression in draw composes correctly', () => {
  // theta4 = draw(weighted(draw(theta2_dist), theta1_dist))
  //
  // The lift pass produces:
  //   __anon_inner = draw(theta2_dist)    (alias to theta2_dist)
  //   __anon_outer = weighted(__anon_inner, theta1_dist)
  //   theta4       = draw(__anon_outer)   (alias to __anon_outer)
  //
  // Computationally identical to the user-named version
  //   tmp = draw(theta2_dist)
  //   theta4_dist = weighted(tmp, theta1_dist)
  //   theta4 = draw(theta4_dist)
  // — both end up reading theta1_dist's samples weighted by
  // log(theta2_dist.samples_i).
  const inlineSrc = `
    theta1_dist = Normal(mu=0, sigma=1)
    theta2_dist = Exponential(rate=1)
    theta4 = draw(weighted(draw(theta2_dist), theta1_dist))
  `;
  const namedSrc = `
    theta1_dist = Normal(mu=0, sigma=1)
    theta2_dist = Exponential(rate=1)
    tmp = draw(theta2_dist)
    theta4_dist = weighted(tmp, theta1_dist)
    theta4 = draw(theta4_dist)
  `;
  const inline = materialise('theta4', processSource(inlineSrc).bindings);
  const named  = materialise('theta4', processSource(namedSrc).bindings);
  assertSameSamples(inline, named, 'inline vs named samples');
  assertSameLogWeights(inline, named, 1e-12, 'inline vs named logWeights');
});

// =====================================================================
// User-defined functions: functionof / kernelof inlining
// =====================================================================
//
// Per spec §sec:functionof "functionof(f(a, b), a=a, b=b) ≡ f", a
// function call is semantically equivalent to its inlined body with
// parameter refs substituted. We assert this identity at the
// EmpiricalMeasure level: `a = f_a(par = beta1)` produces samples
// identical to `a = c * beta1` (the inlined body).

test('user-call inlining: f_a(par=beta1) is identical to the inlined body', () => {
  const inlinedSrc = `
    c = 2.5
    theta_dist = Normal(mu = 0, sigma = 1)
    theta = draw(theta_dist)
    beta1 = 2 * theta
    a = c * beta1
  `;
  const userCallSrc = `
    c = 2.5
    _par = elementof(reals)
    f_a = functionof(c * _par, par = _par)
    theta_dist = Normal(mu = 0, sigma = 1)
    theta = draw(theta_dist)
    beta1 = 2 * theta
    a = f_a(par = beta1)
  `;
  const inlined  = materialise('a', processSource(inlinedSrc).bindings);
  const userCall = materialise('a', processSource(userCallSrc).bindings);
  // Same per-binding seeding produces identical sample sequences when
  // the underlying computation matches.
  assertSameSamples(inlined, userCall, 'user-call vs inlined body samples');
});

test('user-call inlining: nested user calls compose', () => {
  // f composes with g via nested user call sites. Per the identity
  // law, the result is the same as inlining everything by hand.
  const handInlined = `
    theta = draw(Normal(mu = 0, sigma = 1))
    a = 2 * (theta + 1)
  `;
  const composed = `
    _x = elementof(reals)
    g = functionof(_x + 1, x = _x)
    _y = elementof(reals)
    f = functionof(2 * _y, y = _y)
    theta = draw(Normal(mu = 0, sigma = 1))
    a = f(y = g(x = theta))
  `;
  const inlined  = materialise('a', processSource(handInlined).bindings);
  const userCall = materialise('a', processSource(composed).bindings);
  assertSameSamples(inlined, userCall, 'composed user-calls vs hand-inlined samples');
});

test('user-call inlining: kernel application yields a measure', () => {
  // functionof(measure_expr, kw=...) produces a kernel per
  // §sec:functionof-measure. Applying it gives a measure derivation.
  // We sample from the kernel's body in parallel to a hand-inlined
  // version and check they match.
  const handInlined = `
    theta = draw(Normal(mu = 0, sigma = 1))
    obs = draw(Normal(mu = theta, sigma = 1))
  `;
  const kernelApply = `
    _t = elementof(reals)
    fwd = functionof(Normal(mu = _t, sigma = 1), theta = _t)
    theta = draw(Normal(mu = 0, sigma = 1))
    obs = draw(fwd(theta = theta))
  `;
  const inlined = materialise('obs', processSource(handInlined).bindings);
  const applied = materialise('obs', processSource(kernelApply).bindings);
  // Sample-level equivalence given matching seeds.
  assertSameSamples(inlined, applied, 'kernel-application vs inlined-draw samples');
});

// =====================================================================
// Multivariate (record-shaped) measures: SoA sample layout
// =====================================================================
//
// `joint(a=M_a, b=M_b)` and `record(x=v_x, y=v_y)` both produce
// record-shaped EmpiricalMeasures: per-field sub-measures keyed by
// the surface name, sharing one top-level logWeights. Marginals are
// just `m.fields.<name>` — no flattening, no projection.

test('joint: produces a record-shaped measure with per-field sub-measures', () => {
  const src = `
    a = Normal(mu = 0, sigma = 1)
    b = Exponential(rate = 1)
    j = joint(p = a, q = b)
  `;
  const { bindings } = processSource(src);
  const m = materialise('j', bindings);
  assert.equal(m.shape, 'record');
  assert.ok(m.fields.p, 'has field p');
  assert.ok(m.fields.q, 'has field q');
  // Each field is itself an EmpiricalMeasure with its own samples.
  assert.ok(m.fields.p.samples instanceof Float64Array);
  assert.ok(m.fields.q.samples instanceof Float64Array);
  assert.equal(m.fields.p.samples.length, m.fields.q.samples.length);
});

test('joint: marginal of field "p" matches direct materialisation of M_p', () => {
  // Materialising the joint and then projecting field p should
  // produce the same sample array as materialising M_p directly,
  // given matching seeds (same per-binding seeding via nameSeed).
  const src = `
    a = Normal(mu = 0, sigma = 1)
    b = Exponential(rate = 1)
    j = joint(p = a, q = b)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const direct = materialise('a', bindings, { cache });
  const j      = materialise('j', bindings, { cache });
  assert.equal(j.fields.p, direct, 'shared sub-measure object');
});

test('record: value-typed record with per-field variates is also record-shaped', () => {
  // record(x=variate, y=variate) — same SoA shape as joint, just
  // value-typed at the language level.
  const src = `
    a_dist = Normal(mu = 0, sigma = 1)
    b_dist = Exponential(rate = 1)
    a = draw(a_dist)
    b = draw(b_dist)
    r = record(x = a, y = b)
  `;
  const { bindings } = processSource(src);
  const m = materialise('r', bindings);
  assert.equal(m.shape, 'record');
  assert.equal(Object.keys(m.fields).length, 2);
  assert.ok(m.fields.x.samples instanceof Float64Array);
  assert.ok(m.fields.y.samples instanceof Float64Array);
});

test('record: empirical.shapeOf returns the right discriminator', () => {
  const src = `
    m = Normal(mu = 0, sigma = 1)
    r = joint(x = m)
  `;
  const { bindings } = processSource(src);
  const cache = new Map();
  const scalar = materialise('m', bindings, { cache });
  const record = materialise('r', bindings, { cache });
  // Untagged scalar measures default to 'scalar' for back-compat.
  assert.equal(empirical.shapeOf(scalar), 'scalar');
  assert.equal(empirical.shapeOf(record), 'record');
});

// =====================================================================
// iid: array-shaped sampling
// =====================================================================

test('iid: produces an array-shaped measure with N*k flat samples', () => {
  const src = `
    obs_dist = iid(Normal(mu = 0, sigma = 1), 10)
  `;
  const { bindings } = processSource(src);
  const m = materialise('obs_dist', bindings, { sampleCount: 256 });
  assert.equal(m.shape, 'array');
  assert.deepEqual(m.dims, [10]);
  // Atom-major: N atoms × k inner = 256 * 10 = 2560.
  assert.equal(m.samples.length, 2560);
});

test('iid: marginal distribution at each index matches the inner measure', () => {
  // Spec identity (informally): the value distribution of the j-th
  // slot across atoms is the same as the inner measure's value
  // distribution. Test by mean-and-variance comparison: for
  // Normal(2, 0.5), each iid slot's empirical (mean, sd) should
  // be near (2, 0.5).
  const src = `
    obs_dist = iid(Normal(mu = 2, sigma = 0.5), 5)
  `;
  const { bindings } = processSource(src);
  const m = materialise('obs_dist', bindings, { sampleCount: 8192 });
  // Marginal for slot j = atom-major samples at positions j, j+k, j+2k, ...
  const k = m.dims[0];
  for (let j = 0; j < k; j++) {
    let sum = 0, sumSq = 0;
    const N = m.samples.length / k;
    for (let i = 0; i < N; i++) {
      const v = m.samples[i * k + j];
      sum += v; sumSq += v * v;
    }
    const mean = sum / N;
    const sd = Math.sqrt(sumSq / N - mean * mean);
    assert.ok(Math.abs(mean - 2) < 0.05, `slot ${j} mean ${mean} not near 2`);
    assert.ok(Math.abs(sd   - 0.5) < 0.05, `slot ${j} sd ${sd} not near 0.5`);
  }
});

test('iid: nested inside a joint record produces a record with an array field', () => {
  // The `obs_dist = joint(obs = iid(Normal(...), 10))` pattern from
  // bayesian_inference_3.flatppl. Materialised result is a record
  // measure whose obs field is itself an array measure.
  const src = `
    obs_dist = joint(obs = iid(Normal(mu = 0, sigma = 1), 10))
  `;
  const { bindings } = processSource(src);
  const m = materialise('obs_dist', bindings, { sampleCount: 64 });
  assert.equal(m.shape, 'record');
  assert.ok(m.fields.obs);
  const obs = m.fields.obs;
  assert.equal(obs.shape, 'array');
  assert.deepEqual(obs.dims, [10]);
  assert.equal(obs.samples.length, 640);  // 64 atoms × 10 inner
});

test('iid: with a single-atom shape (n=1) is essentially the inner measure', () => {
  // iid(M, 1) is structurally an array<1, [1], ...> measure but
  // statistically equivalent to M. Confirms n=1 doesn't tickle a
  // special case in the worker's repeat path.
  const src = `
    m  = Normal(mu = 0, sigma = 1)
    m1 = iid(m, 1)
  `;
  const { bindings } = processSource(src);
  const direct = materialise('m',  bindings, { sampleCount: 128 });
  const single = materialise('m1', bindings, { sampleCount: 128 });
  assert.equal(single.shape, 'array');
  assert.deepEqual(single.dims, [1]);
  assert.equal(single.samples.length, 128);
  // Independent seeds (different binding names → different seeds)
  // so samples won't be identical, but the marginal stats agree.
  function mean(arr) { let s = 0; for (let i = 0; i < arr.length; i++) s += arr[i]; return s / arr.length; }
  assert.ok(Math.abs(mean(direct.samples) - mean(single.samples)) < 0.2);
});

test('orchestrator: weighted(<measure>, m) is rejected as a type error', () => {
  // Per spec §sec:measure-algebra the first argument of weighted
  // must be a value, not a measure. theta_dist IS a measure, so
  // this should not produce a derivation at all.
  const src = `
    theta_dist = Normal(mu=0, sigma=1)
    m = Normal(mu=0, sigma=1)
    w = weighted(theta_dist, m)
  `;
  const { bindings } = processSource(src);
  const { orchestrator } = require('..');
  const { derivations } = orchestrator.buildDerivations(bindings);
  assert.equal(derivations.w, undefined,
    'weighted(<measure>, m) should be rejected; got ' + JSON.stringify(derivations.w));
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
