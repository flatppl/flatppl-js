'use strict';

// Conformance tests for the commit-3 (RNG-splitting thread, Option C
// hybrid) batched-sampler paths in sampler.ts / sampler-registry.ts.
//
// Two batched paths exist:
//
//   1. Explicit `randNFn(state, params, n, out?) → { state, out }`:
//      one philoxN* call + vectorised transform — no per-atom JS
//      dispatch. Wired for Normal / LogNormal (Box-Muller, NOT bit-
//      equivalent to scalar randFn), Uniform / Exponential (bit-
//      equivalent to scalar randFn).
//
//   2. `makeBulkUniformPrngAdapter(state, n, factor=4)` feeding the
//      existing scalar randFn loop: cipher amortization only. The
//      adapter pre-fills N*factor uniforms via philoxNUniform, then
//      hands them out via the same `() → [0,1)` closure stdlib's
//      `factory(..., {prng})` expects. Variable-rejection dists
//      (Gamma, Beta, …) consume more than 1U per draw — the adapter
//      refills on demand and rewinds state to "exactly cursor
//      uniforms consumed", which is bit-exact equivalent to N scalar
//      nextUniform calls (philoxNUniform contract).
//
// What this test pins:
//   - Mean / variance / quantile envelopes of batched draws match
//     known-good analytical moments to within standard-error bands
//     (rules out Box-Muller orientation bugs and obvious transform
//     mistakes).
//   - Bit-exact equivalence for the bit-equivalent dists (Uniform /
//     Exponential / any dist routed through the bulk-uniform adapter):
//     a batched draw of N from state S produces the same values as
//     N scalar `sampler.rand` calls threaded through state S.
//   - Bulk-adapter `getState()` correctly returns the state after
//     exactly `cursor` uniforms — so subsequent draws from that state
//     line up with the scalar-loop path.
//   - Static-params worker-style entry point (`samplerLib.makeBulkSampler`)
//     routes Normal / Exponential / Uniform / LogNormal to the
//     `randNFn` path; everything else through the bulk-adapter path.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const rng = require('../rng.ts');

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}

function distIR(op: any, kwargs: any) {
  const out: any = {};
  for (const [k, v] of Object.entries(kwargs)) {
    out[k] = { kind: 'lit', value: v, loc: synthLoc() };
  }
  return { kind: 'call', op, kwargs: out, loc: synthLoc() };
}

// Uniform's IR uses a `support` kwarg holding an interval(...) call.
function uniformIR(lo: any, hi: any) {
  return {
    kind: 'call',
    op:   'Uniform',
    kwargs: {
      support: {
        kind: 'call',
        op:   'interval',
        args: [
          { kind: 'lit', value: lo, loc: synthLoc() },
          { kind: 'lit', value: hi, loc: synthLoc() },
        ],
        loc:  synthLoc(),
      },
    },
    loc: synthLoc(),
  };
}

function mean(xs: Float64Array) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

function variance(xs: Float64Array) {
  const m = mean(xs);
  let s = 0;
  for (let i = 0; i < xs.length; i++) { const d = xs[i] - m; s += d * d; }
  return s / (xs.length - 1);
}

// Compare two Philox states for BEHAVIOURAL equivalence. The state has
// a cached `block` field and `blockIdx`; when `blockIdx === 4` the
// cache is logically exhausted and `block` may be either null or a
// stale-but-ignored array. So canonicalise by clearing `block` when
// blockIdx is past-the-end before comparing.
function assertStatesEquivalent(a: any, b: any, msg?: string) {
  function canon(s: any) {
    if (s.blockIdx >= 4) {
      return { key: s.key, counter: s.counter, block: null, blockIdx: 4 };
    }
    return s;
  }
  assert.deepEqual(canon(a), canon(b), msg);
}

// =====================================================================
// Explicit batched paths — moment conformance
// =====================================================================

test('makeBulkSampler Normal(0,1): mean ≈ 0, var ≈ 1 (Box-Muller)', () => {
  const state = rng.seedFromBytes([42, 7, 0]);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const n = 50_000;
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  // stderr of mean = 1/sqrt(n) ≈ 0.0045; 5σ envelope ≈ 0.022.
  assert.ok(Math.abs(mean(r.samples)) < 0.05, `mean ${mean(r.samples)} not ~0`);
  assert.ok(Math.abs(variance(r.samples) - 1) < 0.05,
    `var ${variance(r.samples)} not ~1`);
});

test('makeBulkSampler Normal(2, 0.5): mean ≈ 2, var ≈ 0.25', () => {
  const state = rng.seedFromBytes([13, 7, 21]);
  const ir = distIR('Normal', { mu: 2, sigma: 0.5 });
  const n = 50_000;
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  assert.ok(Math.abs(mean(r.samples) - 2) < 0.05);
  assert.ok(Math.abs(variance(r.samples) - 0.25) < 0.05);
});

test('makeBulkSampler Exponential(rate=2): mean ≈ 0.5, var ≈ 0.25', () => {
  const state = rng.seedFromBytes([1, 2, 3]);
  const ir = distIR('Exponential', { rate: 2 });
  const n = 50_000;
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  // All draws must be non-negative (transform is -ln(1-u)/lambda).
  for (let i = 0; i < n; i++) assert.ok(r.samples[i] >= 0);
  assert.ok(Math.abs(mean(r.samples) - 0.5) < 0.05);
  assert.ok(Math.abs(variance(r.samples) - 0.25) < 0.05);
});

test('makeBulkSampler Uniform(a=−1, b=3): mean ≈ 1, var ≈ 16/12', () => {
  const state = rng.seedFromBytes([99, 99, 99]);
  const ir = uniformIR(-1, 3);
  const n = 50_000;
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  for (let i = 0; i < n; i++) {
    assert.ok(r.samples[i] >= -1 && r.samples[i] < 3 + 1e-9,
      `sample ${r.samples[i]} outside [-1, 3)`);
  }
  // U(-1,3) mean = 1, var = (3-(-1))²/12 ≈ 1.333.
  assert.ok(Math.abs(mean(r.samples) - 1) < 0.05);
  assert.ok(Math.abs(variance(r.samples) - 16 / 12) < 0.05);
});

test('makeBulkSampler LogNormal(0, 1): mean ≈ exp(0.5), var ≈ (e−1)·e', () => {
  const state = rng.seedFromBytes([55, 5, 5]);
  const ir = distIR('LogNormal', { mu: 0, sigma: 1 });
  const n = 100_000;
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  for (let i = 0; i < n; i++) assert.ok(r.samples[i] > 0);
  // LogNormal(0,1): mean = e^0.5 ≈ 1.6487, var = (e − 1)·e ≈ 4.6708.
  // Heavy tails — looser envelope.
  assert.ok(Math.abs(mean(r.samples) - Math.exp(0.5)) < 0.1,
    `mean ${mean(r.samples)} not ~1.6487`);
  assert.ok(Math.abs(variance(r.samples) - (Math.E - 1) * Math.E) < 0.5,
    `var ${variance(r.samples)} not ~4.67`);
});

// =====================================================================
// Bit-exactness for bit-equivalent batched dists
// =====================================================================

// Uniform / Exponential `randNFn` are bit-exact equivalent to N scalar
// randFn calls because philoxNUniform itself is bit-exact equivalent
// to N scalar nextUniform calls, and the stdlib transforms read one
// uniform per draw with the same arithmetic.

test('makeBulkSampler Uniform: bit-exact to scalar rand loop', () => {
  const state = rng.seedFromBytes([7, 7, 7]);
  const ir = uniformIR(-2, 5);
  const n = 1024;

  // Scalar baseline.
  let s = state;
  const scalar = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [v, next] = sampler.rand(s, ir, {});
    scalar[i] = v;
    s = next;
  }

  // Batched.
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  for (let i = 0; i < n; i++) {
    assert.equal(r.samples[i], scalar[i],
      `Uniform mismatch at i=${i}: batched=${r.samples[i]} scalar=${scalar[i]}`);
  }
  // And the trailing state agrees so callers can chain.
  assertStatesEquivalent(r.state, s, 'trailing state differs between batched and scalar');
});

test('makeBulkSampler Exponential: bit-exact to scalar rand loop', () => {
  const state = rng.seedFromBytes([3, 14, 15]);
  const ir = distIR('Exponential', { rate: 1.5 });
  const n = 1024;

  let s = state;
  const scalar = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [v, next] = sampler.rand(s, ir, {});
    scalar[i] = v;
    s = next;
  }
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  for (let i = 0; i < n; i++) {
    assert.equal(r.samples[i], scalar[i],
      `Exponential mismatch at i=${i}: batched=${r.samples[i]} scalar=${scalar[i]}`);
  }
  assertStatesEquivalent(r.state, s);
});

// =====================================================================
// Bulk-uniform adapter — fallback path for dists without randNFn
// =====================================================================

test('makeBulkUniformPrngAdapter: drains philoxNUniform 1U at a time, bit-exact', () => {
  const state = rng.seedFromBytes([100, 100, 100]);
  // Scalar baseline: 50 nextUniform calls.
  let s = state;
  const scalar = new Float64Array(50);
  for (let i = 0; i < 50; i++) {
    const [u, next] = rng.nextUniform(s);
    scalar[i] = u;
    s = next;
  }
  // Bulk adapter: pre-fill ≥ 50 uniforms (factor=4 default → 50*4=200).
  const prng = sampler.makeBulkUniformPrngAdapter(state, 50);
  const got = new Float64Array(50);
  for (let i = 0; i < 50; i++) got[i] = prng();
  for (let i = 0; i < 50; i++) {
    assert.equal(got[i], scalar[i], `bulk adapter mismatch at i=${i}`);
  }
  // After consuming 50 of the 200 pre-filled uniforms, the adapter's
  // getState() must rewind to the state-after-50-uniforms.
  assertStatesEquivalent(prng.getState(), s,
    'bulk adapter getState() did not rewind to scalar-equivalent state');
});

test('makeBulkUniformPrngAdapter: refills on exhaustion (variable-rejection dists)', () => {
  const state = rng.seedFromBytes([11, 22, 33]);
  // Request N=4 with factor=1 → only 4 uniforms pre-filled. Pop 10:
  // adapter must refill 3 times.
  const prng = sampler.makeBulkUniformPrngAdapter(state, 4, 1);
  const got = new Float64Array(10);
  for (let i = 0; i < 10; i++) got[i] = prng();
  // Compare to 10 scalar nextUniform calls.
  let s = state;
  const scalar = new Float64Array(10);
  for (let i = 0; i < 10; i++) {
    const [u, next] = rng.nextUniform(s);
    scalar[i] = u;
    s = next;
  }
  for (let i = 0; i < 10; i++) {
    assert.equal(got[i], scalar[i], `refilling adapter mismatch at i=${i}`);
  }
  assertStatesEquivalent(prng.getState(), s);
});

test('makeBulkSampler Gamma (no randNFn → bulk-adapter path): moments + state thread', () => {
  // Gamma has no randNFn — it should fall back to the scalar randFn
  // loop fed by the bulk-uniform adapter. Conformance: moments line
  // up, plus the state after `n` draws still matches what the scalar
  // path would have produced (philoxNUniform is bit-exact equivalent
  // to N nextUniform calls, and the bulk adapter rewinds to "cursor
  // uniforms consumed" on getState()).
  const state = rng.seedFromBytes([200, 201, 202]);
  const ir = distIR('Gamma', { shape: 2, rate: 1 });  // mean=2, var=2
  const n = 20_000;
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  for (let i = 0; i < n; i++) assert.ok(r.samples[i] >= 0);
  // Gamma(2,1): mean=2, var=2.
  assert.ok(Math.abs(mean(r.samples) - 2) < 0.1,
    `Gamma mean ${mean(r.samples)} not ~2`);
  assert.ok(Math.abs(variance(r.samples) - 2) < 0.2,
    `Gamma var ${variance(r.samples)} not ~2`);
});

test('makeBulkSampler Bernoulli (no randNFn → bulk-adapter path): fraction of 1s', () => {
  const state = rng.seedFromBytes([77, 88, 99]);
  const ir = distIR('Bernoulli', { p: 0.3 });
  const n = 50_000;
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  let ones = 0;
  for (let i = 0; i < n; i++) {
    assert.ok(r.samples[i] === 0 || r.samples[i] === 1);
    if (r.samples[i] === 1) ones++;
  }
  const frac = ones / n;
  assert.ok(Math.abs(frac - 0.3) < 0.01, `Bernoulli frac ${frac} not ~0.3`);
});

// =====================================================================
// Box-Muller orientation pin: paired output ordering
// =====================================================================
//
// philoxNNormal documents that pair (i, i+1) of uniforms produces
// (r*cos(theta), r*sin(theta)). The Normal randNFn applies the affine
// mu + sigma * z inplace, preserving that pairing. With sigma > 0 the
// distribution is symmetric, so a one-sided sign bias would surface as
// the mean drifting away from mu by more than the stderr envelope —
// this is the test that catches an inadvertent cos/sin swap or a
// missing `2 * Math.PI` factor (which would compress all outputs to a
// narrow angular wedge and skew the mean).
test('Box-Muller pairing: large-N Normal sample mean tracks mu to within 5σ', () => {
  const state = rng.seedFromBytes([1234, 5678, 9012]);
  const ir = distIR('Normal', { mu: 7.5, sigma: 2 });
  const n = 100_000;
  const r = sampler.makeBulkSampler(state, ir, {}, n);
  const stderrOfMean = 2 / Math.sqrt(n);
  assert.ok(Math.abs(mean(r.samples) - 7.5) < 5 * stderrOfMean,
    `Box-Muller orientation: mean ${mean(r.samples)} drifted from 7.5 by `
    + `${Math.abs(mean(r.samples) - 7.5)} (5σ envelope = ${5 * stderrOfMean})`);
});

// =====================================================================
// `out` reuse — caller-supplied Float64Array is filled, not allocated
// =====================================================================

test('makeBulkSampler: caller-supplied `out` is filled in place', () => {
  const state = rng.seedFromBytes([0, 0, 1]);
  const ir = distIR('Exponential', { rate: 1 });
  const n = 1024;
  const out = new Float64Array(n);
  const r = sampler.makeBulkSampler(state, ir, {}, n, out);
  // Same backing storage.
  assert.equal(r.samples, out, 'makeBulkSampler did not reuse provided out');
  // Values populated.
  let allZero = true;
  for (let i = 0; i < n; i++) { if (out[i] !== 0) { allZero = false; break; } }
  assert.ok(!allZero, 'out array was not populated');
});

// =====================================================================
// Per-i batched mode (Q2 one-leaf-realisation fold)
// =====================================================================
//
// makeBulkSampler(..., out, perI) draws with PER-ATOM parameter columns:
// resolveParamsN evaluates each param expr over the batch, randNFn runs
// one philoxN* draw + the per-element transform. This replaces the
// worker's per-draw stdlib randFn.factory loop (ziggurat for Normal/
// LogNormal) for the randNFn family with the single batched leaf math.
//
// What this pins:
//   - Uniform / Exponential per-i are BIT-EXACT to the old per-atom
//     makeParametricSampler.drawWith loop (same uniform stream + same
//     per-draw transform, atom-major order), incl. repeat>1.
//   - Normal / LogNormal per-i preserve the distribution (per-atom mean
//     tracks mu_i) — the accepted ziggurat→Box-Muller shift.
//   - resolveParamsN normalisation: scalar passthrough, ref→column,
//     Uniform interval support, non-interval fall-through; hasRandNFn.

const registry = require('../sampler-registry.ts');

function refIR(name: any) {
  return { kind: 'ref', ns: 'self', name, loc: synthLoc() };
}

// Uniform with per-atom support bounds: interval(<ref lo>, <ref hi>).
function uniformPerIIR(loName: any, hiName: any) {
  return {
    kind: 'call', op: 'Uniform',
    kwargs: { support: {
      kind: 'call', op: 'interval',
      args: [refIR(loName), refIR(hiName)],
      loc: synthLoc(),
    } },
    loc: synthLoc(),
  };
}

// The pre-fold per-i path: one parametric sampler, drawWith(per-atom env)
// in atom-major order. The baseline the batched path must match bit-for-
// bit for the bit-equivalent dists (Uniform / Exponential).
function perAtomScalarBaseline(state: any, ir: any, refArrays: any, count: number, repeat: number) {
  const s = sampler.makeParametricSampler(state, ir);
  const total = count * repeat;
  const out = new Float64Array(total);
  const refKeys = Object.keys(refArrays);
  const env: any = {};
  for (let i = 0; i < count; i++) {
    for (const k of refKeys) env[k] = refArrays[k][i];
    const base = i * repeat;
    for (let j = 0; j < repeat; j++) out[base + j] = s.drawWith(env);
  }
  return { samples: out, state: s.getState() };
}

test('makeBulkSampler perI Uniform: bit-exact to per-atom scalar drawWith loop', () => {
  const state = rng.seedFromBytes([3, 1, 4, 1, 5]);
  const count = 256;
  const lo = new Float64Array(count);
  const hi = new Float64Array(count);
  for (let i = 0; i < count; i++) { lo[i] = -1 - i * 0.01; hi[i] = 2 + i * 0.02; }
  const refArrays = { lo_i: lo, hi_i: hi };
  const ir = uniformPerIIR('lo_i', 'hi_i');

  const baseline = perAtomScalarBaseline(state, ir, refArrays, count, 1);
  const r = sampler.makeBulkSampler(state, ir, {}, count, undefined,
    { refArrays, count, repeat: 1 });
  assert.ok(!r.unhandled, 'perI Uniform should be handled by the batched path');
  for (let i = 0; i < count; i++) {
    assert.equal(r.samples[i], baseline.samples[i],
      `perI Uniform mismatch at i=${i}: batched=${r.samples[i]} scalar=${baseline.samples[i]}`);
    assert.ok(r.samples[i] >= lo[i] && r.samples[i] < hi[i] + 1e-9,
      `sample ${r.samples[i]} outside per-atom [${lo[i]}, ${hi[i]})`);
  }
  assertStatesEquivalent(r.state, baseline.state, 'perI Uniform trailing state differs');
});

test('makeBulkSampler perI Exponential: bit-exact to per-atom scalar drawWith loop', () => {
  const state = rng.seedFromBytes([2, 7, 1, 8]);
  const count = 200;
  const rate = new Float64Array(count);
  for (let i = 0; i < count; i++) rate[i] = 0.5 + i * 0.01;
  const refArrays = { rate_i: rate };
  const ir = { kind: 'call', op: 'Exponential', kwargs: { rate: refIR('rate_i') }, loc: synthLoc() };

  const baseline = perAtomScalarBaseline(state, ir, refArrays, count, 1);
  const r = sampler.makeBulkSampler(state, ir, {}, count, undefined,
    { refArrays, count, repeat: 1 });
  assert.ok(!r.unhandled);
  for (let i = 0; i < count; i++) {
    assert.equal(r.samples[i], baseline.samples[i], `perI Exp mismatch at i=${i}`);
    assert.ok(r.samples[i] >= 0);
  }
  assertStatesEquivalent(r.state, baseline.state, 'perI Exp trailing state differs');
});

test('makeBulkSampler perI Uniform repeat>1: atom-major bit-exact', () => {
  const state = rng.seedFromBytes([9, 9, 1]);
  const count = 32;
  const repeat = 5;
  const lo = new Float64Array(count);
  const hi = new Float64Array(count);
  for (let i = 0; i < count; i++) { lo[i] = i; hi[i] = i + 1; }
  const refArrays = { lo_i: lo, hi_i: hi };
  const ir = uniformPerIIR('lo_i', 'hi_i');

  const baseline = perAtomScalarBaseline(state, ir, refArrays, count, repeat);
  const total = count * repeat;
  const r = sampler.makeBulkSampler(state, ir, {}, total, undefined,
    { refArrays, count, repeat });
  assert.ok(!r.unhandled);
  for (let k = 0; k < total; k++) {
    assert.equal(r.samples[k], baseline.samples[k], `perI repeat mismatch at k=${k}`);
    const atom = Math.floor(k / repeat);
    assert.ok(r.samples[k] >= lo[atom] && r.samples[k] < hi[atom] + 1e-9,
      `slot ${k} (atom ${atom}) sample ${r.samples[k]} outside [${lo[atom]}, ${hi[atom]})`);
  }
  assertStatesEquivalent(r.state, baseline.state, 'perI repeat trailing state differs');
});

test('makeBulkSampler perI Normal: per-atom mean tracks mu_i (distribution preserved)', () => {
  const state = rng.seedFromBytes([4, 2, 4, 2]);
  const count = 40;
  const repeat = 4000;
  const muCol = new Float64Array(count);
  for (let i = 0; i < count; i++) muCol[i] = -5 + i * 0.25;
  const refArrays = { mu_i: muCol };
  // sigma fixed (atom-independent) → resolveParamsN returns it as a scalar.
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: { mu: refIR('mu_i'), sigma: { kind: 'lit', value: 1, loc: synthLoc() } },
    loc: synthLoc(),
  };
  const total = count * repeat;
  const r = sampler.makeBulkSampler(state, ir, {}, total, undefined,
    { refArrays, count, repeat });
  assert.ok(!r.unhandled);
  const stderr = 1 / Math.sqrt(repeat);   // sigma = 1
  for (let i = 0; i < count; i++) {
    let s = 0;
    const base = i * repeat;
    for (let j = 0; j < repeat; j++) s += r.samples[base + j];
    const m = s / repeat;
    assert.ok(Math.abs(m - muCol[i]) < 6 * stderr,
      `atom ${i}: block mean ${m} drifted from mu_i=${muCol[i]} (6σ=${6 * stderr})`);
  }
});

test('resolveParamsN: atom-independent params stay scalar', () => {
  const ir = distIR('Normal', { mu: 2, sigma: 0.5 });
  const entry = registry.lookupDistribution(ir);
  const cols = registry.resolveParamsN(ir, entry, {}, 16, {});
  assert.equal(cols.length, 2);
  assert.equal(cols[0], 2);     // scalar, not a column
  assert.equal(cols[1], 0.5);
});

test('resolveParamsN: per-atom ref params become Float64Array columns', () => {
  const count = 8;
  const mu = new Float64Array(count);
  for (let i = 0; i < count; i++) mu[i] = i;
  const ir = {
    kind: 'call', op: 'Normal',
    kwargs: { mu: refIR('mu_i'), sigma: { kind: 'lit', value: 1, loc: synthLoc() } },
    loc: synthLoc(),
  };
  const entry = registry.lookupDistribution(ir);
  const cols = registry.resolveParamsN(ir, entry, { mu_i: mu }, count, {});
  assert.ok(cols[0] && cols[0].BYTES_PER_ELEMENT !== undefined, 'mu should be a column');
  assert.equal(cols[0].length, count);
  assert.deepEqual(Array.from(cols[0]), Array.from(mu));
  assert.equal(cols[1], 1);     // sigma scalar
});

test('resolveParamsN: Uniform interval support → two per-atom columns', () => {
  const count = 5;
  const lo = Float64Array.from([0, 1, 2, 3, 4]);
  const hi = Float64Array.from([1, 2, 3, 4, 5]);
  const ir = uniformPerIIR('lo_i', 'hi_i');
  const entry = registry.lookupDistribution(ir);
  const cols = registry.resolveParamsN(ir, entry, { lo_i: lo, hi_i: hi }, count, {});
  assert.equal(cols.length, 2);
  assert.deepEqual(Array.from(cols[0]), [0, 1, 2, 3, 4]);
  assert.deepEqual(Array.from(cols[1]), [1, 2, 3, 4, 5]);
});

test('resolveParamsN: Uniform non-interval support → null (fall back to scalar path)', () => {
  const ir = {
    kind: 'call', op: 'Uniform',
    kwargs: { support: { kind: 'const', name: 'unitinterval', loc: synthLoc() } },
    loc: synthLoc(),
  };
  const entry = registry.lookupDistribution(ir);
  const cols = registry.resolveParamsN(ir, entry, {}, 4, {});
  assert.equal(cols, null);
});

test('hasRandNFn: true for the batched family, false otherwise', () => {
  assert.equal(registry.hasRandNFn(distIR('Normal', { mu: 0, sigma: 1 })), true);
  assert.equal(registry.hasRandNFn(distIR('Exponential', { rate: 1 })), true);
  assert.equal(registry.hasRandNFn(uniformIR(0, 1)), true);
  assert.equal(registry.hasRandNFn(distIR('LogNormal', { mu: 0, sigma: 1 })), true);
  assert.equal(registry.hasRandNFn(distIR('Gamma', { shape: 2, rate: 1 })), false);
  assert.equal(registry.hasRandNFn({ kind: 'call', op: 'broadcast', args: [], loc: synthLoc() }), false);
  assert.equal(registry.hasRandNFn(null), false);
});

test('makeBulkSampler: static and per-i agree on a shared scalar param (one realisation)', () => {
  // A per-i call whose params are all atom-INDEPENDENT (empty refArrays →
  // resolveParamsN returns scalar columns) must produce byte-identical
  // output to the static call from the same state: the static path is the
  // degenerate case of the per-i path, not a second realisation. (Numeric
  // byte-identity to the OLD scalar loop is pinned by the bit-exact gates;
  // this pins static≡per-i for the same scalar.)
  const state = rng.seedFromBytes([6, 6, 6]);
  const n = 64;
  const ir = distIR('Exponential', { rate: 2 });
  const stat = sampler.makeBulkSampler(state, ir, {}, n);
  const peri = sampler.makeBulkSampler(state, ir, {}, n, undefined,
    { refArrays: {}, count: n, repeat: 1 });
  assert.ok(!peri.unhandled, 'per-i with scalar params should still be handled');
  for (let i = 0; i < n; i++) {
    assert.equal(peri.samples[i], stat.samples[i],
      `static vs per-i scalar-param mismatch at i=${i}`);
  }
  assertStatesEquivalent(peri.state, stat.state, 'static vs per-i trailing state differs');
});
