'use strict';

// Tests for engine/rng.js — the Philox-4x32-10 implementation.
//
// Three layers of coverage, in order from most foundational:
//
//   1. mulhilo32 — the 32×32→64 multiply primitive. Tested against a
//      BigInt reference implementation across many random pairs, plus
//      hand-checked edge cases. If this is wrong, everything else is.
//
//   2. philox4x32_10 — the 10-round cipher. Tested against the canonical
//      Known-Answer-Test (KAT) vectors from the Random123 distribution.
//      These same vectors are what cuRAND, NumPy, PyTorch, TF agree on,
//      so passing them gives us cross-implementation reproducibility for
//      free.
//
//   3. State-threaded API — nextUint32 / nextUniform / seedFromBytes /
//      incrementCounter. Tests determinism, immutability, counter
//      rollover, block-boundary refill, uniform range.
//
// The Random123 KAT vectors are the gold standard for Philox correctness.
// Source: github.com/DEShawResearch/random123 (BSD-3-Clause), `kat_vectors`
// file. All three test triples below match published reference outputs.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const {
  philox4x32_10,
  seedFromBytes,
  stateFromKey,
  nextUint32,
  nextUniform,
  incrementCounter,
  _internal: { mulhilo32 },
} = require('../rng');

// =====================================================================
// 1. mulhilo32
// =====================================================================

// BigInt reference: slow but unambiguously correct. Used to check
// mulhilo32 across many random inputs.
function mulhilo32_bigint(a, b) {
  const product = BigInt(a >>> 0) * BigInt(b >>> 0);
  return [
    Number((product >> 32n) & 0xffffffffn),
    Number(product & 0xffffffffn),
  ];
}

test('mulhilo32: zero × anything = 0', () => {
  assert.deepEqual(mulhilo32(0, 0),         [0, 0]);
  assert.deepEqual(mulhilo32(0, 0xFFFFFFFF), [0, 0]);
  assert.deepEqual(mulhilo32(0xD2511F53, 0), [0, 0]);
});

test('mulhilo32: one × x = x', () => {
  assert.deepEqual(mulhilo32(1, 0),          [0, 0]);
  assert.deepEqual(mulhilo32(1, 0x12345678), [0, 0x12345678]);
  assert.deepEqual(mulhilo32(0xD2511F53, 1), [0, 0xD2511F53]);
});

test('mulhilo32: MAX × MAX (the worst case)', () => {
  // (2^32 - 1)^2 = 2^64 - 2^33 + 1 = 0xFFFFFFFE_00000001
  assert.deepEqual(mulhilo32(0xFFFFFFFF, 0xFFFFFFFF), [0xFFFFFFFE, 0x00000001]);
});

test('mulhilo32: small known values', () => {
  // 2 × 3 = 6
  assert.deepEqual(mulhilo32(2, 3), [0, 6]);
  // 0x10000 × 0x10000 = 0x100000000  →  hi=1, lo=0
  assert.deepEqual(mulhilo32(0x10000, 0x10000), [1, 0]);
  // 0xFFFF × 0xFFFF = 0xFFFE0001
  assert.deepEqual(mulhilo32(0xFFFF, 0xFFFF), [0, 0xFFFE0001]);
});

test('mulhilo32: matches BigInt reference for 1000 random pairs', () => {
  // Seeded rough RNG so the test is deterministic.
  let s = 0x12345678;
  function nextU32() {
    // xorshift32 — perfectly fine for picking test inputs
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;  s >>>= 0;
    return s >>> 0;
  }

  for (let i = 0; i < 1000; i++) {
    const a = nextU32();
    const b = nextU32();
    const fast = mulhilo32(a, b);
    const ref  = mulhilo32_bigint(a, b);
    if (fast[0] !== ref[0] || fast[1] !== ref[1]) {
      assert.fail(`mulhilo32 disagrees with BigInt for a=0x${a.toString(16)}, b=0x${b.toString(16)}: ` +
        `got [0x${fast[0].toString(16)}, 0x${fast[1].toString(16)}], ` +
        `expected [0x${ref[0].toString(16)}, 0x${ref[1].toString(16)}]`);
    }
  }
});

// =====================================================================
// 2. philox4x32_10 — Random123 KAT vectors
// =====================================================================

// KAT vector format: { counter, key, expected_output }
// Source: Random123 distribution, kat_vectors file.
const KAT_VECTORS = [
  {
    name: 'all-zeros',
    counter: [0, 0, 0, 0],
    key:     [0, 0],
    expect:  [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
  },
  {
    name: 'all-ones',
    counter: [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
    key:     [0xffffffff, 0xffffffff],
    expect:  [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
  },
  {
    name: 'pi-and-e',
    // Counter = first 16 hex digits of π. Key = first 8 hex digits of e.
    // These are the canonical "nothing-up-my-sleeve" test inputs the
    // Random123 paper uses.
    counter: [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
    key:     [0xa4093822, 0x299f31d0],
    expect:  [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
  },
];

for (const kat of KAT_VECTORS) {
  test(`philox4x32_10 KAT: ${kat.name}`, () => {
    const out = philox4x32_10(kat.counter, kat.key);
    for (let i = 0; i < 4; i++) {
      assert.equal(
        out[i],
        kat.expect[i],
        `output[${i}] mismatch: got 0x${out[i].toString(16)}, ` +
        `expected 0x${kat.expect[i].toString(16)}`
      );
    }
  });
}

test('philox4x32_10: deterministic — same inputs yield same outputs', () => {
  const c = [1, 2, 3, 4];
  const k = [5, 6];
  const a = philox4x32_10(c, k);
  const b = philox4x32_10(c, k);
  assert.deepEqual(a, b);
});

test('philox4x32_10: does not mutate input arrays', () => {
  const c = [1, 2, 3, 4];
  const k = [5, 6];
  philox4x32_10(c, k);
  assert.deepEqual(c, [1, 2, 3, 4]);
  assert.deepEqual(k, [5, 6]);
});

test('philox4x32_10: returns a fresh array (not aliased to input)', () => {
  const c = [1, 2, 3, 4];
  const k = [5, 6];
  const out = philox4x32_10(c, k);
  out[0] = 0xDEADBEEF;
  assert.equal(c[0], 1);  // input unaffected
});

test('philox4x32_10: small counter changes produce different outputs', () => {
  const k = [0, 0];
  const a = philox4x32_10([0, 0, 0, 0], k);
  const b = philox4x32_10([1, 0, 0, 0], k);
  // The avalanche property: changing one bit of input flips ~50% of output bits.
  // Just check they're different — strong avalanche test isn't our job here.
  assert.notDeepEqual(a, b);
});

test('philox4x32_10: small key changes produce different outputs', () => {
  const c = [0, 0, 0, 0];
  const a = philox4x32_10(c, [0, 0]);
  const b = philox4x32_10(c, [1, 0]);
  assert.notDeepEqual(a, b);
});

test('philox4x32_10: outputs are valid uint32s', () => {
  const out = philox4x32_10([1, 2, 3, 4], [5, 6]);
  for (const v of out) {
    assert.equal(typeof v, 'number');
    assert.ok(Number.isInteger(v), `${v} is not an integer`);
    assert.ok(v >= 0,         `${v} is negative`);
    assert.ok(v <= 0xFFFFFFFF, `${v} exceeds uint32 max`);
  }
});

// =====================================================================
// 3. State-threaded API
// =====================================================================

test('seedFromBytes: same bytes → same key', () => {
  const a = seedFromBytes([1, 2, 3, 4, 5]);
  const b = seedFromBytes([1, 2, 3, 4, 5]);
  assert.deepEqual(a.key, b.key);
});

test('seedFromBytes: different bytes → different key', () => {
  const a = seedFromBytes([1, 2, 3, 4, 5]);
  const b = seedFromBytes([1, 2, 3, 4, 6]);
  assert.notDeepEqual(a.key, b.key);
});

test('seedFromBytes: works with Uint8Array', () => {
  const arr = [1, 2, 3, 4, 5];
  const fromArr = seedFromBytes(arr);
  const fromTyped = seedFromBytes(new Uint8Array(arr));
  assert.deepEqual(fromArr.key, fromTyped.key);
});

test('seedFromBytes: handles empty input', () => {
  const s = seedFromBytes([]);
  assert.equal(typeof s.key[0], 'number');
  assert.equal(typeof s.key[1], 'number');
  assert.deepEqual(s.counter, [0, 0, 0, 0]);
});

test('seedFromBytes: handles long inputs (32+ bytes)', () => {
  const longSeed = new Array(64).fill(0).map((_, i) => i);
  const s = seedFromBytes(longSeed);
  assert.ok(s.key[0] !== 0 || s.key[1] !== 0);
});

test('stateFromKey: produces deterministic stream', () => {
  const s0 = stateFromKey(0xa4093822, 0x299f31d0);
  const [v1] = nextUint32(s0);
  const [v2] = nextUint32(s0);  // same input state → same output
  assert.equal(v1, v2);
});

test('nextUint32: state advances; same key but new state yields next value', () => {
  let state = stateFromKey(0, 0);
  const seen = [];
  for (let i = 0; i < 8; i++) {
    const [v, next] = nextUint32(state);
    seen.push(v);
    state = next;
  }
  // The first four uint32s should match philox4x32_10([0,0,0,0], [0,0]).
  // The next four should match philox4x32_10([1,0,0,0], [0,0]).
  const expectBlock0 = philox4x32_10([0, 0, 0, 0], [0, 0]);
  const expectBlock1 = philox4x32_10([1, 0, 0, 0], [0, 0]);
  assert.deepEqual(seen.slice(0, 4), expectBlock0);
  assert.deepEqual(seen.slice(4, 8), expectBlock1);
});

test('nextUint32: original state is not mutated by call', () => {
  const s0 = stateFromKey(42, 1234);
  const counter0 = [...s0.counter];
  const blockIdx0 = s0.blockIdx;
  nextUint32(s0);
  // s0 should still have the same counter and blockIdx after the call.
  assert.deepEqual(s0.counter, counter0);
  assert.equal(s0.blockIdx, blockIdx0);
});

test('nextUniform: values are in [0, 1)', () => {
  let state = stateFromKey(7, 13);
  for (let i = 0; i < 1000; i++) {
    const [v, next] = nextUniform(state);
    assert.ok(v >= 0,  `value ${v} below 0`);
    assert.ok(v <  1,  `value ${v} not strictly below 1`);
    state = next;
  }
});

test('nextUniform: rough mean for 10000 draws ≈ 0.5', () => {
  let state = stateFromKey(0x1234, 0x5678);
  let sum = 0;
  const n = 10000;
  for (let i = 0; i < n; i++) {
    const [v, next] = nextUniform(state);
    sum += v;
    state = next;
  }
  const mean = sum / n;
  // 3-sigma envelope for U(0,1): σ²=1/12, σ_mean = 1/√(12n) ≈ 0.0029.
  // So 3σ ≈ 0.0087.
  assert.ok(Math.abs(mean - 0.5) < 0.01,
    `sample mean ${mean} differs from 0.5 by more than the 3σ bound 0.01`);
});

test('incrementCounter: simple increment (no carry)', () => {
  assert.deepEqual(incrementCounter([0, 0, 0, 0]),       [1, 0, 0, 0]);
  assert.deepEqual(incrementCounter([42, 0, 0, 0]),      [43, 0, 0, 0]);
  assert.deepEqual(incrementCounter([0xFFFFFFFE, 0, 0, 0]), [0xFFFFFFFF, 0, 0, 0]);
});

test('incrementCounter: carry through one word', () => {
  assert.deepEqual(incrementCounter([0xFFFFFFFF, 0, 0, 0]),  [0, 1, 0, 0]);
  assert.deepEqual(incrementCounter([0xFFFFFFFF, 5, 0, 0]),  [0, 6, 0, 0]);
});

test('incrementCounter: carry through all four words (full rollover)', () => {
  assert.deepEqual(
    incrementCounter([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]),
    [0, 0, 0, 0]
  );
});

test('incrementCounter: chained carries (0xFFFFFFFF → 0xFFFFFFFF → x)', () => {
  assert.deepEqual(
    incrementCounter([0xFFFFFFFF, 0xFFFFFFFF, 0, 0]),
    [0, 0, 1, 0]
  );
  assert.deepEqual(
    incrementCounter([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0]),
    [0, 0, 0, 1]
  );
});

test('incrementCounter: does not mutate input', () => {
  const c = [0xFFFFFFFF, 5, 0, 0];
  incrementCounter(c);
  assert.deepEqual(c, [0xFFFFFFFF, 5, 0, 0]);
});

// Cross-check: the state-threaded uint32 stream should reproduce direct
// cipher calls with consecutive counters, exactly. This is the property
// the entire sampler architecture depends on.

test('stream-vs-cipher equivalence: 100 stream values match 25 cipher blocks', () => {
  let state = stateFromKey(0xCAFEBABE, 0xDEADBEEF);
  const stream = [];
  for (let i = 0; i < 100; i++) {
    const [v, next] = nextUint32(state);
    stream.push(v);
    state = next;
  }

  const direct = [];
  let counter = [0, 0, 0, 0];
  for (let block = 0; block < 25; block++) {
    direct.push(...philox4x32_10(counter, [0xCAFEBABE, 0xDEADBEEF]));
    counter = incrementCounter(counter);
  }

  assert.deepEqual(stream, direct);
});

// Reproducibility: rebuild the state from the same seed, walk N draws,
// expect the exact same values both times. This is the property our
// content-addressed cache relies on.

test('reproducibility: same seed → same stream, every time', () => {
  function takeN(seed, n) {
    let state = seedFromBytes(seed);
    const out = [];
    for (let i = 0; i < n; i++) {
      const [v, next] = nextUint32(state);
      out.push(v);
      state = next;
    }
    return out;
  }
  const seed = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
  const first  = takeN(seed, 50);
  const second = takeN(seed, 50);
  assert.deepEqual(first, second);
});
