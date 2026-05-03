'use strict';

// Philox-4x32-10 — counter-based pseudorandom number generator.
//
// Reference: Salmon, Moraes, Dror, Shaw (SC11),
//   "Parallel Random Numbers: As Easy as 1, 2, 3"
//   https://www.thesalmons.org/john/random123/
//
// =====================================================================
// Why Philox-4x32-10 in FlatPPL
// =====================================================================
//
// Philox is a *cipher-style* RNG: given a (key, counter) pair, the
// output is a fixed pseudorandom 4-tuple. There is no internal sequential
// state to evolve — the entire stream is determined by the (key, counter)
// pair. Three properties this gives us, all directly relevant to the
// FlatPPL sampler:
//
//   1. Trivial reproducibility. `sample[i]` is a pure function of the
//      per-node key and `i` regardless of how many other samples have
//      been computed elsewhere. The cache architecture (content-addressed
//      per-binding) hinges on this: if the i-th sample of a binding is
//      always the same value for a given key, partial re-sampling stays
//      consistent with previously cached values.
//
//   2. Embarrassingly parallel. Workers can compute disjoint counter
//      ranges of the same stream with no coordination. Future Web-Worker
//      sampler will use this.
//
//   3. Cross-implementation reproducibility. Philox-4x32-10 is the
//      de-facto standard counter-based RNG: cuRAND ships it, NumPy ships
//      it, PyTorch ships it, TensorFlow ships it. A future Rust/Julia/
//      Python FlatPPL engine using the same key-derivation rule will
//      produce bit-exact identical samples.
//
// Per the FlatPPL spec (docs/07-functions.md §Random value generation),
// `rand(state, m)` is the user-visible primitive and `rngstates` is its
// opaque type. This module sits one level below: it implements the
// cipher and a thin state-threaded helper API the sampler builds on.
// `rnginit`, `rand`, and `rngstate` themselves live in the sampler/engine
// API layer above.
//
// =====================================================================
// Layout
// =====================================================================
//
//   - `mulhilo32(a, b)`      — 32×32 → 64-bit multiply, returned as [hi, lo]
//   - `philox4x32_10(c, k)`  — the raw cipher: 10 rounds, deterministic
//   - `seedFromBytes(bytes)` — derive a state from a byte vector
//   - `stateFromKey(k0, k1)` — for tests / explicit key construction
//   - `nextUint32(state)`    — produce one uint32 from the stream
//   - `nextUniform(state)`   — produce one float in [0, 1) from the stream
//   - `incrementCounter(c)`  — pure 128-bit increment

// ---------------------------------------------------------------------
// 32×32 → 64-bit multiplication
// ---------------------------------------------------------------------
//
// JS numbers can represent integers exactly up to 2^53. A 32-bit × 32-bit
// product is up to 64 bits and doesn't fit in one number. We split each
// 32-bit operand into two 16-bit halves and compute four 16×16 → 32-bit
// subproducts; partial sums stay within ~2^33, comfortably inside the
// safe-integer range. This avoids BigInt overhead (which would be ~5×
// slower on a hot path that runs millions of times per sampling pass).
//
// Returns [hi, lo] where each is a uint32 (non-negative double).

function mulhilo32(a, b) {
  const aL = a & 0xFFFF;
  const aH = a >>> 16;
  const bL = b & 0xFFFF;
  const bH = b >>> 16;

  const ll = aL * bL;             // 0 .. ~2^32  (fits in double)
  const lh = aL * bH;             // 0 .. ~2^32
  const hl = aH * bL;             // 0 .. ~2^32
  const hh = aH * bH;             // 0 .. ~2^32

  // The 64-bit product is:   ll + (lh + hl) * 2^16 + hh * 2^32
  //
  // Compute mid = lh + hl (up to ~2^33). The low 16 bits of mid contribute
  // to the low half of the result (shifted up by 16); the rest contributes
  // to the high half.
  const mid     = lh + hl;
  const midLo16 = mid & 0xFFFF;
  // Caution: `mid >>> 16` would *first* coerce `mid` to int32 (mod 2^32),
  // discarding any bits >= 32. Use Math.floor / division to keep all bits.
  const midHi   = Math.floor(mid / 0x10000);

  // Low 32 bits = (ll + midLo16 << 16) mod 2^32. Track carry separately
  // because the unsigned sum can momentarily exceed 2^32.
  const sumLow = ll + midLo16 * 0x10000;
  const lo     = sumLow >>> 0;
  const carry  = sumLow >= 0x100000000 ? 1 : 0;

  // High 32 bits = hh + midHi + carry, truncated.
  const hi = (hh + midHi + carry) >>> 0;

  return [hi, lo];
}

// ---------------------------------------------------------------------
// Round constants
// ---------------------------------------------------------------------
//
// Philox-4x32-10 uses two multiplier constants for the S-box and two
// Weyl constants for the per-round key schedule. Values from the
// reference Random123 implementation. They're chosen to be coprime with
// 2^32 and to have good avalanche behavior.

const PHILOX_M_4x32_0 = 0xD2511F53;
const PHILOX_M_4x32_1 = 0xCD9E8D57;

// PHILOX_W_32_0 ≈ φ × 2^32, PHILOX_W_32_1 ≈ √3 × 2^32.
const PHILOX_W_32_0 = 0x9E3779B9;
const PHILOX_W_32_1 = 0xBB67AE85;

// One round of the Philox-4x32 bijection. Algorithm 4 in Salmon et al.:
//
//   multhilo(M[0], C[0])  →  hi0, lo0
//   multhilo(M[1], C[2])  →  hi1, lo1
//   C'[0] = hi1 ⊕ C[1] ⊕ K[0]
//   C'[1] = lo1
//   C'[2] = hi0 ⊕ C[3] ⊕ K[1]
//   C'[3] = lo0

function philox4x32Round(c0, c1, c2, c3, k0, k1) {
  const [hi0, lo0] = mulhilo32(PHILOX_M_4x32_0, c0);
  const [hi1, lo1] = mulhilo32(PHILOX_M_4x32_1, c2);
  return [
    (hi1 ^ c1 ^ k0) >>> 0,
    lo1,
    (hi0 ^ c3 ^ k1) >>> 0,
    lo0,
  ];
}

// ---------------------------------------------------------------------
// 10-round Philox-4x32-10 cipher
// ---------------------------------------------------------------------
//
// `counter`: array of 4 uint32s.  `key`: array of 2 uint32s.
// Returns: array of 4 uint32s (a fresh array; inputs are not mutated).
//
// 10 is the standard round count: passes BigCrush and provides ample
// security margin for non-cryptographic use. 7-round variants exist but
// have known statistical weaknesses; 10 is the de facto cross-library
// default (cuRAND, PyTorch, TF, NumPy, …).

function philox4x32_10(counter, key) {
  let c0 = counter[0] >>> 0;
  let c1 = counter[1] >>> 0;
  let c2 = counter[2] >>> 0;
  let c3 = counter[3] >>> 0;
  let k0 = key[0] >>> 0;
  let k1 = key[1] >>> 0;

  // 10 rounds; the round key advances by Weyl constants between rounds.
  // No bump after the final round (the 11th key would be unused).
  for (let round = 0; round < 10; round++) {
    [c0, c1, c2, c3] = philox4x32Round(c0, c1, c2, c3, k0, k1);
    if (round < 9) {
      k0 = (k0 + PHILOX_W_32_0) >>> 0;
      k1 = (k1 + PHILOX_W_32_1) >>> 0;
    }
  }

  return [c0, c1, c2, c3];
}

// ---------------------------------------------------------------------
// State-threaded API
// ---------------------------------------------------------------------
//
// The sampler runs draws as `(value, new_state) = rand(state, measure)`,
// matching the spec's pure-functional state-threading. This module
// exposes the lowest layer of that pattern: get one uint32 (or one
// uniform float) per call, with state advancing.
//
// State shape:
//
//   { key:      [u32, u32],
//     counter:  [u32, u32, u32, u32],   // 128-bit, little-endian: counter[0] is low
//     block:    [u32, u32, u32, u32] | null,
//     blockIdx: 0..4 }
//
// We cache the last cipher output in `block`. Each call consumes one
// uint32 from it and increments `blockIdx`. When `blockIdx === 4` we
// re-encrypt with the next counter value. This amortizes the (40-mul)
// cost of a Philox call across 4 draws, important for fast-path samplers
// that consume one uniform per call (e.g. ITS samplers).
//
// Every state-mutating helper returns a NEW state object — caller-side
// references to the old state remain valid and unchanged. Underlying
// arrays are copied where mutation would otherwise leak.

// Build a state from a seed byte vector. Per FlatPPL spec, `rngseed`
// is a vector of bytes; the spec recommends ≥32 bytes for sufficient
// entropy. We compress arbitrary-length seeds into a 64-bit Philox key
// via FNV-1a-64 (small, fast, deterministic, no crypto requirement).
//
// Different seed lengths still produce different keys: FNV's avalanche
// is coarse but the iteration over each byte differentiates them.
//
// `bytes` may be any iterable of integers in [0, 255] (Array, Uint8Array,
// Buffer). Values outside that range are masked.

function seedFromBytes(bytes) {
  let h = 0xcbf29ce484222325n;       // FNV offset basis (64-bit)
  const PRIME = 0x100000001b3n;      // FNV prime (64-bit)
  for (const b of bytes) {
    h ^= BigInt(b & 0xff);
    h = (h * PRIME) & 0xffffffffffffffffn;
  }
  const k0 = Number((h >> 32n) & 0xffffffffn);
  const k1 = Number(h & 0xffffffffn);
  return {
    key:      [k0, k1],
    counter:  [0, 0, 0, 0],
    block:    null,
    blockIdx: 4,                     // forces refill on first nextUint32 call
  };
}

// Build a state from an explicit (k0, k1) key. Mostly for tests and
// for callers that want full control over the cipher key derivation.

function stateFromKey(k0, k1) {
  return {
    key:      [k0 >>> 0, k1 >>> 0],
    counter:  [0, 0, 0, 0],
    block:    null,
    blockIdx: 4,
  };
}

// Increment the 128-bit counter (4 × uint32, little-endian: counter[0]
// is the low word). Pure: returns a new array; the original is left
// alone so callers' captured states remain valid.

function incrementCounter(counter) {
  const out = [counter[0], counter[1], counter[2], counter[3]];
  for (let i = 0; i < 4; i++) {
    out[i] = (out[i] + 1) >>> 0;
    if (out[i] !== 0) break;          // no carry; done
    // Otherwise the word wrapped to 0; carry into the next word.
  }
  return out;
}

// Get the next uint32 from the stream.  Returns [value, new_state].
//
// If the cached block is exhausted (or hasn't been populated yet), runs
// the cipher with the current counter, advances the counter, and starts
// reading from the new block.

function nextUint32(state) {
  let { key, counter, block, blockIdx } = state;

  if (blockIdx >= 4) {
    block    = philox4x32_10(counter, key);
    counter  = incrementCounter(counter);
    blockIdx = 0;
  }

  const value = block[blockIdx] >>> 0;
  return [
    value,
    { key, counter, block, blockIdx: blockIdx + 1 },
  ];
}

// Get the next uniform float in [0, 1).  Returns [value, new_state].
//
// Standard uint32-to-float conversion: divide by 2^32. The result is in
// [0, 1) — open at 1 because the maximum uint32 (0xFFFFFFFF) divided by
// 2^32 is 1 - 2^-32, which rounds to a representable double < 1.
//
// We use 32 bits of randomness here, not 53 (full mantissa). For most
// distribution samplers (Box–Muller, ITS, ratio-of-uniforms) 32 bits is
// plenty; the rare sampler that needs 53 bits can synthesize it from
// two consecutive uint32s.

function nextUniform(state) {
  const [u, s] = nextUint32(state);
  return [u / 0x100000000, s];
}

module.exports = {
  // Core cipher
  philox4x32_10,

  // State-threaded API
  seedFromBytes,
  stateFromKey,
  nextUint32,
  nextUniform,
  incrementCounter,

  // Internal — exported for tests; not part of the public surface.
  _internal: { mulhilo32, philox4x32Round, PHILOX_M_4x32_0, PHILOX_M_4x32_1 },
};
