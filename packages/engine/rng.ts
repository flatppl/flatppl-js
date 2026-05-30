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

function mulhilo32(a: number, b: number) {
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

function philox4x32Round(c0: number, c1: number, c2: number, c3: number, k0: number, k1: number) {
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

function philox4x32_10(counter: ArrayLike<number>, key: ArrayLike<number>) {
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

function seedFromBytes(bytes: Iterable<number>) {
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

function stateFromKey(k0: number, k1: number) {
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

function incrementCounter(counter: ArrayLike<number>) {
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

function nextUint32(state: any) {
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

function nextUniform(state: any) {
  const [u, s] = nextUint32(state) as [number, any];
  return [u / 0x100000000, s];
}

// =====================================================================
// State serialization (per spec §sec:random `rngstate(bytes)`)
// =====================================================================
//
// Serialize a Philox state to a byte vector and back, with a header that
// lets us reject incompatible states (different RNG algorithm, different
// engine, …) on deserialization. Layout (24 bytes total):
//
//   bytes 0..3   "PX10" magic — Philox-4×32-10 marker
//   bytes 4..7   key[0]      (uint32, big-endian)
//   bytes 8..11  key[1]
//   bytes 12..15 counter[0]
//   bytes 16..19 counter[1]
//   bytes 20..23 counter[2]
//
// counter[3] is omitted: the cipher only ever increments counter[0..2]
// in our usage (4×32-bit block per increment of counter[0..3] but we
// don't span 2⁹⁶ blocks in any realistic run). `block` and `blockIdx`
// are NOT serialized — round-tripping through bytes resets the cache,
// which costs at most one cipher call on next read.
//
// Engines with a different cipher should use a different magic; loading
// a state whose magic doesn't match this implementation throws.

const PHILOX_MAGIC = [0x50, 0x58, 0x31, 0x30]; // "PX10"

function bytesFromState(state: any) {
  const out = new Array(24);
  for (let i = 0; i < 4; i++) out[i] = PHILOX_MAGIC[i];
  writeU32BE(out, 4,  state.key[0]);
  writeU32BE(out, 8,  state.key[1]);
  writeU32BE(out, 12, state.counter[0]);
  writeU32BE(out, 16, state.counter[1]);
  writeU32BE(out, 20, state.counter[2]);
  return out;
}

function stateFromBytes(bytes: Iterable<number>) {
  const arr = Array.from(bytes, (b: any) => (b as number) & 0xff);
  if (arr.length < 24) {
    throw new Error(
      `rng.stateFromBytes: need ≥24 bytes, got ${arr.length}`
    );
  }
  for (let i = 0; i < 4; i++) {
    if (arr[i] !== PHILOX_MAGIC[i]) {
      throw new Error(
        `rng.stateFromBytes: magic mismatch — bytes do not encode a ` +
        `Philox-4×32-10 state (this engine only accepts its own format)`
      );
    }
  }
  return {
    key:      [readU32BE(arr, 4),  readU32BE(arr, 8)],
    counter:  [readU32BE(arr, 12), readU32BE(arr, 16), readU32BE(arr, 20), 0],
    block:    null,
    blockIdx: 4,
  };
}

function writeU32BE(out: number[], off: number, v: number) {
  out[off    ] = (v >>> 24) & 0xff;
  out[off + 1] = (v >>> 16) & 0xff;
  out[off + 2] = (v >>>  8) & 0xff;
  out[off + 3] = (v       ) & 0xff;
}

function readU32BE(arr: ArrayLike<number>, off: number) {
  return (((arr[off] << 24) | (arr[off + 1] << 16)
        | (arr[off + 2] << 8) |  arr[off + 3]) >>> 0);
}

// =====================================================================
// Splittable PRNG primitives (commit 1 — JAX-style split/foldIn over
// Philox-4x32)
// =====================================================================
//
// The streaming `next*` API above is a state-threaded *sequential* view
// of one Philox stream. The primitives in this section instead treat
// Philox as a keyed bijection and derive *new keys* from a parent key
// + a discriminator, in the manner of JAX's `jax.random.split` /
// `fold_in`. This unlocks two things future commits will use:
//
//   - independent substreams per sampler node (no shared counter, no
//     ordering coupling between unrelated draws);
//   - bulk uniform/normal kernels that produce N values in one call,
//     amortizing the per-call overhead of the state-threaded API.
//
// Nothing in this section mutates or replaces the existing streaming
// API; `nextUniform` / `nextUint32` remain byte-compatible with the
// pre-commit-1 implementation.

// xxHash32 round constants. These are the canonical values from Yann
// Collet's reference implementation (github.com/Cyan4973/xxHash,
// xxhash32.c). They are coprime with 2^32 and chosen for good avalanche
// behaviour on small inputs.

const XXH_PRIME32_1 = 0x9E3779B1 | 0;   // |0 keeps these as int32 literals
const XXH_PRIME32_2 = 0x85EBCA77 | 0;
const XXH_PRIME32_3 = 0xC2B2AE3D | 0;
const XXH_PRIME32_4 = 0x27D4EB2F | 0;
const XXH_PRIME32_5 = 0x165667B1 | 0;

// Lazy UTF-8 encoder for the xxhash32(string) path. Allocating
// TextEncoder at module load would penalize callers that never hash
// strings (the sampler hot path hashes bytes / ints, not strings).

let _xxhTextEncoder: TextEncoder | null = null;
function _getTextEncoder(): TextEncoder {
  if (_xxhTextEncoder === null) _xxhTextEncoder = new TextEncoder();
  return _xxhTextEncoder;
}

// 32-bit left rotate. `Math.imul` not needed here; `<<` / `>>>` keep
// us inside int32 / uint32 lanes.
function _xxhRotl32(x: number, r: number): number {
  return ((x << r) | (x >>> (32 - r))) >>> 0;
}

// Non-streaming xxHash32 (Yann Collet's algorithm). Returns a uint32.
//
// Supported inputs:
//   - `number` — hashed as a little-endian uint32. Useful for hashing
//     tags / integer ids without manually packing them into bytes.
//   - `Uint8Array` — hashed directly, zero-copy.
//   - `string` — UTF-8 encoded via a lazily-allocated TextEncoder.
//
// Implementation follows the xxhash32.c reference: an accumulator-based
// main loop for ≥16-byte inputs (4 lanes × 4-byte stripes), then a
// shorter mixing path for the tail (4-byte chunks down to single bytes).
// `Math.imul` is mandatory: JS multiplication of two 32-bit values
// goes through doubles and silently loses low bits for products that
// overflow 2^53.

type Xxh32Input = number | Uint8Array | string;

function xxhash32(input: Xxh32Input, seed: number = 0): number {
  let bytes: Uint8Array;
  if (typeof input === 'number') {
    // Pack as little-endian uint32. We deliberately do NOT pack as a
    // double — callers that want to hash a float should pass its byte
    // representation as a Uint8Array.
    const v = input >>> 0;
    bytes = new Uint8Array(4);
    bytes[0] =  v        & 0xff;
    bytes[1] = (v >>>  8) & 0xff;
    bytes[2] = (v >>> 16) & 0xff;
    bytes[3] = (v >>> 24) & 0xff;
  } else if (typeof input === 'string') {
    bytes = _getTextEncoder().encode(input);
  } else {
    bytes = input;
  }

  const len = bytes.length;
  const s   = seed >>> 0;
  let h32: number;
  let p = 0;

  if (len >= 16) {
    // Main loop: four parallel accumulators consume 16 bytes per
    // iteration. Each accumulator is round(acc, lane) where
    //   round(a, l) = rotl32(a + l*PRIME2, 13) * PRIME1.
    let v1 = (s + XXH_PRIME32_1 + XXH_PRIME32_2) >>> 0;
    let v2 = (s + XXH_PRIME32_2) >>> 0;
    let v3 = (s + 0) >>> 0;
    let v4 = (s - XXH_PRIME32_1) >>> 0;
    const limit = len - 16;
    while (p <= limit) {
      const l1 = (bytes[p    ] | (bytes[p + 1] << 8) | (bytes[p + 2] << 16) | (bytes[p + 3] << 24)) >>> 0;
      const l2 = (bytes[p + 4] | (bytes[p + 5] << 8) | (bytes[p + 6] << 16) | (bytes[p + 7] << 24)) >>> 0;
      const l3 = (bytes[p + 8] | (bytes[p + 9] << 8) | (bytes[p + 10] << 16) | (bytes[p + 11] << 24)) >>> 0;
      const l4 = (bytes[p + 12] | (bytes[p + 13] << 8) | (bytes[p + 14] << 16) | (bytes[p + 15] << 24)) >>> 0;
      v1 = Math.imul(_xxhRotl32((v1 + Math.imul(l1, XXH_PRIME32_2)) >>> 0, 13), XXH_PRIME32_1) >>> 0;
      v2 = Math.imul(_xxhRotl32((v2 + Math.imul(l2, XXH_PRIME32_2)) >>> 0, 13), XXH_PRIME32_1) >>> 0;
      v3 = Math.imul(_xxhRotl32((v3 + Math.imul(l3, XXH_PRIME32_2)) >>> 0, 13), XXH_PRIME32_1) >>> 0;
      v4 = Math.imul(_xxhRotl32((v4 + Math.imul(l4, XXH_PRIME32_2)) >>> 0, 13), XXH_PRIME32_1) >>> 0;
      p += 16;
    }
    h32 = (_xxhRotl32(v1, 1) + _xxhRotl32(v2, 7) + _xxhRotl32(v3, 12) + _xxhRotl32(v4, 18)) >>> 0;
  } else {
    h32 = (s + XXH_PRIME32_5) >>> 0;
  }

  h32 = (h32 + len) >>> 0;

  // 4-byte tail chunks: each lane mixes in with rotl17 then *PRIME4.
  while (p + 4 <= len) {
    const lane = (bytes[p] | (bytes[p + 1] << 8) | (bytes[p + 2] << 16) | (bytes[p + 3] << 24)) >>> 0;
    h32 = Math.imul(_xxhRotl32((h32 + Math.imul(lane, XXH_PRIME32_3)) >>> 0, 17), XXH_PRIME32_4) >>> 0;
    p += 4;
  }

  // 1-byte tail bytes: each byte mixes in with rotl11 then *PRIME1.
  while (p < len) {
    h32 = Math.imul(_xxhRotl32((h32 + Math.imul(bytes[p], XXH_PRIME32_5)) >>> 0, 11), XXH_PRIME32_1) >>> 0;
    p += 1;
  }

  // Final avalanche: three xor-shift+multiply rounds.
  h32 = Math.imul(h32 ^ (h32 >>> 15), XXH_PRIME32_2) >>> 0;
  h32 = Math.imul(h32 ^ (h32 >>> 13), XXH_PRIME32_3) >>> 0;
  h32 = (h32 ^ (h32 >>> 16)) >>> 0;

  return h32;
}

// A Philox key is just a pair of uint32s. Aliased here for readability
// of the split/foldIn signatures.
type PhiloxKey = [number, number];

// Derive a Philox key from a seed. Two input shapes:
//
//   - integer: lane 0 is the seed itself (zero-cost passthrough for
//     `keyFromSeed(0)`); lane 1 is `seed * PRIME32_1`, which is a
//     well-known cheap mixing step and matches the pattern xxhash32
//     uses for its accumulator init. This is a deliberate change from
//     the old `stateFromKey(seed, 0)` convention — splitting an
//     unmixed (seed, 0) key concentrates entropy in one lane and
//     produces visibly-correlated children at small split counts.
//
//   - byte vector: xxhash32 the bytes twice with two different seeds
//     (0 and PRIME32_1). Two passes give us two well-mixed lanes from
//     a single algorithm; using the same hash function for both lanes
//     keeps the cross-engine reproducibility story simple.
//
// `stateFromKey` (unchanged above) still exists for callers that want
// to build a state from an explicit key pair without going through
// keyFromSeed; it remains byte-compatible with pre-commit-1 behaviour.

function keyFromSeed(seed: number | Uint8Array): PhiloxKey {
  if (typeof seed === 'number') {
    const k0 = seed >>> 0;
    const k1 = Math.imul(seed | 0, XXH_PRIME32_1) >>> 0;
    return [k0, k1];
  }
  const k0 = xxhash32(seed, 0);
  const k1 = xxhash32(seed, XXH_PRIME32_1 >>> 0);
  return [k0, k1];
}

// JAX-style split. Produces `n` independent child keys from a parent
// key, by running philox4x32_10 with counter `[i, n + i, 0, 0]` for
// i in 0..n and taking the first 2 lanes of each output block as the
// child key. The `(i, n + i)` layout (rather than just `i`) is the
// JAX convention: it ensures that splitting twice with different `n`
// values produces non-overlapping child-key sequences, so callers can
// safely re-split a parent key at different fan-outs without aliasing.

function split(key: PhiloxKey, n: number): PhiloxKey[] {
  const out: PhiloxKey[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const block = philox4x32_10([i >>> 0, (n + i) >>> 0, 0, 0], key);
    out[i] = [block[0] >>> 0, block[1] >>> 0];
  }
  return out;
}

// Derive a single child key from a parent + 32-bit tag. Used to absorb
// a discriminator (loop index, branch id, hashed binding name, ...)
// into a key without changing its "fan-out".
//
// Domain separation: foldIn uses counter `[0, tag, 0, 1]` — lane 3 = 1
// is a constant that distinguishes the foldIn counter space from
// `split`'s `[i, n+i, 0, 0]` counter space, guaranteeing
// `foldIn(K, x) != split(K, n)[i]` for any (K, x, n, i). Without the
// domain separator a collision exists when `tag = n+0 = n` (caught by
// the structural-independence KAT test).

function foldIn(key: PhiloxKey, tag: number): PhiloxKey {
  const block = philox4x32_10([0, tag >>> 0, 0, 1], key);
  return [block[0] >>> 0, block[1] >>> 0];
}

// =====================================================================
// Bulk uniform / normal fast-paths (commit 1 — kernels only; wiring
// into the sampler comes in commit 3)
// =====================================================================
//
// These produce N values in one call, amortizing the per-call closure
// / object-allocation cost of nextUniform. The contract is:
//
//   philoxNUniform(state, n, out?) MUST be bit-exact equivalent to
//   calling nextUniform(state) n times in sequence — same uint32s,
//   same `lane / 2^32` mapping, same counter advance, same cache
//   semantics on the trailing partial block.
//
// philoxNNormal is a separate Box-Muller kernel and does NOT match
// any scalar normal sampler bit-for-bit; it draws m = 2*ceil(n/2)
// uniforms via philoxNUniform and pairs them into (cos, sin) outputs.

interface PhiloxState {
  key:      [number, number];
  counter:  number[];
  block:    number[] | null;
  blockIdx: number;
}

interface PhiloxBulkResult {
  state: PhiloxState;
  out:   Float64Array;
}

function philoxNUniform(
  state: PhiloxState,
  n: number,
  out?: Float64Array,
): PhiloxBulkResult {
  const dest = out ?? new Float64Array(n);
  if (n <= 0) {
    return { state, out: dest };
  }

  // Take a working copy of the parts we'll mutate. We never assign back
  // into the caller's state object; the function is pure.
  let counter  = state.counter;
  let block    = state.block;
  let blockIdx = state.blockIdx;
  const key    = state.key;

  let i = 0;

  // 1. Drain any cached output from a previous nextUniform call. This
  //    preserves bit-exact compatibility with scalar nextUniform: if
  //    the caller had blockIdx=2, the next two values come from the
  //    cached block before we start any new cipher calls.
  if (block !== null) {
    while (blockIdx < 4 && i < n) {
      dest[i++] = (block[blockIdx++] >>> 0) / 0x100000000;
    }
  }

  // 2. Bulk loop: while at least 4 outputs remain, run the cipher and
  //    consume the full block of 4 uint32s without caching it.
  while (i + 4 <= n) {
    const b = philox4x32_10(counter, key);
    counter = incrementCounter(counter);
    dest[i    ] = (b[0] >>> 0) / 0x100000000;
    dest[i + 1] = (b[1] >>> 0) / 0x100000000;
    dest[i + 2] = (b[2] >>> 0) / 0x100000000;
    dest[i + 3] = (b[3] >>> 0) / 0x100000000;
    i += 4;
    block    = null;
    blockIdx = 4;
  }

  // 3. Tail: fewer than 4 outputs remain. Run one more cipher call,
  //    consume what we need, and cache the rest so the next scalar /
  //    bulk call sees an identical state to "scalar nextUniform was
  //    called n times".
  if (i < n) {
    const b = philox4x32_10(counter, key);
    counter = incrementCounter(counter);
    blockIdx = 0;
    while (blockIdx < 4 && i < n) {
      dest[i++] = (b[blockIdx++] >>> 0) / 0x100000000;
    }
    block = b;
  }

  return {
    state: { key, counter, block, blockIdx },
    out:   dest,
  };
}

// Bulk normal sampling via paired Box-Muller. Draws `m = 2*ceil(n/2)`
// uniforms then pairs them into (r*cos(theta), r*sin(theta)). The
// `max(u1, 2^-32)` floor avoids log(0) when the uniform draws the
// smallest representable value (a 1-in-2^32 event, but worth guarding
// since log(0) would silently produce -Inf -> NaN).
//
// Bit-equivalence to scalar nextNormal is NOT required: we'll wire
// this in as a separate fast-path in commit 3 alongside a similarly
// re-keyed scalar normal. Until then it's exported as a kernel only.

const _BM_FLOOR = Math.pow(2, -32);    // smallest possible uniform draw

function philoxNNormal(
  state: PhiloxState,
  n: number,
  out?: Float64Array,
): PhiloxBulkResult {
  const dest = out ?? new Float64Array(n);
  if (n <= 0) {
    return { state, out: dest };
  }

  const m = 2 * Math.ceil(n / 2);
  // Draw the uniforms into a scratch buffer; reusing `dest` would
  // require `dest.length >= m`, which we can't assume when n is odd.
  const uniforms = new Float64Array(m);
  const { state: nextState } = philoxNUniform(state, m, uniforms);

  for (let i = 0; i < n; i += 2) {
    const u1Raw = uniforms[i];
    const u2    = uniforms[i + 1];
    const u1    = u1Raw < _BM_FLOOR ? _BM_FLOOR : u1Raw;
    const r     = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    dest[i] = r * Math.cos(theta);
    if (i + 1 < n) {
      dest[i + 1] = r * Math.sin(theta);
    }
  }

  return { state: nextState, out: dest };
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

  // Serialization (per spec §sec:random)
  bytesFromState,
  stateFromBytes,

  // Splittable PRNG primitives (commit 1)
  xxhash32,
  keyFromSeed,
  split,
  foldIn,
  philoxNUniform,
  philoxNNormal,

  // Internal — exported for tests; not part of the public surface.
  _internal: { mulhilo32, philox4x32Round, PHILOX_M_4x32_0, PHILOX_M_4x32_1 },
};
