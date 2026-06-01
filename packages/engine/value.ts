'use strict';

// =====================================================================
// Value — shape-tagged numeric value used throughout the engine
// =====================================================================
//
// The Value type is the engine's internal representation for every
// numeric quantity that flows through walkers, ops, and materialisers.
// It pairs a flat Float64Array storage with an explicit `shape: number[]`
// describing the dimensions, and an optional `dtype` slot reserved for
// future backend tagging (TF.js / typed-tensor swaps).
//
// Design choices (see TODO-flatppl-js.md + ARCHITECTURE.md
// invariant #2):
//
//   - **Uniform storage.** Every shape uses Float64Array; even
//     shape=[] (atom-indep scalar) is stored as Float64Array(1). No
//     JS-number internal representation. JS numbers are accepted only
//     at engine API boundaries (test ergonomics, worker messages) and
//     converted via `asValue`.
//
//   - **Leading axis = batch axis.** Per-atom slices are row-major
//     contiguous. Matches JAX / NumPy / PyTorch / TF / Stan / matIid
//     conventions. Differing engines (Julia column-major, future Rust)
//     pick their own internal layout — the IR stays shape-agnostic.
//
//   - **Plain `{shape, data, dtype?}` objects, no class.** V8 inlines
//     monomorphic property access; no `instanceof` dispatch needed.
//
//   - **dtype is optional and defaults to 'f64'.** Storage is always
//     Float64Array for now; the slot exists so future backends can
//     distinguish bool / int32 / complex tensors without retrofitting
//     every call site. TF.js's Tensor.dtype uses the same idea.
//
// Shape conventions (N is atom count, k/m/n are intrinsic dims):
//
//   shape=[]           atom-indep scalar              data length 1
//   shape=[N]          atom-batched scalar            data length N
//   shape=[k]          atom-indep vector              data length k
//   shape=[N, k]       atom-batched vector            data length N*k
//                      atom-major: atom i occupies indices [i*k, (i+1)*k)
//   shape=[m, n]       atom-indep matrix              data length m*n
//                      row-major: row i at indices [i*n, (i+1)*n)
//   shape=[N, m, n]    atom-batched matrix            data length N*m*n
//                      atom-major over row-major slices
//
// Ambiguity note: a Float64Array(N) standing in for shape=[N] vs a
// shape=[k=N] vector cannot be disambiguated from shape alone. The
// surrounding context (caller knows its N) resolves this — same
// convention the engine's current `isBatch(v, N)` check already uses.
//
// ---------------------------------------------------------------------
// Transpose / adjoint tag (Klein-4 algebra)
// ---------------------------------------------------------------------
//
// Every Value carries an optional `t` slot — a four-state tag from the
// Klein-4 group representing the value's orientation against transpose
// and complex conjugation:
//
//   'N'  normal (default; absent ⇒ 'N')                  — no flip, no conj
//   'T'  transposed                                       — flip, no conj
//   'A'  adjoint = transpose + conjugate                  — flip, conj
//   'C'  conjugated only (= T ∘ A)                        — no flip, conj
//
// Bit decomposition: (swapped, conjugated)
//   N = (0, 0)   T = (1, 0)   A = (1, 1)   C = (0, 1)
//
// Operations:
//   transpose toggles `swapped`              N↔T,  A↔C
//   adjoint   toggles both bits               N↔A,  T↔C
//   conjugate toggles `conjugated`            N↔C,  T↔A
//
// For real-valued data (dtype='f64') the conjugate bit is mathematically
// a no-op: N is observationally identical to C, T to A. The tag is
// nonetheless preserved through compositions so the algebra is correct
// once complex dtypes are introduced (no migration of call sites).
//
// `shape` is the LOGICAL shape (post-tag). For matrices, transpose
// updates both: a [m, n] Value with t='N' becomes a [n, m] Value with
// t='T'. `data` is preserved (no allocation); consumers compute the
// underlying storage shape via `_dataShape(v)` when they need to index
// the buffer directly.
//
// For vectors (shape=[k]) transpose toggles the tag but leaves the
// shape unchanged: row and column vectors are both 1-D objects in
// FlatPPL, distinguished by orientation tag rather than by being [1,k]
// vs [k,1] matrices. This matches the spec's vector / single-row-matrix
// distinction (a row matrix has shape [1, k] and is NOT a vector).
//
// BLAS analogue: the tag corresponds to the TRANSA/TRANSB flag on
// dgemm. When crossing backend boundaries (TF.js, future Rust /
// StableHLO), the tag is realized by an explicit transpose call once
// per crossing; internal hot paths get free transposes.

const DEFAULT_DTYPE = 'f64';
const DEFAULT_TAG   = 'N';

// Klein-4 tag transitions, computed from the (swapped, conjugated) bit
// representation. Pre-computed for cheap dispatch.
const _TAG_TRANSPOSE: Record<string, string> = { N: 'T', T: 'N', A: 'C', C: 'A' };
const _TAG_ADJOINT: Record<string, string>   = { N: 'A', T: 'C', A: 'N', C: 'T' };
const _TAG_CONJUGATE: Record<string, string> = { N: 'C', T: 'A', A: 'T', C: 'N' };
// Effective bits (used by consumers that need to know "should I treat
// the data as transposed?" without caring about conjugation).
const _TAG_SWAPPED: Record<string, boolean>    = { N: false, T: true,  A: true,  C: false };
const _TAG_CONJUGATED: Record<string, boolean> = { N: false, T: false, A: true,  C: true  };

// ---------------------------------------------------------------------
// Structured-matrix tag (`struct` — ORTHOGONAL to the Klein-4 tag)
// ---------------------------------------------------------------------
//
// One small integer bitmask with two layers:
//
//   Occupancy (where nonzeros MAY be) — exact boolean algebra:
//     ST_LOWER  1   strict lower triangle
//     ST_DIAG   2   the diagonal
//     ST_UPPER  4   strict upper triangle
//
//   Refinements (value FACTS, conservative — NOT occupancy):
//     ST_UNIT    8  diagonal ≡ 1 (implicit; cleared by `+`)
//     ST_SYM    16  symmetric / Hermitian
//     ST_POSDEF 32  positive-definite (implies ST_SYM)
//
// Absent `struct` ⇒ dense (LOWER|DIAG|UPPER = 7). An explicit `0` is a
// genuine all-zero matrix (the OR identity), distinct from absent.
//
// Algebra (kept as tiny pure functions, not pair-tables):
//   A + B          → occ = occ(A) | occ(B); SYM/POSDEF = AND; UNIT clear
//   c · A          → occ unchanged; SYM keep; POSDEF iff c>0; UNIT clear
//   transpose(A)   → swap LOWER↔UPPER; DIAG + refinements kept
//   adjoint(A)     → as transpose (entry-conj is the orthogonal Klein-4
//                    bit; it does not move the triangle)
//   conjugate(A)   → struct unchanged
//   A · B          → small occupancy propagation (see value-ops); generic
//                    mul clears SYM/POSDEF (gram/cholesky producers set
//                    them — mul never infers them)
//
// `struct` is orthogonal to the Klein-4 `t` tag: a matrix can be e.g.
// upper-triangular AND read transposed; the two never share a bit.
//
// Storage: only `diag` is vector-backed (data = the m-vector, logical
// shape = [m, m]); every other structure is dense + flag in v1. Any
// consumer without a structured fast-path calls `densify(v)` first, so
// correctness never depends on a fast-path existing (same contract as
// readComplex / _dataShape).
const ST_LOWER = 1, ST_DIAG = 2, ST_UPPER = 4;
const ST_UNIT = 8, ST_SYM = 16, ST_POSDEF = 32;
const ST_OCC_MASK = ST_LOWER | ST_DIAG | ST_UPPER;   // 7
const ST_DENSE = ST_OCC_MASK;                        // 7

// transpose / adjoint structure transition: swap LOWER↔UPPER, keep the
// diagonal bit and all refinements. conjugate leaves struct unchanged.
function _structTranspose(s: number) {
  let r = s & ~(ST_LOWER | ST_UPPER);
  if (s & ST_LOWER) r |= ST_UPPER;
  if (s & ST_UPPER) r |= ST_LOWER;
  return r;
}

// ---------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------

// Number of elements implied by a shape array. `[]` (scalar) → 1.
function numel(shape: ArrayLike<number>) {
  let n = 1;
  for (let i = 0; i < shape.length; i++) n *= shape[i];
  return n;
}

// Infer shape from a nested JS Array. Validates rectangularity at every
// level; throws on ragged structure. Empty array → shape=[0]; nested
// empties (e.g. [[]]) → shape=[1, 0].
function inferShapeFromNested(arr: any) {
  // Treat both JS Arrays and Float64Arrays as "nested-level" containers.
  // The legacy broadcast-reduce default in sampler-aggregate.ts emits
  // results in the shape `[Float64Array, Float64Array, …]` (array of
  // typed-array rows) — that's a rank-2 matrix, not a rank-1 vector
  // with NaN entries. Detect via instanceof Float64Array so the deeper
  // dim is captured and `flattenNested` copies the actual numeric data.
  const shape: number[] = [];
  let cur = arr;
  function isNestable(x: any) {
    return Array.isArray(x) || x instanceof Float64Array;
  }
  while (isNestable(cur)) {
    shape.push(cur.length);
    if (cur.length === 0) break;
    // Check rectangularity at this level.
    const first = cur[0];
    if (isNestable(first)) {
      const len = first.length;
      for (let i = 1; i < cur.length; i++) {
        const sib = cur[i];
        if (!isNestable(sib) || sib.length !== len) {
          throw new Error('asValue: nested array is ragged at depth ' + shape.length);
        }
      }
    } else {
      for (let i = 1; i < cur.length; i++) {
        if (isNestable(cur[i])) {
          throw new Error('asValue: nested array mixes scalars and arrays at depth ' + shape.length);
        }
      }
    }
    cur = cur[0];
  }
  return shape;
}

// Flatten a (possibly nested) JS Array into a Float64Array in row-major
// order. Caller has already validated rectangularity via
// `inferShapeFromNested`.
function flattenNested(arr: any, out: any, offset: number, depth: number, shape: ArrayLike<number>) {
  if (depth === shape.length) {
    out[offset] = +arr;
    return offset + 1;
  }
  for (let i = 0; i < arr.length; i++) {
    offset = flattenNested(arr[i], out, offset, depth + 1, shape);
  }
  return offset;
}

// ---------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------

function getShape(v: any) { return v.shape; }
function getData(v: any)  { return v.data; }
function getDType(v: any) { return v.dtype || DEFAULT_DTYPE; }
function getTag(v: any): string   { return v.t || DEFAULT_TAG; }

// Tag bit extractors. Use these in dispatch code rather than equality
// against the four state strings — they make "is this view swapped?"
// and "is this view conjugated?" clear at the call site.
function isTransposeView(v: any) { return _TAG_SWAPPED[getTag(v)]; }
function isConjugateView(v: any) { return _TAG_CONJUGATED[getTag(v)]; }

// Storage shape — the dimensions of the underlying Float64Array as laid
// out in memory, before applying the transpose tag. For a Value with
// shape=[m, n] and t='T', the stored data is laid out [n, m] row-major.
// For vectors and scalars (1-D or 0-D), the storage shape equals shape.
//
// For higher-rank tensors (rank ≥ 3, e.g. atom-batched matrices
// shape=[N, m, n]), transpose swaps the LAST TWO axes — the
// NumPy/JAX/PyTorch convention. This makes transpose well-defined per-
// atom for atom-batched matrices: shape=[N, m, n] with t='T' is
// observationally an atom-batched matrix where every atom's m×n slice
// is transposed to n×m.
function _dataShape(v: any) {
  if (v.shape.length < 2) return v.shape;
  if (!isTransposeView(v)) return v.shape;
  const out = v.shape.slice();
  const last = out.length - 1;
  const tmp = out[last];
  out[last] = out[last - 1];
  out[last - 1] = tmp;
  return out;
}

// Structural predicate: is `v` a Value-shaped object? Cheap check used
// by polymorphic dispatch sites (e.g. broadcast helpers in sampler.js)
// to distinguish Value inputs from bare JS numbers / Float64Arrays.
//
// Type predicate (`v is Value`) — TS narrows callers automatically:
//   if (valueLib.isValue(x)) { /* x is now typed as Value */ }
import type { Value as _ValueType } from './engine-types';
function isValue(v: any): v is _ValueType {
  return v != null && typeof v === 'object'
    && Array.isArray(v.shape)
    && v.data instanceof Float64Array;
}

// Is `v` batched along an outer axis of size N?
function isBatched(v: any, N: number) {
  return v.shape.length > 0 && v.shape[0] === N;
}

// =====================================================================
// Atom-axis canonical predicates (P3 — engine-concepts §2.1, §18, §20)
// =====================================================================
//
// "Atom-batched" = the value has a LEADING axis of size N = the engine
// atom count, i.e. each per-atom slice is `atomShape(v, N)`-shaped.
// Per the §2.1 shape contract every Value carries shape with the
// leading axis as the batch (atom) axis.
//
// Before P3 the engine had FOUR parallel predicates answering the same
// question — `_shapeAwareCandidate` (sampler-eval-batched.ts),
// `_classifyArg` (ops.ts), `_hasAtomAxis` (value-ops.ts), and inline
// `shape[0] === N` checks scattered across ~20 sites in 10 files.
// Each carried subtly different conventions about what "atom-batched"
// meant for shape=[N] vs shape=[N,k] vs raw Float64Array(N).
//
// The canonical contract:
//   - `isAtomBatched(v, N)` → boolean. True iff v is a Value or
//     Float64Array with leading dim of size N. When `v.outerRank` is
//     set (the producer signalled nested-vector semantics), require
//     `v.outerRank >= 1` so an iid output (outerRank=1, shape=[N,k])
//     is atom-batched but a per-atom matrix (no outerRank,
//     shape=[N,m,n]) is also atom-batched — both have the atom dim at
//     position 0 regardless. The outerRank tag only DISAMBIGUATES the
//     intrinsic structure of the per-atom slice; it doesn't gate the
//     atom-axis-presence check.
//   - `atomShape(v, N)` → number[] | null. Returns `v.shape.slice(1)`
//     when isAtomBatched, else null.
//
// These are PURELY shape-based, NOT name-based. Atom-batched
// detection by REF NAME (the safer convention used inside the
// aggregate body lifter — see sampler-aggregate.ts) still applies
// when a value coming through could be atom-indep with a coincidental
// shape[0] === N; that's a stricter check the call sites that need it
// continue to implement. `isAtomBatched(v, N)` answers "could this be
// atom-batched on the §2.1 contract"; the caller decides whether to
// promote that to "is".

function isAtomBatched(v: any, N: number): boolean {
  if (v == null) return false;
  // Raw Float64Array of length N — atom-batched scalar.
  if (v.BYTES_PER_ELEMENT !== undefined && v.length === N) return true;
  if (!isValue(v)) return false;
  if (v.shape.length === 0 || v.shape[0] !== N) return false;
  // If outerRank is set, it must include the atom axis (>= 1).
  // If unset, default behaviour: every leading axis is a loop axis,
  // so atom-batched is true.
  if (typeof v.outerRank === 'number' && v.outerRank < 1) return false;
  return true;
}

function atomShape(v: any, N: number): number[] | null {
  if (!isAtomBatched(v, N)) return null;
  if (v.BYTES_PER_ELEMENT !== undefined) return [];   // [N] Float64Array
  return (v.shape as number[]).slice(1);
}

// `isAtomBatchedScalar(v, N)` — the STRICTER atom-batched predicate
// used by the per-atom scalar broadcast paths. True iff `v` is
// atom-batched AND each per-atom slice is a bare scalar (i.e.
// `atomShape(v, N)` is `[]`). Equivalent to `isAtomBatched(v, N) &&
// (atomShape(v, N) || []).length === 0`, but skips the `.slice()` of
// the looser form for the hot path.
//
// Concretely true for:
//   - bare Float64Array of length N
//   - Value with shape=[N] (rank-1 atom-batched scalar)
// Returns false for shape=[N, k…] (per-atom vector/matrix) — those
// are atom-batched but NOT scalar atoms. Used by sampler-eval-
// batched's `isBatch` + complex argument accessors where the per-atom
// scalar-broadcast inner loops only handle rank-1 inputs.
function isAtomBatchedScalar(v: any, N: number): boolean {
  if (v == null) return false;
  if (v.BYTES_PER_ELEMENT !== undefined) return v.length === N;
  if (!isValue(v)) return false;
  if (v.shape.length !== 1 || v.shape[0] !== N) return false;
  if (typeof v.outerRank === 'number' && v.outerRank < 1) return false;
  return true;
}

// =====================================================================
// Nested-vector tag (engine-concepts §2.1, outerRank)
// =====================================================================
//
// Optional `Value.outerRank` field: number of LEADING axes that are
// "outer" / "loop" / "nested-collection" axes. The remaining
// trailing axes are the "inner" / "cell" axes — what each loop cell
// holds.
//
// When the tag is absent, the value is a flat tensor: every axis is
// a loop axis (cell = scalar). When set:
//
//   outerRank = 1, shape = [N]                    → flat vector of N scalars
//                                                   (equivalent to absent)
//   outerRank = 1, shape = [N, k]                 → nested vector:
//                                                   N inner vectors of length k
//   outerRank = 1, shape = [N, m, n]              → nested vector:
//                                                   N inner m×n matrices
//   outerRank = 2, shape = [N, M, k]              → 2-D nested vector:
//                                                   N×M outer loop axes,
//                                                   inner length-k vectors
//
// Per spec §03: "vectors of vectors are not interpreted as matrices
// implicitly". The outerRank tag is how the engine carries that
// distinction at the runtime-value level — without it, a Value with
// shape [N, k] could equally be a matrix (outerRank absent, every
// axis a loop axis from a broadcast's POV) OR a nested vector
// (outerRank=1, length-N collection of length-k vectors). Operations
// that care about the distinction (broadcast, type inference, matrix
// linear algebra) consult the tag.
function outerRankOf(v: any): number {
  if (!v || !Array.isArray(v.shape)) return 0;
  return (typeof v.outerRank === 'number') ? v.outerRank : v.shape.length;
}

// "Is this Value a nested vector?" — true iff outerRank is explicitly
// set AND less than shape.length (i.e. there are inner axes).
function isNestedVectorValue(v: any): boolean {
  return !!v && typeof v.outerRank === 'number'
    && Array.isArray(v.shape)
    && v.outerRank < v.shape.length;
}

// Inner-shape (cell-shape) of a tagged value. Returns the trailing
// axes after the outer-rank split. For a flat tensor (tag absent or
// outerRank == shape.length) this is `[]`.
function innerShapeOf(v: any): number[] {
  if (!isValue(v)) return [];
  const r = outerRankOf(v);
  return v.shape.slice(r);
}

// Outer-shape (loop-axis shape) of a tagged value. The leading
// `outerRank` entries of `shape`.
function outerShapeOf(v: any): number[] {
  if (!isValue(v)) return [];
  const r = outerRankOf(v);
  return v.shape.slice(0, r);
}

// Cheap helper used by linear-algebra ops that accept "vector of
// vectors" inputs (rowstack, colstack, matvec when the matrix is
// given as a vec-of-vecs, …). Detects either form:
//   (a) Legacy JS array of inner Values/scalars (host-shape form).
//   (b) Tagged nested-vector Value (outerRank=1, shape=[N, ...inner]).
//
// Returns `{ kind: 'js-array', items }` or `{ kind: 'nested-value',
// V }`, or `null` if `v` isn't a vector-of-vectors-shaped input.
function asVectorOfVectors(v: any): any {
  if (Array.isArray(v) && v.length > 0) {
    // JS-array form (legacy). Caller iterates `v[i]` directly.
    return { kind: 'js-array', items: v };
  }
  if (isValue(v) && isNestedVectorValue(v)) {
    return { kind: 'nested-value', V: v };
  }
  return null;
}

// Extract the i-th row of a vector-of-vectors as a Value (for
// tagged form) or whatever the JS-array's i-th element is (for
// legacy form). For tagged form: slice the inner shape out of
// the flat buffer; for JS-array form: return the element as-is.
function vovRowAt(vov: any, i: number): any {
  if (vov.kind === 'js-array') return vov.items[i];
  const V = vov.V;
  const inner = innerShapeOf(V);
  const innerLen = inner.reduce((a: number, b: number) => a * b, 1);
  const slice = new Float64Array(innerLen);
  for (let k = 0; k < innerLen; k++) slice[k] = V.data[i * innerLen + k];
  return { shape: inner.slice(), data: slice };
}

// Length (outer-axis size, i.e. number of inner vectors).
function vovLength(vov: any): number {
  if (vov.kind === 'js-array') return vov.items.length;
  return vov.V.shape[0];
}

// ---------------------------------------------------------------------
// Klein-4 tag operations: transpose / adjoint / conjugate
// ---------------------------------------------------------------------
//
// All three are lazy: they update the tag and (for matrices) swap the
// `shape` entries, but never touch the `data` buffer. Cost is O(rank)
// for the shape swap; constant for vectors and scalars.

// Swap the LAST TWO shape entries on a fresh array — used by transpose
// and adjoint for rank-≥2 values. Higher-rank tensors (e.g. atom-
// batched matrices shape=[N, m, n]) have their last two axes swapped,
// matching the NumPy/JAX/PyTorch convention. This means transpose is
// per-atom on atom-batched matrices: [N, m, n] → [N, n, m].
function _swappedShape(shape: number[]) {
  if (shape.length < 2) return shape.slice();
  const out = shape.slice();
  const last = out.length - 1;
  const tmp = out[last];
  out[last] = out[last - 1];
  out[last - 1] = tmp;
  return out;
}

// transpose(v): toggles the swapped bit; flips shape entries for
// rank-≥2 values. For scalars (rank 0) transpose is identity; for
// vectors (rank 1) transpose toggles the tag without changing shape.
function transpose(v: any) {
  if (!isValue(v)) throw new Error('transpose: argument is not a Value');
  const newTag = _TAG_TRANSPOSE[getTag(v)];
  const newShape = (v.shape.length >= 2) ? _swappedShape(v.shape) : v.shape.slice();
  const out: any = { shape: newShape, data: v.data, t: newTag };
  if (v.dtype) out.dtype = v.dtype;
  // Complex payload rides along untouched: the imaginary buffer shares
  // the same storage layout as `data`, so the lazy axis-swap applies to
  // it identically via _dataShape. The conjugation bit (in newTag) is
  // honoured at read time by readComplex, not here.
  if (v.im instanceof Float64Array) out.im = v.im;
  if (v.struct !== undefined) out.struct = _structTranspose(v.struct);
  return out;
}

// adjoint(v): toggles both bits (transpose + conjugate). Shape behaves
// the same as transpose. For real-valued data the conjugate bit is a
// no-op observationally but the tag tracks it for correctness once
// complex values arrive.
function adjoint(v: any) {
  if (!isValue(v)) throw new Error('adjoint: argument is not a Value');
  const newTag = _TAG_ADJOINT[getTag(v)];
  const newShape = (v.shape.length >= 2) ? _swappedShape(v.shape) : v.shape.slice();
  const out: any = { shape: newShape, data: v.data, t: newTag };
  if (v.dtype) out.dtype = v.dtype;
  if (v.im instanceof Float64Array) out.im = v.im;
  if (v.struct !== undefined) out.struct = _structTranspose(v.struct);
  return out;
}

// conjugate(v): toggles only the conjugate bit; shape unchanged.
// Real-valued no-op (still tag-tracked).
function conjugate(v: any) {
  if (!isValue(v)) throw new Error('conjugate: argument is not a Value');
  const newTag = _TAG_CONJUGATE[getTag(v)];
  const out: any = { shape: v.shape.slice(), data: v.data, t: newTag };
  if (v.dtype) out.dtype = v.dtype;
  if (v.im instanceof Float64Array) out.im = v.im;
  if (v.struct !== undefined) out.struct = v.struct;   // conj keeps structure
  return out;
}

// ---------------------------------------------------------------------
// Structured matrices (`struct` bitmask)
// ---------------------------------------------------------------------
//
// See the `struct` tag section above for the bit layout and algebra.
// v1 implements the `diag` storage form fully; the occupancy/refinement
// bits are defined and carried so producers (lower_cholesky → tri,
// row_gram → sym, …) and their fast-paths land as pure additions.

// Full bitmask (absent ⇒ dense; explicit 0 ⇒ all-zero matrix).
function getStruct(v: any) {
  return (v && v.struct !== undefined) ? v.struct : ST_DENSE;
}
// Occupancy sub-mask (low 3 bits).
function structOcc(v: any) { return getStruct(v) & ST_OCC_MASK; }

// Occupancy predicates. `isDiagStruct` is occupancy-diagonal-only; a
// diag Value is additionally vector-backed (data length = m, not m²) —
// `isDiagStored` is the storage-level check used before raw indexing.
function isDenseStruct(v: any) { return structOcc(v) === ST_DENSE; }
function isDiagStruct(v: any)  { return structOcc(v) === ST_DIAG; }
function isDiagStored(v: any) {
  return isValue(v) && ((v.struct ?? 0) & ST_OCC_MASK) === ST_DIAG
    && v.shape.length === 2 && v.shape[0] === v.shape[1]
    && v.data.length === v.shape[0];
}

// Diagonal-matrix constructor: logical m×m, but `data` stores only the
// m-vector diagonal (O(m) storage). Complex via parallel `im` (the
// diagonal of a complex diagonal matrix). This is the one vector-backed
// structured form (everything else is dense + flag in v1).
function diagMatrix(diagVec: any, imVec?: any) {
  const d = diagVec instanceof Float64Array ? diagVec : Float64Array.from(diagVec);
  const m = d.length;
  const out: any = { shape: [m, m], data: d, struct: ST_DIAG };
  if (imVec != null) {
    out.im = imVec instanceof Float64Array ? imVec : Float64Array.from(imVec);
    if (out.im.length !== m) {
      throw new Error('diagMatrix: im length ' + out.im.length +
                      ' != diagonal length ' + m);
    }
    out.dtype = 'complex';
  }
  return out;
}

// Materialize a structured Value to a plain dense Value (struct cleared,
// data length = numel(shape)). The single fallback every op without a
// structured fast-path calls — correctness never depends on a fast-path
// existing. Dense input is returned unchanged (no copy). Only the
// vector-backed `diag` form needs real expansion in v1; flagged-dense
// structures (tri/sym, later) just drop the flag.
function densify(v: any) {
  if (!isValue(v)) throw new Error('densify: argument is not a Value');
  if (v.struct === undefined || (v.struct & ST_OCC_MASK) === ST_DENSE) {
    return v;
  }
  if (isDiagStored(v)) {
    const m = v.shape[0];
    const data = new Float64Array(m * m);
    for (let i = 0; i < m; i++) data[i * m + i] = v.data[i];
    const out: any = { shape: [m, m], data: data };
    if (v.im instanceof Float64Array) {
      const im = new Float64Array(m * m);
      for (let i = 0; i < m; i++) im[i * m + i] = v.im[i];
      out.im = im;
      out.dtype = 'complex';
    }
    return out;                       // struct cleared ⇒ dense
  }
  // Flagged-dense structure (tri/sym in later versions): data is already
  // dense and full; just drop the structural flag. The implicit-zero /
  // implicit-unit-diagonal forms (strict/unit triangular) are not
  // produced yet, so no masking is required in v1.
  const out: any = { shape: v.shape.slice(), data: v.data };
  if (v.t && v.t !== 'N') out.t = v.t;
  if (v.dtype) out.dtype = v.dtype;
  if (v.im instanceof Float64Array) out.im = v.im;
  return out;
}

// ---------------------------------------------------------------------
// Complex Values (dtype: 'complex')
// ---------------------------------------------------------------------
//
// A complex Value is the planar generalization of the scalar `{re, im}`
// representation into the batched Value pipeline:
//
//   { shape, data: Float64Array, im: Float64Array, dtype: 'complex', t? }
//
// `data` holds the real parts, `im` the imaginary parts, in PARALLEL
// (planar) layout — NOT interleaved. Both buffers have identical length
// numel(shape) and share the same shape / storage-layout / transpose-tag
// semantics as a real Value. This matches TF.js's complex-tensor
// representation (separate real/imag tensors) and makes the structural
// constructors free:
//
//   - complex(re, im)  →  pair two existing buffers (no arithmetic)
//   - real(z)/imag(z)  →  relabel one buffer (no copy)
//   - conj(z)          →  lazy Klein-4 conjugate-bit flip (no copy)
//
// **Conjugation is lazy.** conjugate()/adjoint() only flip the tag bit;
// the stored `im` buffer is canonical (tag-'N' interpretation). Any
// consumer that needs the concrete logical imaginary part calls
// readComplex(v), which applies the conjugation sign once. Real Values
// keep paying nothing for the conj bit (no one reads "logical im").

// Is `v` a complex Value? Cheap structural check used by op dispatch to
// route the (re, im) buffer-wise algebra. Type predicate narrows callers
// so `v.im` reads after `if (isComplexValue(v))` need no `!` assertion.
type _ComplexValue = _ValueType & { dtype: 'complex'; im: Float64Array };
function isComplexValue(v: any): v is _ComplexValue {
  return v != null && typeof v === 'object'
    && v.dtype === 'complex'
    && Array.isArray(v.shape)
    && v.data instanceof Float64Array
    && v.im instanceof Float64Array;
}

// Raw stored imaginary buffer (tag-'N' interpretation), or null for a
// real Value. Prefer readComplex when you need the logical value.
function getImag(v: any) { return (v && v.im instanceof Float64Array) ? v.im : null; }

// Complex Value constructor. `re` and `im` are array-likes of equal
// length; `shape` defaults to [length] (atom-batched scalar / vector,
// disambiguated by the caller's N exactly like the real constructors).
// Borrows Float64Array storage when given; copies otherwise.
function complexValue(re: any, im: any, shape?: any) {
  const reD = re instanceof Float64Array ? re : Float64Array.from(re);
  const imD = im instanceof Float64Array ? im : Float64Array.from(im);
  if (reD.length !== imD.length) {
    throw new Error('complexValue: re length ' + reD.length +
                    ' != im length ' + imD.length);
  }
  const shp = shape ? shape.slice() : [reD.length];
  if (numel(shp) !== reD.length) {
    throw new Error('complexValue: numel(' + JSON.stringify(shp) +
                    ') != buffer length ' + reD.length);
  }
  return { shape: shp, data: reD, im: imD, dtype: 'complex' };
}

// Materialize the LOGICAL complex parts of `v`, honouring the Klein-4
// conjugation bit: returns `{ re, im }` Float64Arrays where `im` is the
// stored buffer negated iff the value is a conjugate view. `re` is
// always the stored buffer (conjugation never touches the real part).
// The negation allocates a fresh buffer only when conjugated — the lazy
// conj cost is realized here, at the one consumer that needs it. For a
// real Value the imaginary part is an implicit zero buffer.
function readComplex(v: any) {
  if (!isComplexValue(v)) {
    return { re: v.data, im: new Float64Array(v.data.length) };
  }
  if (!isConjugateView(v)) return { re: v.data, im: v.im };
  const im = new Float64Array(v.im.length);
  for (let i = 0; i < im.length; i++) im[i] = -v.im[i];
  return { re: v.data, im: im };
}

// ---------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------

// Atom-indep scalar. `x` must be a JS number (or coercible).
function scalar(x: any) {
  const data = new Float64Array(1);
  data[0] = +x;
  return { shape: [], data: data };
}

// Atom-batched scalar from an array-like of length N. Borrows
// Float64Array storage when given; otherwise copies into one.
function batchedScalar(arr: any) {
  const data = arr instanceof Float64Array ? arr : Float64Array.from(arr);
  return { shape: [data.length], data: data };
}

// Atom-indep vector (length k). Returned shape=[k].
function vector(data: any) {
  const out = data instanceof Float64Array ? data : Float64Array.from(data);
  return { shape: [out.length], data: out };
}

// Atom-batched vector. `flatData` length must be N*k; shape=[N, k].
function batchedVector(flatData: any, k: number) {
  const data = flatData instanceof Float64Array ? flatData : Float64Array.from(flatData);
  if (data.length % k !== 0) {
    throw new Error('batchedVector: data length ' + data.length + ' not divisible by k=' + k);
  }
  const N = data.length / k;
  return { shape: [N, k], data: data };
}

// Atom-indep matrix m×n. flatData row-major; length must equal m*n.
function matrix(flatData: any, m: number, n: number) {
  const data = flatData instanceof Float64Array ? flatData : Float64Array.from(flatData);
  if (data.length !== m * n) {
    throw new Error('matrix: data length ' + data.length + ' != m*n = ' + (m * n));
  }
  return { shape: [m, n], data: data };
}

// Atom-batched matrix. flatData atom-major over row-major slices.
// length must be N*m*n.
function batchedMatrix(flatData: any, m: number, n: number) {
  const data = flatData instanceof Float64Array ? flatData : Float64Array.from(flatData);
  if (data.length % (m * n) !== 0) {
    throw new Error('batchedMatrix: data length ' + data.length +
                    ' not divisible by m*n = ' + (m * n));
  }
  const N = data.length / (m * n);
  return { shape: [N, m, n], data: data };
}

// Generic shape constructor — last resort when none of the typed
// constructors fit. Validates that data length matches numel(shape).
function withShape(flatData: any, shape: number[]) {
  const data = flatData instanceof Float64Array ? flatData : Float64Array.from(flatData);
  const expected = numel(shape);
  if (data.length !== expected) {
    throw new Error('withShape: data length ' + data.length +
                    ' != numel(' + JSON.stringify(shape) + ') = ' + expected);
  }
  return { shape: shape.slice(), data: data };
}

// ---------------------------------------------------------------------
// Coercions
// ---------------------------------------------------------------------

// Canonical Value form. Accepts:
//   - Value                                    → returned as-is
//   - JS number (or finite-coercible primitive)→ scalar
//   - Float64Array                             → shape=[length]
//   - typed array (other)                      → shape=[length] after copy to F64
//   - nested JS Array                          → row-major shape inferred
function asValue(x: any): any {
  if (x === null || x === undefined) {
    throw new Error('asValue: null/undefined is not a value');
  }
  // Already a Value? Cheap structural check — `shape` is an Array
  // and `data` is a Float64Array.
  if (typeof x === 'object' &&
      Array.isArray(x.shape) &&
      x.data instanceof Float64Array) {
    return x;
  }
  if (typeof x === 'number' || typeof x === 'boolean') {
    return scalar(+x);
  }
  if (x instanceof Float64Array) {
    return { shape: [x.length], data: x };
  }
  // Other typed arrays: copy through Float64Array.
  if (ArrayBuffer.isView(x) && typeof (x as any).length === 'number') {
    const data = Float64Array.from(x as any);
    return { shape: [data.length], data: data };
  }
  if (Array.isArray(x)) {
    const shape = inferShapeFromNested(x);
    const data = new Float64Array(numel(shape));
    flattenNested(x, data, 0, 0, shape);
    return { shape: shape, data: data };
  }
  throw new Error('asValue: cannot coerce ' + typeof x + ' to Value');
}

// Extract a JS number from a shape=[] Value. Strict: anything else
// throws. Use for engine-API-boundary conversions where the caller has
// asserted scalarity.
function asScalar(v: any) {
  if (!v || !Array.isArray(v.shape) || !(v.data instanceof Float64Array)) {
    throw new Error('asScalar: argument is not a Value');
  }
  if (v.shape.length !== 0) {
    throw new Error('asScalar: shape is [' + v.shape.join(',') + '], expected []');
  }
  return v.data[0];
}

// Extract the underlying Float64Array(N) of a shape=[N] batched scalar.
// Strict: any other shape (including [] or [N, k]) throws.
function asBatch(v: any, N: number) {
  if (!v || !Array.isArray(v.shape) || !(v.data instanceof Float64Array)) {
    throw new Error('asBatch: argument is not a Value');
  }
  if (v.shape.length !== 1 || v.shape[0] !== N) {
    throw new Error('asBatch: shape is [' + v.shape.join(',') + '], expected [' + N + ']');
  }
  return v.data;
}

module.exports = {
  // accessors
  getShape: getShape,
  getData: getData,
  getDType: getDType,
  getTag: getTag,
  isTransposeView: isTransposeView,
  isConjugateView: isConjugateView,
  isValue: isValue,
  isComplexValue: isComplexValue,
  getImag: getImag,
  isBatched: isBatched,
  isAtomBatched: isAtomBatched,
  isAtomBatchedScalar: isAtomBatchedScalar,
  atomShape: atomShape,
  numel: numel,
  // outerRank / nested-vector tag (engine-concepts §2.1)
  outerRankOf: outerRankOf,
  isNestedVectorValue: isNestedVectorValue,
  innerShapeOf: innerShapeOf,
  outerShapeOf: outerShapeOf,
  asVectorOfVectors: asVectorOfVectors,
  vovRowAt: vovRowAt,
  vovLength: vovLength,
  // structured-matrix tag (orthogonal to the Klein-4 tag)
  ST_LOWER: ST_LOWER, ST_DIAG: ST_DIAG, ST_UPPER: ST_UPPER,
  ST_UNIT: ST_UNIT, ST_SYM: ST_SYM, ST_POSDEF: ST_POSDEF,
  ST_OCC_MASK: ST_OCC_MASK, ST_DENSE: ST_DENSE,
  getStruct: getStruct,
  structOcc: structOcc,
  isDenseStruct: isDenseStruct,
  isDiagStruct: isDiagStruct,
  isDiagStored: isDiagStored,
  diagMatrix: diagMatrix,
  densify: densify,
  // tag-flipping operations (lazy; never touch data)
  transpose: transpose,
  adjoint: adjoint,
  conjugate: conjugate,
  // constructors
  scalar: scalar,
  batchedScalar: batchedScalar,
  vector: vector,
  batchedVector: batchedVector,
  matrix: matrix,
  batchedMatrix: batchedMatrix,
  withShape: withShape,
  complexValue: complexValue,
  readComplex: readComplex,
  // coercions
  asValue: asValue,
  asScalar: asScalar,
  asBatch: asBatch,
};
