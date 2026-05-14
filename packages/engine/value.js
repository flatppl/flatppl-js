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
// Design choices (see TODO-flatppl-js.md Phase 0 + ARCHITECTURE.md
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
const _TAG_TRANSPOSE = { N: 'T', T: 'N', A: 'C', C: 'A' };
const _TAG_ADJOINT   = { N: 'A', T: 'C', A: 'N', C: 'T' };
const _TAG_CONJUGATE = { N: 'C', T: 'A', A: 'T', C: 'N' };
// Effective bits (used by consumers that need to know "should I treat
// the data as transposed?" without caring about conjugation).
const _TAG_SWAPPED    = { N: false, T: true,  A: true,  C: false };
const _TAG_CONJUGATED = { N: false, T: false, A: true,  C: true  };

// ---------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------

// Number of elements implied by a shape array. `[]` (scalar) → 1.
function numel(shape) {
  let n = 1;
  for (let i = 0; i < shape.length; i++) n *= shape[i];
  return n;
}

// Infer shape from a nested JS Array. Validates rectangularity at every
// level; throws on ragged structure. Empty array → shape=[0]; nested
// empties (e.g. [[]]) → shape=[1, 0].
function inferShapeFromNested(arr) {
  const shape = [];
  let cur = arr;
  while (Array.isArray(cur)) {
    shape.push(cur.length);
    if (cur.length === 0) break;
    // Check rectangularity at this level.
    const first = cur[0];
    if (Array.isArray(first)) {
      const len = first.length;
      for (let i = 1; i < cur.length; i++) {
        const sib = cur[i];
        if (!Array.isArray(sib) || sib.length !== len) {
          throw new Error('asValue: nested array is ragged at depth ' + shape.length);
        }
      }
    } else {
      for (let i = 1; i < cur.length; i++) {
        if (Array.isArray(cur[i])) {
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
function flattenNested(arr, out, offset, depth, shape) {
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

function getShape(v) { return v.shape; }
function getData(v)  { return v.data; }
function getDType(v) { return v.dtype || DEFAULT_DTYPE; }
function getTag(v)   { return v.t || DEFAULT_TAG; }

// Tag bit extractors. Use these in dispatch code rather than equality
// against the four state strings — they make "is this view swapped?"
// and "is this view conjugated?" clear at the call site.
function isTransposeView(v) { return _TAG_SWAPPED[getTag(v)]; }
function isConjugateView(v) { return _TAG_CONJUGATED[getTag(v)]; }

// Storage shape — the dimensions of the underlying Float64Array as laid
// out in memory, before applying the transpose tag. For a Value with
// shape=[m, n] and t='T', the stored data is laid out [n, m] row-major.
// For vectors and scalars (1-D or 0-D), the storage shape equals shape.
function _dataShape(v) {
  if (v.shape.length < 2) return v.shape;
  if (!isTransposeView(v)) return v.shape;
  const out = new Array(v.shape.length);
  for (let i = 0; i < v.shape.length; i++) {
    out[i] = v.shape[v.shape.length - 1 - i];
  }
  return out;
}

// Structural predicate: is `v` a Value-shaped object? Cheap check used
// by polymorphic dispatch sites (e.g. broadcast helpers in sampler.js)
// to distinguish Value inputs from bare JS numbers / Float64Arrays.
function isValue(v) {
  return v != null && typeof v === 'object'
    && Array.isArray(v.shape)
    && v.data instanceof Float64Array;
}

// Is `v` batched along an outer axis of size N?
function isBatched(v, N) {
  return v.shape.length > 0 && v.shape[0] === N;
}

// ---------------------------------------------------------------------
// Klein-4 tag operations: transpose / adjoint / conjugate
// ---------------------------------------------------------------------
//
// All three are lazy: they update the tag and (for matrices) swap the
// `shape` entries, but never touch the `data` buffer. Cost is O(rank)
// for the shape swap; constant for vectors and scalars.

// Swap shape entries in place on a fresh array — used by transpose and
// adjoint for rank-≥2 values.
function _swappedShape(shape) {
  if (shape.length < 2) return shape.slice();
  const out = new Array(shape.length);
  for (let i = 0; i < shape.length; i++) out[i] = shape[shape.length - 1 - i];
  return out;
}

// transpose(v): toggles the swapped bit; flips shape entries for
// rank-≥2 values. For scalars (rank 0) transpose is identity; for
// vectors (rank 1) transpose toggles the tag without changing shape.
function transpose(v) {
  if (!isValue(v)) throw new Error('transpose: argument is not a Value');
  const newTag = _TAG_TRANSPOSE[getTag(v)];
  const newShape = (v.shape.length >= 2) ? _swappedShape(v.shape) : v.shape.slice();
  const out = { shape: newShape, data: v.data, t: newTag };
  if (v.dtype) out.dtype = v.dtype;
  return out;
}

// adjoint(v): toggles both bits (transpose + conjugate). Shape behaves
// the same as transpose. For real-valued data the conjugate bit is a
// no-op observationally but the tag tracks it for correctness once
// complex values arrive.
function adjoint(v) {
  if (!isValue(v)) throw new Error('adjoint: argument is not a Value');
  const newTag = _TAG_ADJOINT[getTag(v)];
  const newShape = (v.shape.length >= 2) ? _swappedShape(v.shape) : v.shape.slice();
  const out = { shape: newShape, data: v.data, t: newTag };
  if (v.dtype) out.dtype = v.dtype;
  return out;
}

// conjugate(v): toggles only the conjugate bit; shape unchanged.
// Real-valued no-op (still tag-tracked).
function conjugate(v) {
  if (!isValue(v)) throw new Error('conjugate: argument is not a Value');
  const newTag = _TAG_CONJUGATE[getTag(v)];
  const out = { shape: v.shape.slice(), data: v.data, t: newTag };
  if (v.dtype) out.dtype = v.dtype;
  return out;
}

// ---------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------

// Atom-indep scalar. `x` must be a JS number (or coercible).
function scalar(x) {
  const data = new Float64Array(1);
  data[0] = +x;
  return { shape: [], data: data };
}

// Atom-batched scalar from an array-like of length N. Borrows
// Float64Array storage when given; otherwise copies into one.
function batchedScalar(arr) {
  const data = arr instanceof Float64Array ? arr : Float64Array.from(arr);
  return { shape: [data.length], data: data };
}

// Atom-indep vector (length k). Returned shape=[k].
function vector(data) {
  const out = data instanceof Float64Array ? data : Float64Array.from(data);
  return { shape: [out.length], data: out };
}

// Atom-batched vector. `flatData` length must be N*k; shape=[N, k].
function batchedVector(flatData, k) {
  const data = flatData instanceof Float64Array ? flatData : Float64Array.from(flatData);
  if (data.length % k !== 0) {
    throw new Error('batchedVector: data length ' + data.length + ' not divisible by k=' + k);
  }
  const N = data.length / k;
  return { shape: [N, k], data: data };
}

// Atom-indep matrix m×n. flatData row-major; length must equal m*n.
function matrix(flatData, m, n) {
  const data = flatData instanceof Float64Array ? flatData : Float64Array.from(flatData);
  if (data.length !== m * n) {
    throw new Error('matrix: data length ' + data.length + ' != m*n = ' + (m * n));
  }
  return { shape: [m, n], data: data };
}

// Atom-batched matrix. flatData atom-major over row-major slices.
// length must be N*m*n.
function batchedMatrix(flatData, m, n) {
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
function withShape(flatData, shape) {
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
function asValue(x) {
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
  if (ArrayBuffer.isView(x) && typeof x.length === 'number') {
    const data = Float64Array.from(x);
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
function asScalar(v) {
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
function asBatch(v, N) {
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
  isBatched: isBatched,
  numel: numel,
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
  // coercions
  asValue: asValue,
  asScalar: asScalar,
  asBatch: asBatch,
};
