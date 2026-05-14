'use strict';

// =====================================================================
// value-ops — shape-aware arithmetic primitives over Value
// =====================================================================
//
// Implements the shape-dispatched arithmetic the spec requires:
//
//   - mul:  scalar/scalar, scalar/array (broadcast), matrix/matrix,
//           matrix/vector, vector/transpose(vector) (outer),
//           transpose(vector)/vector (inner). vector*vector is an
//           error per spec §07 wording + design clarification.
//   - add / sub: scalar/scalar, elementwise on arrays of same shape,
//                with scalar broadcast in either direction.
//   - neg:  pointwise negation; shape and tag preserved.
//
// All operations consume Values and produce Values. The Klein-4
// transpose/adjoint tag (`v.t ∈ {'N','T','A','C'}`, see value.js) is
// honoured throughout: matmul/matvec read with index permutation
// according to the operand tags (BLAS gemm-flag style — no
// materialisation of transposes), inner/outer dispatch on vector
// orientation, and tag is preserved through scalar broadcast.
//
// For real-valued data (dtype='f64') the conjugate bit is observation-
// ally a no-op, but it is plumbed through compositions so the algebra
// stays correct once complex dtypes arrive: conjugate-aware reads (in
// matmul, inner-product, etc.) would call a complex-aware multiply
// instead of a plain `*` — the existing dispatch sites are the
// extension points.
//
// =====================================================================

const valueLib = require('./value');
const {
  isValue, getTag, isTransposeView, isConjugateView,
  scalar, batchedScalar, vector, withShape,
} = valueLib;

// ---------------------------------------------------------------------
// Indexing helpers
// ---------------------------------------------------------------------
// For a matrix Value with logical shape [m, n] the underlying
// Float64Array layout depends on the tag's swapped bit:
//   - swapped=false (tag N or C): data is row-major [m × n].
//     logical (i, k) lives at data[i*n + k].
//   - swapped=true  (tag T or A): data is row-major in the
//     pre-transpose shape [n × m].
//     logical (i, k) lives at data[k*m + i].
//
// These two helpers compute the linear index for a logical (i, j)
// position in O(1) without allocation.

function _matIdxN(i, j, n) { return i * n + j; }
function _matIdxT(i, j, m) { return j * m + i; }

// ---------------------------------------------------------------------
// mul — shape-dispatched multiplication
// ---------------------------------------------------------------------

function mul(a, b) {
  if (!isValue(a) || !isValue(b)) {
    throw new Error('value-ops.mul: both operands must be Values');
  }
  const sa = a.shape, sb = b.shape;
  // scalar × anything
  if (sa.length === 0) return _scalarBroadcastMul(a.data[0], b);
  if (sb.length === 0) return _scalarBroadcastMul(b.data[0], a);
  // vector × vector
  if (sa.length === 1 && sb.length === 1) return _vecVecMul(a, b);
  // matrix × vector
  if (sa.length === 2 && sb.length === 1) return _matVecMul(a, b);
  // vector × matrix
  if (sa.length === 1 && sb.length === 2) return _vecMatMul(a, b);
  // matrix × matrix
  if (sa.length === 2 && sb.length === 2) return _matMatMul(a, b);
  throw new Error(
    'value-ops.mul: unsupported shape combination ' +
    JSON.stringify(sa) + ' × ' + JSON.stringify(sb)
  );
}

// scalar (JS number) × Value (any shape) → Value with same shape and tag.
function _scalarBroadcastMul(s, v) {
  const out = new Float64Array(v.data.length);
  for (let i = 0; i < v.data.length; i++) out[i] = s * v.data[i];
  const r = { shape: v.shape.slice(), data: out };
  if (v.t && v.t !== 'N') r.t = v.t;  // preserve orientation
  if (v.dtype) r.dtype = v.dtype;
  return r;
}

// vector × vector → scalar (inner) or matrix (outer), depending on tags.
function _vecVecMul(u, v) {
  const uSwapped = isTransposeView(u);
  const vSwapped = isTransposeView(v);
  // Klein-4 vec×vec rules (real-valued; conjugation no-op):
  //   N × N → error (column × column undefined)
  //   T × N → scalar  (row × column = inner product, requires same length)
  //   N × T → matrix  (column × row = outer product, lengths independent)
  //   T × T → error
  if (uSwapped && !vSwapped) return _innerProduct(u, v);
  if (!uSwapped && vSwapped) return _outerProduct(u, v);
  if (!uSwapped && !vSwapped) {
    throw new Error(
      'mul: vector * vector is not defined; use transpose(v1) * v2 ' +
      'for inner product or v1 * transpose(v2) for outer product');
  }
  throw new Error(
    'mul: transpose(v1) * transpose(v2) is not defined (two row vectors)');
}

// Inner product: u (row, tag T or A) × v (column, tag N or C) → scalar.
// Both vectors must have the same length.
// For complex (when implemented) the row's conjugate bit determines
// whether to conjugate u's entries on read.
function _innerProduct(u, v) {
  const k = u.shape[0];
  if (v.shape[0] !== k) {
    throw new Error('mul: inner-product vector length mismatch (' +
      k + ' vs ' + v.shape[0] + ')');
  }
  let s = 0;
  for (let i = 0; i < k; i++) s += u.data[i] * v.data[i];
  return scalar(s);
}

// Outer product: u (column, tag N or C) × v (row, tag T or A) → matrix [m, n].
function _outerProduct(u, v) {
  const m = u.shape[0], n = v.shape[0];
  const out = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const ui = u.data[i];
    for (let j = 0; j < n; j++) out[i * n + j] = ui * v.data[j];
  }
  return { shape: [m, n], data: out };
}

// matrix(m, n) × vector(n) → vector(m).
// Matrix tag may be any of {N,T,A,C}; vector must be column-oriented
// (tag N or C; transposed/row vectors aren't valid right operands of
// matrix multiplication per spec §07).
function _matVecMul(A, v) {
  const [m, n] = A.shape;
  if (v.shape[0] !== n) {
    throw new Error(
      'mul: matrix×vector dimension mismatch (' +
      JSON.stringify(A.shape) + ' × [' + v.shape[0] + '])');
  }
  if (isTransposeView(v)) {
    throw new Error(
      'mul: matrix * (transposed/row vector) is not defined; ' +
      'mul requires a column vector on the right');
  }
  const aSwapped = isTransposeView(A);
  const out = new Float64Array(m);
  if (!aSwapped) {
    // A in row-major (m × n).
    for (let i = 0; i < m; i++) {
      let s = 0;
      const row = i * n;
      for (let k = 0; k < n; k++) s += A.data[row + k] * v.data[k];
      out[i] = s;
    }
  } else {
    // A stored as [n × m] row-major (since logical [m, n] with t='T');
    // logical (i, k) = data[k*m + i].
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += A.data[k * m + i] * v.data[k];
      out[i] = s;
    }
  }
  return { shape: [m], data: out };
}

// vector(k) × matrix(k, p) → vector(p). Only valid if the vector is
// row-oriented (tag T or A). Result is a row vector (tag T).
function _vecMatMul(u, B) {
  if (!isTransposeView(u)) {
    throw new Error(
      'mul: (column vector) * matrix is not defined; ' +
      'mul requires matrix on the left of a column vector or ' +
      'transpose(v) on the left of a matrix');
  }
  const k = u.shape[0];
  const [bRows, p] = B.shape;
  if (bRows !== k) {
    throw new Error(
      'mul: vector×matrix dimension mismatch ([' + k + '] × ' +
      JSON.stringify(B.shape) + ')');
  }
  const bSwapped = isTransposeView(B);
  const out = new Float64Array(p);
  if (!bSwapped) {
    // B in row-major (k × p).
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < k; i++) s += u.data[i] * B.data[i * p + j];
      out[j] = s;
    }
  } else {
    // B stored as [p × k] row-major; logical (i, j) = data[j*k + i].
    for (let j = 0; j < p; j++) {
      let s = 0;
      const base = j * k;
      for (let i = 0; i < k; i++) s += u.data[i] * B.data[base + i];
      out[j] = s;
    }
  }
  // Result is a row vector — tag T (or A if u was A, since conjugation
  // commutes with the structural row/col identity).
  const tagOut = (getTag(u) === 'A') ? 'A' : 'T';
  return { shape: [p], data: out, t: tagOut };
}

// matrix(m, n) × matrix(n, p) → matrix(m, p). Tags read via index
// permutation (BLAS gemm-flag style); output is canonical (tag N).
function _matMatMul(A, B) {
  const [m, n] = A.shape;
  const [bRows, p] = B.shape;
  if (bRows !== n) {
    throw new Error(
      'mul: matrix×matrix dimension mismatch (' +
      JSON.stringify(A.shape) + ' × ' + JSON.stringify(B.shape) + ')');
  }
  const aSwap = isTransposeView(A);
  const bSwap = isTransposeView(B);
  const out = new Float64Array(m * p);
  // Inner-loop indexing functions: pick per-operand based on tag once.
  // (Branching inside the i,j,k loop would dominate small-matrix
  // benchmarks; this version branches once at the top.)
  if (!aSwap && !bSwap) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += A.data[i * n + k] * B.data[k * p + j];
        out[i * p + j] = s;
      }
    }
  } else if (aSwap && !bSwap) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += A.data[k * m + i] * B.data[k * p + j];
        out[i * p + j] = s;
      }
    }
  } else if (!aSwap && bSwap) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += A.data[i * n + k] * B.data[j * n + k];
        out[i * p + j] = s;
      }
    }
  } else {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += A.data[k * m + i] * B.data[j * n + k];
        out[i * p + j] = s;
      }
    }
  }
  return { shape: [m, p], data: out };
}

module.exports = {
  mul,
  // Exposed for direct use / test access; the public `mul` already
  // covers every dispatch path.
  _innerProduct,
  _outerProduct,
  _matVecMul,
  _vecMatMul,
  _matMatMul,
  _scalarBroadcastMul,
  _matIdxN,
  _matIdxT,
};
