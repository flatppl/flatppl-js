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
  isComplexValue, readComplex, complexValue,
  scalar, batchedScalar, vector, withShape,
} = valueLib;

// ---------------------------------------------------------------------
// Complex elementwise helpers
// ---------------------------------------------------------------------
//
// Complex Values are planar: parallel re (`.data`) / im (`.im`) buffers
// with identical shape + layout (see value.js). All elementwise ops
// therefore reduce to running the real algebra over the re buffers and
// the matching imaginary algebra over the im buffers, with shapes
// already validated by the shared shape checks. readComplex applies the
// Klein-4 conjugation bit once, so downstream cell math never re-checks
// the tag.
//
// `_isCx(a, b)` — does this op need the complex path?
function _isCx(a, b) {
  return isComplexValue(a) || (b !== undefined && isComplexValue(b));
}

// Pack two equal-length re/im buffers back into a complex Value with the
// given logical shape. readComplex already resolved any input
// conjugation into the buffers, so the result carries no conj bit; but
// it leaves the data in STORED (pre-transpose) layout, so the swapped
// bit must be carried forward when the governing operand was a
// transpose view. `swapped` true ⇒ tag 'T' (pure transpose, conj
// already folded), else canonical 'N'.
function _packCx(re, im, shape, swapped) {
  const v = complexValue(re, im, shape);
  if (swapped) v.t = 'T';
  return v;
}

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
  if (_isCx(a, b)) {
    // Conjugation-aware complex gemm. readComplex (inside each helper)
    // folds the Klein-4 conj bit, so adjoint(·) gives the Hermitian
    // form for free; the swapped bit still drives index permutation.
    if (sa.length === 0 || sb.length === 0) return _complexScalarBroadcastMul(a, b);
    if (sa.length === 1 && sb.length === 1) return _cxVecVecMul(a, b);
    if (sa.length === 2 && sb.length === 1) return _cxMatVecMul(a, b);
    if (sa.length === 1 && sb.length === 2) return _cxVecMatMul(a, b);
    if (sa.length === 2 && sb.length === 2) return _cxMatMatMul(a, b);
    throw new Error(
      'value-ops.mul: unsupported complex shape combination ' +
      JSON.stringify(sa) + ' × ' + JSON.stringify(sb));
  }
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

// Complex scalar broadcast multiply: one operand is shape=[] (a complex
// or real scalar), the other any shape. Full complex multiply per cell
// (the scalar's imaginary part participates). Multiplication is
// commutative, so the re/im formula is order-independent. The array
// operand governs output shape and transpose orientation; conjugation
// of both operands is resolved by readComplex up front.
function _complexScalarBroadcastMul(a, b) {
  const aScalar = a.shape.length === 0;
  const scalarV = aScalar ? a : b;
  const arrV    = aScalar ? b : a;
  const s = readComplex(scalarV);
  const w = readComplex(arrV);
  const sr = s.re[0], si = s.im[0];
  const n = w.re.length;
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const wr = w.re[i], wi = w.im[i];
    re[i] = sr * wr - si * wi;
    im[i] = sr * wi + si * wr;
  }
  return _packCx(re, im, arrV.shape, isTransposeView(arrV));
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

// =====================================================================
// Conjugation-aware complex gemm (vec/mat products)
// =====================================================================
//
// The real product helpers above read raw `.data` and apply the
// Klein-4 swapped bit via index permutation (BLAS gemm-flag style).
// The complex variants below are structurally identical but:
//
//   - operands are read through readComplex(), which folds the
//     conjugation bit into the logical buffers ONCE. This is exactly
//     what makes the Hermitian forms fall out for free:
//       transpose(v) * v  → tag T (no conj) → bilinear  vᵀv
//       adjoint(v)   * v  → tag A (conj)    → sesquilin. v̄ᵀv
//     No explicit conjugation logic is needed here — the lazy conj
//     design (value.js) already did it.
//   - the swapped bit still drives index permutation (readComplex
//     leaves data in STORED layout, identical to the real path).
//   - accumulation is the complex MAC (ar·br − ai·bi, ar·bi + ai·br).
//   - results are canonical (conj folded into data ⇒ tag-'N' conj
//     bit); _vecMat returns a row vector with the swapped bit set.
//
// Mixed real×complex works transparently: readComplex on a real Value
// yields a zero imaginary buffer.

function _cxVecVecMul(u, v) {
  const uSwapped = isTransposeView(u);
  const vSwapped = isTransposeView(v);
  if (uSwapped && !vSwapped) return _cxInnerProduct(u, v);
  if (!uSwapped && vSwapped) return _cxOuterProduct(u, v);
  if (!uSwapped && !vSwapped) {
    throw new Error(
      'mul: vector * vector is not defined; use transpose(v1) * v2 ' +
      'for inner product or v1 * transpose(v2) for outer product');
  }
  throw new Error(
    'mul: transpose(v1) * transpose(v2) is not defined (two row vectors)');
}

function _cxInnerProduct(u, v) {
  const k = u.shape[0];
  if (v.shape[0] !== k) {
    throw new Error('mul: inner-product vector length mismatch (' +
      k + ' vs ' + v.shape[0] + ')');
  }
  const cu = readComplex(u), cv = readComplex(v);
  let sr = 0, si = 0;
  for (let i = 0; i < k; i++) {
    const ar = cu.re[i], ai = cu.im[i], br = cv.re[i], bi = cv.im[i];
    sr += ar * br - ai * bi;
    si += ar * bi + ai * br;
  }
  return _packCx([sr], [si], [], false);
}

function _cxOuterProduct(u, v) {
  const m = u.shape[0], n = v.shape[0];
  const cu = readComplex(u), cv = readComplex(v);
  const re = new Float64Array(m * n), im = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const ar = cu.re[i], ai = cu.im[i];
    for (let j = 0; j < n; j++) {
      const br = cv.re[j], bi = cv.im[j];
      re[i * n + j] = ar * br - ai * bi;
      im[i * n + j] = ar * bi + ai * br;
    }
  }
  return _packCx(re, im, [m, n], false);
}

function _cxMatVecMul(A, v) {
  const [m, n] = A.shape;
  if (v.shape[0] !== n) {
    throw new Error('mul: matrix×vector dimension mismatch (' +
      JSON.stringify(A.shape) + ' × [' + v.shape[0] + '])');
  }
  if (isTransposeView(v)) {
    throw new Error('mul: matrix * (transposed/row vector) is not ' +
      'defined; mul requires a column vector on the right');
  }
  const cA = readComplex(A), cv = readComplex(v);
  const aSwap = isTransposeView(A);
  const re = new Float64Array(m), im = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let sr = 0, si = 0;
    for (let k = 0; k < n; k++) {
      const idx = aSwap ? (k * m + i) : (i * n + k);
      const ar = cA.re[idx], ai = cA.im[idx];
      const br = cv.re[k],   bi = cv.im[k];
      sr += ar * br - ai * bi;
      si += ar * bi + ai * br;
    }
    re[i] = sr; im[i] = si;
  }
  return _packCx(re, im, [m], false);
}

function _cxVecMatMul(u, B) {
  if (!isTransposeView(u)) {
    throw new Error('mul: (column vector) * matrix is not defined; ' +
      'mul requires matrix on the left of a column vector or ' +
      'transpose(v) on the left of a matrix');
  }
  const k = u.shape[0];
  const [bRows, p] = B.shape;
  if (bRows !== k) {
    throw new Error('mul: vector×matrix dimension mismatch ([' + k +
      '] × ' + JSON.stringify(B.shape) + ')');
  }
  const cu = readComplex(u), cB = readComplex(B);
  const bSwap = isTransposeView(B);
  const re = new Float64Array(p), im = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let sr = 0, si = 0;
    for (let i = 0; i < k; i++) {
      const idx = bSwap ? (j * k + i) : (i * p + j);
      const ar = cu.re[i], ai = cu.im[i];
      const br = cB.re[idx], bi = cB.im[idx];
      sr += ar * br - ai * bi;
      si += ar * bi + ai * br;
    }
    re[j] = sr; im[j] = si;
  }
  // Row vector: swapped bit set. Conjugation already folded into data
  // by readComplex, so the result is a pure transpose view (tag 'T'),
  // not 'A'.
  return _packCx(re, im, [p], true);
}

function _cxMatMatMul(A, B) {
  const [m, n] = A.shape;
  const [bRows, p] = B.shape;
  if (bRows !== n) {
    throw new Error('mul: matrix×matrix dimension mismatch (' +
      JSON.stringify(A.shape) + ' × ' + JSON.stringify(B.shape) + ')');
  }
  const cA = readComplex(A), cB = readComplex(B);
  const aSwap = isTransposeView(A), bSwap = isTransposeView(B);
  const re = new Float64Array(m * p), im = new Float64Array(m * p);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < p; j++) {
      let sr = 0, si = 0;
      for (let k = 0; k < n; k++) {
        const ai_ = aSwap ? (k * m + i) : (i * n + k);
        const bi_ = bSwap ? (j * n + k) : (k * p + j);
        const ar = cA.re[ai_], aim = cA.im[ai_];
        const br = cB.re[bi_], bim = cB.im[bi_];
        sr += ar * br - aim * bim;
        si += ar * bim + aim * br;
      }
      re[i * p + j] = sr; im[i * p + j] = si;
    }
  }
  return _packCx(re, im, [m, p], false);
}

// =====================================================================
// add / sub — shape-dispatched elementwise addition / subtraction
// =====================================================================
//
// Spec §07: `add` and `sub` operate on "scalars or arrays of same
// shape". Both operands must share LOGICAL shape AND the swapped bit
// of their tag (a column vector and a row vector of the same length
// are NOT compatible — they have the same `shape` field but differ in
// orientation, and elementwise data-level addition would be a category
// error). The conjugate bit can differ for real-valued data without
// observational effect; once complex dtypes arrive, conjugation
// differences will need explicit handling at the per-cell level.
//
// Scalar broadcast is allowed in either direction (scalar + array
// scales the scalar over every cell, tag preserved).

// Build the elementwise binary op from a scalar primitive. Used to
// generate `add` and `sub` from `(a,b) => a+b` and `(a,b) => a-b`.
// Complex elementwise add/sub. add/sub are ℂ-linear: the real algebra
// runs independently over the re and im buffers, so the same scalar
// primitive (`(x,y)=>x+y` etc.) applies to both. Shape/orientation
// rules are identical to the real path; broadcast handled by treating
// the missing im of a real operand as an implicit zero buffer (which
// readComplex already supplies).
function _complexLinearBinop(scalarFn, a, b, opName) {
  const sa = a.shape, sb = b.shape;
  const ca = readComplex(a), cb = readComplex(b);
  // scalar ∘ scalar
  if (sa.length === 0 && sb.length === 0) {
    return _packCx(
      [scalarFn(ca.re[0], cb.re[0])],
      [scalarFn(ca.im[0], cb.im[0])], [], false);
  }
  // scalar ∘ array  (broadcast the scalar over every cell; the array
  // operand governs shape AND orientation)
  if (sa.length === 0 || sb.length === 0) {
    const scal = sa.length === 0 ? ca : cb;
    const arr  = sa.length === 0 ? cb : ca;
    const arrV = sa.length === 0 ? b : a;
    const sLeft = sa.length === 0;
    const n = arr.re.length;
    const re = new Float64Array(n), im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = sLeft ? scalarFn(scal.re[0], arr.re[i]) : scalarFn(arr.re[i], scal.re[0]);
      im[i] = sLeft ? scalarFn(scal.im[0], arr.im[i]) : scalarFn(arr.im[i], scal.im[0]);
    }
    return _packCx(re, im, arrV.shape, isTransposeView(arrV));
  }
  // array ∘ array — shapes + orientation must agree (same checks as real)
  if (sa.length !== sb.length) {
    throw new Error(opName + ': rank mismatch (' + JSON.stringify(sa) +
      ' vs ' + JSON.stringify(sb) + ')');
  }
  for (let i = 0; i < sa.length; i++) {
    if (sa[i] !== sb[i]) {
      throw new Error(opName + ': shape mismatch (' + JSON.stringify(sa) +
        ' vs ' + JSON.stringify(sb) + ')');
    }
  }
  if (isTransposeView(a) !== isTransposeView(b)) {
    throw new Error(opName + ': cannot combine values of opposite ' +
      'orientation (one is transposed). Apply transpose to align them first.');
  }
  const n = ca.re.length;
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    re[i] = scalarFn(ca.re[i], cb.re[i]);
    im[i] = scalarFn(ca.im[i], cb.im[i]);
  }
  return _packCx(re, im, sa, isTransposeView(a));
}

function _makeElementwiseBinop(scalarFn, opName) {
  return function elementwiseBinop(a, b) {
    if (!isValue(a) || !isValue(b)) {
      throw new Error('value-ops.' + opName + ': both operands must be Values');
    }
    if (_isCx(a, b)) return _complexLinearBinop(scalarFn, a, b, opName);
    const sa = a.shape, sb = b.shape;
    // scalar × anything → broadcast (preserve tag of the non-scalar)
    if (sa.length === 0 && sb.length === 0) {
      return scalar(scalarFn(a.data[0], b.data[0]));
    }
    if (sa.length === 0) return _scalarBroadcastBinop(scalarFn, a.data[0], b, true);
    if (sb.length === 0) return _scalarBroadcastBinop(scalarFn, b.data[0], a, false);
    // Both have shape. Shapes must match length-by-length.
    if (sa.length !== sb.length) {
      throw new Error(
        opName + ': rank mismatch (' + JSON.stringify(sa) +
        ' vs ' + JSON.stringify(sb) + ')');
    }
    for (let i = 0; i < sa.length; i++) {
      if (sa[i] !== sb[i]) {
        throw new Error(
          opName + ': shape mismatch (' + JSON.stringify(sa) +
          ' vs ' + JSON.stringify(sb) + ')');
      }
    }
    // Orientation (swapped bit) must agree.
    if (isTransposeView(a) !== isTransposeView(b)) {
      throw new Error(
        opName + ': cannot combine values of opposite orientation ' +
        '(one is transposed). Apply transpose to align them first.');
    }
    // Elementwise on the underlying buffers — since shape and swapped
    // bit agree, the data is laid out identically.
    const out = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      out[i] = scalarFn(a.data[i], b.data[i]);
    }
    const r = { shape: a.shape.slice(), data: out };
    // Preserve tag — both operands share the swapped bit; for the
    // conjugate bit (real-valued: no-op) prefer the LHS's tag.
    if (a.t && a.t !== 'N') r.t = a.t;
    if (a.dtype) r.dtype = a.dtype;
    return r;
  };
}

// scalar (JS number) + Value (any shape) → Value, with elementwise
// broadcast. `scalarLeft` is true iff the scalar was the LHS operand
// (important for non-commutative `sub`).
function _scalarBroadcastBinop(scalarFn, s, v, scalarLeft) {
  const out = new Float64Array(v.data.length);
  if (scalarLeft) {
    for (let i = 0; i < v.data.length; i++) out[i] = scalarFn(s, v.data[i]);
  } else {
    for (let i = 0; i < v.data.length; i++) out[i] = scalarFn(v.data[i], s);
  }
  const r = { shape: v.shape.slice(), data: out };
  if (v.t && v.t !== 'N') r.t = v.t;
  if (v.dtype) r.dtype = v.dtype;
  return r;
}

const add = _makeElementwiseBinop((a, b) => a + b, 'add');
const sub = _makeElementwiseBinop((a, b) => a - b, 'sub');

// =====================================================================
// neg — pointwise negation
// =====================================================================
//
// Tag and shape are preserved; data is allocated fresh (caller may
// mutate the input independently after this returns).

function neg(a) {
  if (!isValue(a)) throw new Error('value-ops.neg: argument must be a Value');
  if (isComplexValue(a)) {
    // Negate raw stored buffers and keep the full Klein-4 tag — neg
    // commutes with transpose and conjugation, so no materialization is
    // needed (mirrors the real path: stored layout + tag preserved).
    const reO = new Float64Array(a.data.length);
    const imO = new Float64Array(a.im.length);
    for (let i = 0; i < reO.length; i++) reO[i] = -a.data[i];
    for (let i = 0; i < imO.length; i++) imO[i] = -a.im[i];
    const r = { shape: a.shape.slice(), data: reO, im: imO, dtype: 'complex' };
    if (a.t && a.t !== 'N') r.t = a.t;
    return r;
  }
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = -a.data[i];
  const r = { shape: a.shape.slice(), data: out };
  if (a.t && a.t !== 'N') r.t = a.t;
  if (a.dtype) r.dtype = a.dtype;
  return r;
}

// =====================================================================
// Value ↔ nested-JS-array bridges (Phase 3)
// =====================================================================
//
// The legacy linear-algebra implementations in sampler.js (Cholesky,
// LU, Gauss–Jordan, etc.) operate on nested JS arrays `M[i][j]`. To
// expose those ops on Values without re-implementing each algorithm
// on flat Float64Arrays (per the TODO Phase 3 plan — these are atom-
// indep one-shot ops where reallocation cost is negligible), we
// provide thin bridges: `_valueToNested` materialises a Value into
// nested form, honouring the Klein-4 transpose tag via index
// permutation. `_nestedToValue` packs the result back. The bridges
// support 1-D (vectors → flat JS array) and 2-D (matrices → nested
// JS array) shapes — the ranks the linalg ops accept.

function _valueToNested(v) {
  if (!isValue(v)) throw new Error('_valueToNested: not a Value');
  const r = v.shape.length;
  if (r === 1) {
    // Vector → flat JS array. Tag toggles row/column orientation but
    // doesn't change the data layout (vectors are 1-D).
    return Array.from(v.data);
  }
  if (r === 2) {
    const [m, n] = v.shape;
    const swapped = isTransposeView(v);
    const out = new Array(m);
    if (!swapped) {
      for (let i = 0; i < m; i++) {
        const row = new Array(n);
        const base = i * n;
        for (let j = 0; j < n; j++) row[j] = v.data[base + j];
        out[i] = row;
      }
    } else {
      // data is laid out [n, m] row-major; logical (i, j) at data[j*m + i].
      for (let i = 0; i < m; i++) {
        const row = new Array(n);
        for (let j = 0; j < n; j++) row[j] = v.data[j * m + i];
        out[i] = row;
      }
    }
    return out;
  }
  throw new Error('_valueToNested: only rank-1 and rank-2 supported (got ' +
    JSON.stringify(v.shape) + ')');
}

function _nestedToValue(nested) {
  if (Array.isArray(nested) && nested.length > 0 && Array.isArray(nested[0])) {
    const m = nested.length, n = nested[0].length;
    const data = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      const row = nested[i];
      if (row.length !== n) {
        throw new Error('_nestedToValue: ragged matrix at row ' + i);
      }
      const base = i * n;
      for (let j = 0; j < n; j++) data[base + j] = +row[j];
    }
    return { shape: [m, n], data: data };
  }
  if (Array.isArray(nested)) {
    return { shape: [nested.length], data: Float64Array.from(nested) };
  }
  // Bare scalar — wrap.
  return { shape: [], data: new Float64Array([+nested]) };
}

// =====================================================================
// Atom-batched cross (Phase 2d)
// =====================================================================
//
// When an operand carries a leading axis of size N (the atom count),
// it represents one independent intrinsic value per atom — a per-atom
// scalar, vector, or matrix. The atom-indep `mul` / `add` / `sub` /
// `neg` defined above don't know about N; the `…N(args, N)` variants
// below dispatch the atom-batched cases that MvNormal-style models
// require (e.g. `mu + L * z` where L is atom-indep and z is shape=[N,
// n]).
//
// Today's coverage:
//
//   - matrix(m, n) × shape=[N, n] → shape=[N, m]           (mulN)
//   - shape=[k] + shape=[N, k]    → shape=[N, k]           (addN/subN)
//   - shape=[N, k] + shape=[N, k] → shape=[N, k] (delegate to atom-
//                                                 indep add: same data
//                                                 layout)
//   - scalar + shape=[N, ...]     → broadcast (data-level; works via
//                                              the atom-indep add)
//   - pointwise neg               → works at any rank via atom-indep neg
//
// Deferred (uncommon today; lands as Phase 6 / 7 needs surface):
//   - shape=[N, m, n] × shape=[N, n]    (atom-batched matrix × vector)
//   - shape=[N] (batched scalar) ⊙ shape=[N, k]
//   - per-atom matrix × per-atom matrix

// matrix(m, n) × shape=[N, n] → shape=[N, m]. Atom-major output;
// per-atom matvec with shared matrix. Tag on the matrix is honoured
// (BLAS gemm-flag style).
function _matBatchedVecMul(A, V, N) {
  const [m, n] = A.shape;
  if (V.shape.length !== 2 || V.shape[0] !== N || V.shape[1] !== n) {
    throw new Error(
      'mulN: matrix×batchedVector shape mismatch (' +
      JSON.stringify(A.shape) + ' × ' + JSON.stringify(V.shape) +
      '; expected batched vector shape=[N=' + N + ', n=' + n + '])');
  }
  if (isTransposeView(V)) {
    throw new Error('mulN: batched vector must be column-oriented');
  }
  const aSwap = isTransposeView(A);
  const out = new Float64Array(N * m);
  if (!aSwap) {
    for (let atom = 0; atom < N; atom++) {
      const vBase = atom * n;
      const oBase = atom * m;
      for (let i = 0; i < m; i++) {
        let s = 0;
        const row = i * n;
        for (let k = 0; k < n; k++) s += A.data[row + k] * V.data[vBase + k];
        out[oBase + i] = s;
      }
    }
  } else {
    for (let atom = 0; atom < N; atom++) {
      const vBase = atom * n;
      const oBase = atom * m;
      for (let i = 0; i < m; i++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += A.data[k * m + i] * V.data[vBase + k];
        out[oBase + i] = s;
      }
    }
  }
  return { shape: [N, m], data: out };
}

// Complex atom-batched matrix × shape=[N, n] → complex shape=[N, m].
// Per-atom matvec with a shared (possibly transposed/adjoint) matrix;
// readComplex folds the matrix's conj bit (Hermitian for adjoint(A)).
function _cxMatBatchedVecMul(A, V, N) {
  const [m, n] = A.shape;
  if (V.shape.length !== 2 || V.shape[0] !== N || V.shape[1] !== n) {
    throw new Error(
      'mulN: matrix×batchedVector shape mismatch (' +
      JSON.stringify(A.shape) + ' × ' + JSON.stringify(V.shape) +
      '; expected batched vector shape=[N=' + N + ', n=' + n + '])');
  }
  if (isTransposeView(V)) {
    throw new Error('mulN: batched vector must be column-oriented');
  }
  const cA = readComplex(A), cV = readComplex(V);
  const aSwap = isTransposeView(A);
  const re = new Float64Array(N * m), im = new Float64Array(N * m);
  for (let atom = 0; atom < N; atom++) {
    const vBase = atom * n, oBase = atom * m;
    for (let i = 0; i < m; i++) {
      let sr = 0, si = 0;
      for (let k = 0; k < n; k++) {
        const ai_ = aSwap ? (k * m + i) : (i * n + k);
        const ar = cA.re[ai_], aim = cA.im[ai_];
        const br = cV.re[vBase + k], bim = cV.im[vBase + k];
        sr += ar * br - aim * bim;
        si += ar * bim + aim * br;
      }
      re[oBase + i] = sr; im[oBase + i] = si;
    }
  }
  return _packCx(re, im, [N, m], false);
}

// Atom-indep value broadcast over the leading N axis of an atom-batched
// value, applied via a binary scalar fn. `batched` has shape=[N, ...rest];
// `indep` must have shape=rest (same orientation). Result has the
// batched shape.
function _atomBroadcastBinop(scalarFn, batched, indep, N, swapArgs, opName) {
  if (batched.shape[0] !== N) {
    throw new Error(opName + 'N: leading axis (' + batched.shape[0] +
      ') is not the atom count N=' + N);
  }
  const restLen = batched.shape.length - 1;
  if (indep.shape.length !== restLen) {
    throw new Error(opName + 'N: atom-indep rank ' + indep.shape.length +
      ' doesn\'t match atom-batched per-atom rank ' + restLen);
  }
  for (let i = 0; i < restLen; i++) {
    if (batched.shape[i + 1] !== indep.shape[i]) {
      throw new Error(opName + 'N: per-atom shape mismatch ' +
        JSON.stringify(batched.shape.slice(1)) + ' vs ' +
        JSON.stringify(indep.shape));
    }
  }
  if (isTransposeView(batched) !== isTransposeView(indep)) {
    throw new Error(opName + 'N: opposite orientation between atom-batched ' +
      'and atom-indep operands');
  }
  if (isComplexValue(batched) || isComplexValue(indep)) {
    // Complex add/sub is ℂ-linear: run the same scalar primitive over
    // the re and im buffers independently. readComplex resolves any
    // conjugation; the swapped bit (transpose) rides along on the
    // batched operand exactly as in the real path.
    const cb = readComplex(batched), ci = readComplex(indep);
    const stride = ci.re.length;
    const re = new Float64Array(cb.re.length);
    const im = new Float64Array(cb.im.length);
    for (let atom = 0; atom < N; atom++) {
      const base = atom * stride;
      for (let i = 0; i < stride; i++) {
        const bi = base + i;
        if (swapArgs) {
          re[bi] = scalarFn(ci.re[i], cb.re[bi]);
          im[bi] = scalarFn(ci.im[i], cb.im[bi]);
        } else {
          re[bi] = scalarFn(cb.re[bi], ci.re[i]);
          im[bi] = scalarFn(cb.im[bi], ci.im[i]);
        }
      }
    }
    return _packCx(re, im, batched.shape, isTransposeView(batched));
  }
  const stride = indep.data.length;
  const out = new Float64Array(batched.data.length);
  if (swapArgs) {
    for (let atom = 0; atom < N; atom++) {
      const base = atom * stride;
      for (let i = 0; i < stride; i++) {
        out[base + i] = scalarFn(indep.data[i], batched.data[base + i]);
      }
    }
  } else {
    for (let atom = 0; atom < N; atom++) {
      const base = atom * stride;
      for (let i = 0; i < stride; i++) {
        out[base + i] = scalarFn(batched.data[base + i], indep.data[i]);
      }
    }
  }
  const r = { shape: batched.shape.slice(), data: out };
  if (batched.t && batched.t !== 'N') r.t = batched.t;
  if (batched.dtype) r.dtype = batched.dtype;
  return r;
}

// Atom-batched marker: leading axis is the atom count AND there is a
// non-trivial per-atom shape (rank ≥ 2).
function _hasAtomAxis(v, N) {
  return v.shape.length >= 2 && v.shape[0] === N;
}

// mulN: atom-aware multiplication. Routes the MvNormal-style
// matrix × shape=[N, n] case to _matBatchedVecMul; otherwise delegates
// to the atom-indep `mul` (which already handles scalar broadcast,
// matmul, matvec etc. correctly when neither operand has an atom axis).
function mulN(a, b, N) {
  if (!isValue(a) || !isValue(b)) {
    throw new Error('value-ops.mulN: both operands must be Values');
  }
  const aBatched = _hasAtomAxis(a, N);
  const bBatched = _hasAtomAxis(b, N);
  // matrix × shape=[N, n]: a is shape [m, n], b is shape [N, n].
  if (!aBatched && bBatched
      && a.shape.length === 2 && b.shape.length === 2) {
    if (_isCx(a, b)) return _cxMatBatchedVecMul(a, b, N);
    return _matBatchedVecMul(a, b, N);
  }
  // Atom-indep case.
  if (!aBatched && !bBatched) return mul(a, b);
  // Other atom-batched cases land in Phase 6 / 7 as needed.
  throw new Error(
    'mulN: unsupported atom-batched shape combination ' +
    JSON.stringify(a.shape) + ' × ' + JSON.stringify(b.shape) +
    ' with N=' + N);
}

// addN / subN: atom-aware. Handles the atom-indep + atom-batched
// broadcast that MvNormal's `mu + L*z` needs (mu is atom-indep,
// L*z is atom-batched).
function _makeAtomAwareBinop(scalarFn, atomIndepImpl, opName) {
  return function atomAwareBinop(a, b, N) {
    if (!isValue(a) || !isValue(b)) {
      throw new Error('value-ops.' + opName + 'N: both operands must be Values');
    }
    const aBatched = _hasAtomAxis(a, N);
    const bBatched = _hasAtomAxis(b, N);
    if (aBatched && bBatched) {
      // Both atom-batched: same shape required; delegate to atom-indep
      // elementwise add (the data layouts agree and rank includes N).
      return atomIndepImpl(a, b);
    }
    if (aBatched && !bBatched) {
      return _atomBroadcastBinop(scalarFn, a, b, N, false, opName);
    }
    if (!aBatched && bBatched) {
      return _atomBroadcastBinop(scalarFn, b, a, N, true, opName);
    }
    return atomIndepImpl(a, b);
  };
}

const addN = _makeAtomAwareBinop((x, y) => x + y, add, 'add');
const subN = _makeAtomAwareBinop((x, y) => x - y, sub, 'sub');

// negN: pointwise — the atom-indep neg already iterates over the
// whole data buffer regardless of rank, so atom-batched values are
// handled correctly without extra plumbing.
function negN(a, _N) {
  return neg(a);
}

module.exports = {
  mul,
  add,
  sub,
  neg,
  mulN,
  addN,
  subN,
  negN,
  // Exposed for direct use / test access; the public functions cover
  // every dispatch path.
  _valueToNested,
  _nestedToValue,
  _innerProduct,
  _outerProduct,
  _matVecMul,
  _vecMatMul,
  _matMatMul,
  _matBatchedVecMul,
  _atomBroadcastBinop,
  _scalarBroadcastMul,
  _scalarBroadcastBinop,
  _matIdxN,
  _matIdxT,
};
