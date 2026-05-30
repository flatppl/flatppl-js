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

import type { Value } from './engine-types';
const valueLib = require('./value.ts');
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
function _isCx(a: any, b: any) {
  return isComplexValue(a) || (b !== undefined && isComplexValue(b));
}

// Pack two equal-length re/im buffers back into a complex Value with the
// given logical shape. readComplex already resolved any input
// conjugation into the buffers, so the result carries no conj bit; but
// it leaves the data in STORED (pre-transpose) layout, so the swapped
// bit must be carried forward when the governing operand was a
// transpose view. `swapped` true ⇒ tag 'T' (pure transpose, conj
// already folded), else canonical 'N'.
function _packCx(re: any, im: any, shape: any, swapped: any) {
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

function _matIdxN(i: any, j: any, n: any) { return i * n + j; }
function _matIdxT(i: any, j: any, m: any) { return j * m + i; }

// ---------------------------------------------------------------------
// mul — shape-dispatched multiplication
// ---------------------------------------------------------------------

// Diagonal fast-paths for `mul`. A diag Value is vector-backed
// (data = the m-vector, logical shape [m,m]); these avoid both the
// O(m²) densification and the O(m³) gemm. Returns a Value, or null
// when the combination isn't fast-pathed (caller densifies the diag
// operand(s) and falls through to the generic dense path — correctness
// never depends on a fast-path existing). Real diagonals only; a
// complex diagonal returns null → densify (rare; cov is real).
function _diagMul(a: any, b: any) {
  const aD = valueLib.isDiagStored(a) && !a.im;
  const bD = valueLib.isDiagStored(b) && !b.im;
  if (!aD && !bD) return null;
  const sa = a.shape, sb = b.shape;
  // scalar × diag  /  diag × scalar  → diag (structure preserved)
  if (aD && sb.length === 0) {
    const s = b.data[0], d = new Float64Array(a.data.length);
    for (let i = 0; i < d.length; i++) d[i] = a.data[i] * s;
    return valueLib.diagMatrix(d);
  }
  if (bD && sa.length === 0) {
    const s = a.data[0], d = new Float64Array(b.data.length);
    for (let i = 0; i < d.length; i++) d[i] = s * b.data[i];
    return valueLib.diagMatrix(d);
  }
  // diag × diag → diag (Hadamard of the diagonals)
  if (aD && bD) {
    if (a.data.length !== b.data.length) {
      throw new Error('mul: diagonal dimension mismatch (' +
        a.data.length + ' vs ' + b.data.length + ')');
    }
    const d = new Float64Array(a.data.length);
    for (let i = 0; i < d.length; i++) d[i] = a.data[i] * b.data[i];
    return valueLib.diagMatrix(d);
  }
  // diag(m×m) × column-vector(m) → vector(m): elementwise scale
  if (aD && sb.length === 1 && !isTransposeView(b)) {
    const d = a.data;
    if (b.shape[0] !== d.length) {
      throw new Error('mul: matrix×vector dimension mismatch ([' +
        d.length + ',' + d.length + '] × [' + b.shape[0] + '])');
    }
    const out = new Float64Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = d[i] * b.data[i];
    return { shape: [d.length], data: out };
  }
  // row-vector(m) × diag(m×m) → row vector(m): elementwise scale
  if (bD && sa.length === 1 && isTransposeView(a)) {
    const d = b.data;
    if (a.shape[0] !== d.length) {
      throw new Error('mul: vector×matrix dimension mismatch');
    }
    const out = new Float64Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = a.data[i] * d[i];
    return { shape: [d.length], data: out, t: 'T' };
  }
  // diag(m×m) × dense matrix(m×p) → scale rows  (canonical, real)
  if (aD && sb.length === 2 && !b.im && !isTransposeView(b)
      && b.struct === undefined) {
    const d = a.data, m = d.length, p = b.shape[1];
    if (b.shape[0] !== m) {
      throw new Error('mul: matrix×matrix dimension mismatch');
    }
    const out = new Float64Array(m * p);
    for (let i = 0; i < m; i++) {
      const di = d[i], base = i * p;
      for (let j = 0; j < p; j++) out[base + j] = di * b.data[base + j];
    }
    return { shape: [m, p], data: out };
  }
  // dense matrix(p×m) × diag(m×m) → scale columns  (canonical, real)
  if (bD && sa.length === 2 && !a.im && !isTransposeView(a)
      && a.struct === undefined) {
    const d = b.data, m = d.length, p = a.shape[0];
    if (a.shape[1] !== m) {
      throw new Error('mul: matrix×matrix dimension mismatch');
    }
    const out = new Float64Array(p * m);
    for (let i = 0; i < p; i++) {
      const base = i * m;
      for (let j = 0; j < m; j++) out[base + j] = a.data[base + j] * d[j];
    }
    return { shape: [p, m], data: out };
  }
  return null;   // not fast-pathed → caller densifies
}

// Lazy reference to ops.dispatch — assigned on first call to avoid a
// hard module-load dep on ops.ts (which would require ops-declarations.ts
// to have already registered the mul variants). Callers of this file
// during engine bootstrap (e.g. row_gram registration in
// ops-declarations.ts) invoke `mul` AFTER variants are registered;
// the lazy lookup guarantees correct sequencing in either order.
let _opsModule: any = null;
function _ops(): any {
  if (!_opsModule) _opsModule = require('./ops.ts');
  return _opsModule;
}

function mul(a: Value, b: Value): Value {
  // Runtime check stays — also narrows the type predicate's promise
  // when called from `any`-typed sites that haven't migrated yet.
  if (!isValue(a) || !isValue(b)) {
    throw new Error('value-ops.mul: both operands must be Values');
  }
  // Diag fast-path (pre-dispatch). `_diagMul` covers diag×scalar,
  // diag×vec, diag×mat, mat×diag, and the diag×diag Hadamard. On
  // any combination it doesn't fast-path (e.g. complex diag) it
  // returns null, the diag operand(s) are densified, and we fall
  // through to the generic dense path.
  if (valueLib.isDiagStored(a) || valueLib.isDiagStored(b)) {
    const r = _diagMul(a, b);
    if (r !== null) return r;
    if (valueLib.isDiagStored(a)) a = valueLib.densify(a);
    if (valueLib.isDiagStored(b)) b = valueLib.densify(b);
  }
  // Complex branch (pre-dispatch). Conjugation-aware complex gemm via
  // dedicated `_cx*` helpers. Migrating complex into the variant
  // registry is a future P1 follow-up; today the per-arg ArgPattern
  // doesn't naturally express "any operand is complex" without
  // duplicating every shape variant per-arg-complex case.
  const sa = a.shape, sb = b.shape;
  if (_isCx(a, b)) {
    if (sa.length === 0 || sb.length === 0) return _complexScalarBroadcastMul(a, b);
    if (sa.length === 1 && sb.length === 1) return _cxVecVecMul(a, b);
    if (sa.length === 2 && sb.length === 1) return _cxMatVecMul(a, b);
    if (sa.length === 1 && sb.length === 2) return _cxVecMatMul(a, b);
    if (sa.length === 2 && sb.length === 2) return _cxMatMatMul(a, b);
    throw new Error(
      'value-ops.mul: unsupported complex shape combination ' +
      JSON.stringify(sa) + ' × ' + JSON.stringify(sb));
  }
  // Real path: dispatched via the variant registry (engine-concepts
  // §18.11). Each (rank-A tag-X) × (rank-B tag-Y) combination maps
  // to one registered variant entry pointing at the corresponding
  // helper (_scalarBroadcastMul / _vecVecMul split into inner /
  // outer / error / _matVecMul / _vecMatMul / _matMatMul).
  return _ops().dispatch('mul', [a, b], { wrappingOp: 'direct' });
}

// scalar (JS number) × Value (any shape) → Value with same shape and tag.
function _scalarBroadcastMul(s: any, v: any) {
  const out = new Float64Array(v.data.length);
  for (let i = 0; i < v.data.length; i++) out[i] = s * v.data[i];
  const r: any = { shape: v.shape.slice(), data: out };
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
function _complexScalarBroadcastMul(a: any, b: any) {
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
function _vecVecMul(u: any, v: any) {
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
function _innerProduct(u: any, v: any) {
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
function _outerProduct(u: any, v: any) {
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
function _matVecMul(A: any, v: any) {
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
function _vecMatMul(u: any, B: any) {
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
  const tagOut: 'T' | 'A' = (getTag(u) === 'A') ? 'A' : 'T';
  return { shape: [p], data: out, t: tagOut };
}

// matrix(m, n) × matrix(n, p) → matrix(m, p). Tags read via index
// permutation (BLAS gemm-flag style); output is canonical (tag N).
function _matMatMul(A: any, B: any) {
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

function _cxVecVecMul(u: any, v: any) {
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

function _cxInnerProduct(u: any, v: any) {
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

function _cxOuterProduct(u: any, v: any) {
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

function _cxMatVecMul(A: any, v: any) {
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

function _cxVecMatMul(u: any, B: any) {
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

function _cxMatMatMul(A: any, B: any) {
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
function _complexLinearBinop(scalarFn: any, a: any, b: any, opName: any) {
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
  // array ∘ array — same rank required; per-axis sizes match OR one
  // is size 1 (NumPy / spec §04 singleton-axis broadcast).
  if (sa.length !== sb.length) {
    throw new Error(opName + ': rank mismatch (' + JSON.stringify(sa) +
      ' vs ' + JSON.stringify(sb) + ')');
  }
  let needsBroadcast = false;
  for (let i = 0; i < sa.length; i++) {
    if (sa[i] === sb[i]) continue;
    if (sa[i] === 1 || sb[i] === 1) { needsBroadcast = true; continue; }
    throw new Error(opName + ': shape mismatch (' + JSON.stringify(sa) +
      ' vs ' + JSON.stringify(sb) + ')');
  }
  if (isTransposeView(a) !== isTransposeView(b)) {
    throw new Error(opName + ': cannot combine values of opposite ' +
      'orientation (one is transposed). Apply transpose to align them first.');
  }
  if (!needsBroadcast) {
    const n = ca.re.length;
    const re = new Float64Array(n), im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = scalarFn(ca.re[i], cb.re[i]);
      im[i] = scalarFn(ca.im[i], cb.im[i]);
    }
    return _packCx(re, im, sa, isTransposeView(a));
  }
  // Singleton-axis broadcast — same coord-walker as the real path,
  // applied separately to re and im buffers.
  const out_shape = new Array(sa.length);
  for (let i = 0; i < sa.length; i++) out_shape[i] = Math.max(sa[i], sb[i]);
  const aStrides = _broadcastStridesForShape(sa, out_shape);
  const bStrides = _broadcastStridesForShape(sb, out_shape);
  const outSize = out_shape.reduce((p, n) => p * n, 1);
  const re = new Float64Array(outSize), im = new Float64Array(outSize);
  const coord = new Array(sa.length).fill(0);
  for (let linear = 0; linear < outSize; linear++) {
    let aOff = 0, bOff = 0;
    for (let i = 0; i < sa.length; i++) {
      aOff += coord[i] * aStrides[i];
      bOff += coord[i] * bStrides[i];
    }
    re[linear] = scalarFn(ca.re[aOff], cb.re[bOff]);
    im[linear] = scalarFn(ca.im[aOff], cb.im[bOff]);
    for (let i = sa.length - 1; i >= 0; i--) {
      coord[i]++;
      if (coord[i] < out_shape[i]) break;
      coord[i] = 0;
    }
  }
  return _packCx(re, im, out_shape, isTransposeView(a));
}

function _makeElementwiseBinop(scalarFn: any, opName: any) {
  return function elementwiseBinop(a: any, b: any) {
    if (!isValue(a) || !isValue(b)) {
      throw new Error('value-ops.' + opName + ': both operands must be Values');
    }
    if (valueLib.isDiagStored(a) || valueLib.isDiagStored(b)) {
      // diag ∘ diag (real) stays diag — operate on the m-vectors.
      // Any other mix densifies the diag operand(s) (occupancy of
      // diag+dense is dense anyway) so the generic path is correct.
      if (valueLib.isDiagStored(a) && valueLib.isDiagStored(b)
          && !a.im && !b.im && a.data.length === b.data.length) {
        const d = new Float64Array(a.data.length);
        for (let i = 0; i < d.length; i++) d[i] = scalarFn(a.data[i], b.data[i]);
        return valueLib.diagMatrix(d);
      }
      if (valueLib.isDiagStored(a)) a = valueLib.densify(a);
      if (valueLib.isDiagStored(b)) b = valueLib.densify(b);
    }
    if (_isCx(a, b)) return _complexLinearBinop(scalarFn, a, b, opName);
    const sa = a.shape, sb = b.shape;
    // scalar × anything → broadcast (preserve tag of the non-scalar)
    if (sa.length === 0 && sb.length === 0) {
      return scalar(scalarFn(a.data[0], b.data[0]));
    }
    if (sa.length === 0) return _scalarBroadcastBinop(scalarFn, a.data[0], b, true);
    if (sb.length === 0) return _scalarBroadcastBinop(scalarFn, b.data[0], a, false);
    // Both have shape. Shapes must match length-by-length OR allow
    // singleton-axis broadcast (NumPy convention; spec §04: "size-
    // one array axes are implicitly expanded by repetition to match
    // the size of the other collection arguments along these axes").
    if (sa.length !== sb.length) {
      throw new Error(
        opName + ': rank mismatch (' + JSON.stringify(sa) +
        ' vs ' + JSON.stringify(sb) + ')');
    }
    let needsBroadcast = false;
    for (let i = 0; i < sa.length; i++) {
      if (sa[i] === sb[i]) continue;
      if (sa[i] === 1 || sb[i] === 1) { needsBroadcast = true; continue; }
      throw new Error(
        opName + ': shape mismatch (' + JSON.stringify(sa) +
        ' vs ' + JSON.stringify(sb) + '; singleton-axis '
        + 'broadcast requires the differing dim to be 1 on at '
        + 'least one operand)');
    }
    // Orientation (swapped bit) must agree.
    if (isTransposeView(a) !== isTransposeView(b)) {
      throw new Error(
        opName + ': cannot combine values of opposite orientation ' +
        '(one is transposed). Apply transpose to align them first.');
    }
    if (!needsBroadcast) {
      // Equal shapes — flat elementwise on the underlying buffers.
      const out = new Float64Array(a.data.length);
      for (let i = 0; i < a.data.length; i++) {
        out[i] = scalarFn(a.data[i], b.data[i]);
      }
      const r: any = { shape: a.shape.slice(), data: out };
      if (a.t && a.t !== 'N') r.t = a.t;
      if (a.dtype) r.dtype = a.dtype;
      return r;
    }
    // Singleton-axis broadcast: compute the per-operand row-major
    // strides where any size-1 dim contributes stride 0 (the
    // singleton broadcasts by repetition). Output shape: max of
    // the two per-axis. NumPy convention.
    const out_shape = new Array(sa.length);
    for (let i = 0; i < sa.length; i++) {
      out_shape[i] = Math.max(sa[i], sb[i]);
    }
    const aStrides = _broadcastStridesForShape(sa, out_shape);
    const bStrides = _broadcastStridesForShape(sb, out_shape);
    const N = sa.length;
    const outSize = out_shape.reduce((p: number, n: number) => p * n, 1);
    const out = new Float64Array(outSize);
    // Rank-specific fast paths for the common cases (rank 1/2/3);
    // generic coord-walker otherwise.
    if (N === 1) {
      const D = out_shape[0], aS = aStrides[0], bS = bStrides[0];
      for (let i = 0; i < D; i++) out[i] = scalarFn(a.data[i * aS], b.data[i * bS]);
    } else if (N === 2) {
      const D0 = out_shape[0], D1 = out_shape[1];
      const aS0 = aStrides[0], aS1 = aStrides[1];
      const bS0 = bStrides[0], bS1 = bStrides[1];
      let o = 0;
      for (let i = 0; i < D0; i++) {
        const aB = i * aS0, bB = i * bS0;
        for (let j = 0; j < D1; j++) out[o++] = scalarFn(a.data[aB + j * aS1], b.data[bB + j * bS1]);
      }
    } else if (N === 3) {
      const D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
      const aS0 = aStrides[0], aS1 = aStrides[1], aS2 = aStrides[2];
      const bS0 = bStrides[0], bS1 = bStrides[1], bS2 = bStrides[2];
      let o = 0;
      for (let i = 0; i < D0; i++) {
        const aB0 = i * aS0, bB0 = i * bS0;
        for (let j = 0; j < D1; j++) {
          const aB1 = aB0 + j * aS1, bB1 = bB0 + j * bS1;
          for (let k = 0; k < D2; k++) out[o++] = scalarFn(a.data[aB1 + k * aS2], b.data[bB1 + k * bS2]);
        }
      }
    } else {
      const coord = new Array(N).fill(0);
      for (let linear = 0; linear < outSize; linear++) {
        let aOff = 0, bOff = 0;
        for (let i = 0; i < N; i++) {
          aOff += coord[i] * aStrides[i];
          bOff += coord[i] * bStrides[i];
        }
        out[linear] = scalarFn(a.data[aOff], b.data[bOff]);
        for (let i = N - 1; i >= 0; i--) {
          coord[i]++;
          if (coord[i] < out_shape[i]) break;
          coord[i] = 0;
        }
      }
    }
    const r: any = { shape: out_shape, data: out };
    if (a.t && a.t !== 'N') r.t = a.t;
    if (a.dtype) r.dtype = a.dtype;
    return r;
  };
}

// Per-axis broadcasting stride for source shape `src` against target
// `out_shape`. Stride is 0 wherever src[i] === 1 (singleton dim →
// repeat-by-broadcast); else the row-major stride of src.
function _broadcastStridesForShape(src: number[], out_shape: number[]): number[] {
  const N = src.length;
  const out = new Array(N);
  // Row-major strides for src: stride[i] = product of src[i+1..].
  let s = 1;
  for (let i = N - 1; i >= 0; i--) {
    out[i] = (src[i] === 1) ? 0 : s;
    s *= src[i];
  }
  return out;
}

// scalar (JS number) + Value (any shape) → Value, with elementwise
// broadcast. `scalarLeft` is true iff the scalar was the LHS operand
// (important for non-commutative `sub`).
function _scalarBroadcastBinop(scalarFn: any, s: any, v: any, scalarLeft: any) {
  const out = new Float64Array(v.data.length);
  if (scalarLeft) {
    for (let i = 0; i < v.data.length; i++) out[i] = scalarFn(s, v.data[i]);
  } else {
    for (let i = 0; i < v.data.length; i++) out[i] = scalarFn(v.data[i], s);
  }
  const r: any = { shape: v.shape.slice(), data: out };
  if (v.t && v.t !== 'N') r.t = v.t;
  if (v.dtype) r.dtype = v.dtype;
  return r;
}

const add = _makeElementwiseBinop((a: any, b: any) => a + b, 'add');
const sub = _makeElementwiseBinop((a: any, b: any) => a - b, 'sub');

// =====================================================================
// Batched-elementwise primitives for ops whose spec form has different
// semantics (multiplicative arith) or is scalar-only (unary maths).
// =====================================================================
//
// These are the engine primitives for `broadcasted(<op>)(args)` — the
// canonical batched form of each scalar op (engine-concepts §20.1; spec
// §04 `broadcasted(f)`). They differ from spec-`<op>` for ops whose
// spec semantics overload on rank (matrix product for `mul`, etc.) —
// `broadcast(mul, A, B)` is elementwise per cell, distinct from
// `mul(A, B)` which is matrix product on rank-2.
//
// Real-only for the binary multiplicative family; complex dispatch
// falls back to per-cell broadcast (rare in practice; covered by
// existing `_cxBroadcast`). The factory pattern matches `add`/`sub` —
// elementwise loop over flat row-major data; rank-0 × rank-N
// broadcasts via `_scalarBroadcastBinop`; shapes must otherwise
// match.

const mulElem = _makeElementwiseBinop((a: any, b: any) => a * b, 'mulElem');
const divElem = _makeElementwiseBinop((a: any, b: any) => a / b, 'divElem');
const powElem = _makeElementwiseBinop((a: any, b: any) => Math.pow(a, b), 'powElem');
const modElem = _makeElementwiseBinop((a: any, b: any) => a % b, 'modElem');

// Unary elementwise factory: same shape as `_makeElementwiseBinop` but
// for one-argument scalar functions. Applies `scalarFn` pointwise over
// the flat data buffer regardless of rank. Diag-stored falls back to
// densify (the scalar fn need not preserve zero, so we can't keep diag
// storage in general).
// `_makeElementwiseUnop(scalarFn, opName, complexFn?)`.
//
// `complexFn` (optional) is a complex unary scalar function from
// sampler-complex.ts (`_cExp` / `_cLog` / `_cSqrt` / `_cAbs` /
// `_cAbs2` / etc.). When the input is a complex Value AND
// `complexFn` is supplied, the elementwise loop iterates over the
// flat re/im buffers in parallel and writes back to fresh re/im
// outputs — exactly the planar/TF.js convention.
//
// Special case: `_cAbs` and `_cAbs2` return a REAL number from a
// complex input. The factory detects this by sniffing the first
// per-cell result; if it's a number, the output is a real Value
// (no `im` buffer). Otherwise the output is complex.
//
// When `complexFn` is NOT supplied AND the input is complex, the
// factory throws — preserves the legacy behaviour for ops that
// haven't been complex-extended (most won't need it; e.g. floor /
// ceil / round are integer-domain ops).
function _makeElementwiseUnop(scalarFn: any, opName: any, complexFn?: any) {
  return function elementwiseUnop(a: any): any {
    if (!isValue(a)) {
      throw new Error('value-ops.' + opName + ': operand must be a Value');
    }
    if (valueLib.isDiagStored(a)) a = valueLib.densify(a);
    if (isComplexValue(a)) {
      if (!complexFn) {
        throw new Error('value-ops.' + opName + ': complex input not supported '
          + 'on this fast path; expected real-only Value');
      }
      // Planar complex unary loop: read each (re[i], im[i]) cell as
      // a scalar complex, apply complexFn, write back. The §2.1
      // shape contract is preserved (same shape; same atom-axis
      // convention; same Klein-4 tag).
      const n = a.data.length;
      const cRe = a.data, cIm = a.im;
      // Detect the result kind by peeking the first cell. If
      // complexFn returns a number, the output is real; else
      // complex. ({re, im} object).
      const probe = complexFn({ re: cRe[0], im: cIm[0] });
      const wantReal = typeof probe === 'number';
      const outRe = new Float64Array(n);
      const outIm = wantReal ? null : new Float64Array(n);
      outRe[0] = wantReal ? probe : probe.re;
      if (!wantReal) (outIm as Float64Array)[0] = probe.im;
      for (let i = 1; i < n; i++) {
        const r = complexFn({ re: cRe[i], im: cIm[i] });
        if (wantReal) outRe[i] = r as number;
        else { outRe[i] = (r as any).re; (outIm as Float64Array)[i] = (r as any).im; }
      }
      if (wantReal) {
        const r: any = { shape: a.shape.slice(), data: outRe };
        if (a.t && a.t !== 'N') r.t = a.t;
        return r;
      }
      const r: any = {
        shape: a.shape.slice(), data: outRe, im: outIm, dtype: 'complex',
      };
      if (a.t && a.t !== 'N') r.t = a.t;
      return r;
    }
    const out = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) out[i] = scalarFn(a.data[i]);
    const r: any = { shape: a.shape.slice(), data: out };
    if (a.t && a.t !== 'N') r.t = a.t;
    if (a.dtype) r.dtype = a.dtype;
    return r;
  };
}

// Engine primitives for `broadcasted(<unary-scalar-op>)(arg)`. Each
// applies the underlying JS scalar fn pointwise over flat data at any
// rank. Names use the `<op>Elem` convention to keep them distinct
// from the existing scalar-only `ARITH_OPS.<op>` entries.
// Complex-aware unary scalar fns from sampler-complex. Lazy require:
// sampler-complex is a pure ESM leaf with no engine-internal deps,
// so the load order is safe. Lazy access via getter to defer
// resolution until first call (sampler-complex is a sibling module
// in the same package).
let _cx: any = null;
function _cxImpl() {
  if (_cx === null) {
    _cx = require('./sampler-complex.ts');
  }
  return _cx;
}
// Complex unary functions paired with their real counterparts.
// _cAbs / _cAbs2 return a REAL number (complex → real); the
// factory detects this and outputs a real Value. The others
// (exp/log/sqrt) return complex.
const expElem    = _makeElementwiseUnop(Math.exp,    'expElem',    (z: any) => _cxImpl()._cExp(z));
const logElem    = _makeElementwiseUnop(Math.log,    'logElem',    (z: any) => _cxImpl()._cLog(z));
const sqrtElem   = _makeElementwiseUnop(Math.sqrt,   'sqrtElem',   (z: any) => _cxImpl()._cSqrt(z));
const sinElem    = _makeElementwiseUnop(Math.sin,    'sinElem',    (z: any) => _cxImpl()._cSin(z));
const cosElem    = _makeElementwiseUnop(Math.cos,    'cosElem',    (z: any) => _cxImpl()._cCos(z));
const tanElem    = _makeElementwiseUnop(Math.tan,    'tanElem',    (z: any) => _cxImpl()._cTan(z));
// abs / abs2 of complex returns a real value.
const absElem    = _makeElementwiseUnop(Math.abs,    'absElem',    (z: any) => _cxImpl()._cAbs(z));
const abs2Elem   = _makeElementwiseUnop((x: any) => x * x, 'abs2Elem', (z: any) => _cxImpl()._cAbs2(z));
const log10Elem  = _makeElementwiseUnop(Math.log10,  'log10Elem');
const log1pElem  = _makeElementwiseUnop(Math.log1p,  'log1pElem');
const expm1Elem  = _makeElementwiseUnop(Math.expm1,  'expm1Elem');
const floorElem  = _makeElementwiseUnop(Math.floor,  'floorElem');
const ceilElem   = _makeElementwiseUnop(Math.ceil,   'ceilElem');
const roundElem  = _makeElementwiseUnop(Math.round,  'roundElem');

// =====================================================================
// P9 additions — comparisons / predicates / logic / extra trig + math
// =====================================================================
//
// Element-wise impls for the remaining ARITH_OPS_N entries. Each is
// a one-liner over the shared `_makeElementwise*` factories. Once
// registered as broadcast variants (ops-declarations.ts BCAST_TABLE),
// these flow through the unified variant dispatcher instead of
// falling through to ARITH_OPS_N's broadcast1/2/3.
//
// Bool / predicate ops return 0/1 (the JS convention for boolean-
// in-Float64Array); the type system already tags these as boolean
// via SIGNATURE_FACTORIES.

// Pairwise reductions
const minElem    = _makeElementwiseBinop((a: any, b: any) => Math.min(a, b), 'minElem');
const maxElem    = _makeElementwiseBinop((a: any, b: any) => Math.max(a, b), 'maxElem');
// Comparisons (return 0/1)
const ltElem     = _makeElementwiseBinop((a: any, b: any) => a <  b ? 1 : 0, 'ltElem');
const leElem     = _makeElementwiseBinop((a: any, b: any) => a <= b ? 1 : 0, 'leElem');
const gtElem     = _makeElementwiseBinop((a: any, b: any) => a >  b ? 1 : 0, 'gtElem');
const geElem     = _makeElementwiseBinop((a: any, b: any) => a >= b ? 1 : 0, 'geElem');
const equalElem  = _makeElementwiseBinop((a: any, b: any) => a === b ? 1 : 0, 'equalElem');
const unequalElem = _makeElementwiseBinop((a: any, b: any) => a !== b ? 1 : 0, 'unequalElem');
// Predicates
const isfiniteElem = _makeElementwiseUnop((x: any) => Number.isFinite(x) ? 1 : 0, 'isfiniteElem');
const isinfElem    = _makeElementwiseUnop((x: any) => (!Number.isFinite(x) && !Number.isNaN(x)) ? 1 : 0, 'isinfElem');
const isnanElem    = _makeElementwiseUnop((x: any) => Number.isNaN(x) ? 1 : 0, 'isnanElem');
const iszeroElem   = _makeElementwiseUnop((x: any) => x === 0 ? 1 : 0, 'iszeroElem');
// Logic — assumes inputs are 0/1 booleans-as-numbers
const landElem  = _makeElementwiseBinop((a: any, b: any) => (a && b) ? 1 : 0, 'landElem');
const lorElem   = _makeElementwiseBinop((a: any, b: any) => (a || b) ? 1 : 0, 'lorElem');
const lxorElem  = _makeElementwiseBinop((a: any, b: any) => ((!a) !== (!b)) ? 1 : 0, 'lxorElem');
const lnotElem  = _makeElementwiseUnop((x: any) => x ? 0 : 1, 'lnotElem');
// atan2 (two-arg trig)
const atan2Elem = _makeElementwiseBinop(Math.atan2, 'atan2Elem');
// Remaining trig + hyperbolic
const asinElem  = _makeElementwiseUnop(Math.asin,  'asinElem',  (z: any) => _cxImpl()._cAsin(z));
const acosElem  = _makeElementwiseUnop(Math.acos,  'acosElem',  (z: any) => _cxImpl()._cAcos(z));
const atanElem  = _makeElementwiseUnop(Math.atan,  'atanElem',  (z: any) => _cxImpl()._cAtan(z));
const sinhElem  = _makeElementwiseUnop(Math.sinh,  'sinhElem',  (z: any) => _cxImpl()._cSinh(z));
const coshElem  = _makeElementwiseUnop(Math.cosh,  'coshElem',  (z: any) => _cxImpl()._cCosh(z));
const tanhElem  = _makeElementwiseUnop(Math.tanh,  'tanhElem',  (z: any) => _cxImpl()._cTanh(z));
const asinhElem = _makeElementwiseUnop(Math.asinh, 'asinhElem', (z: any) => _cxImpl()._cAsinh(z));
const acoshElem = _makeElementwiseUnop(Math.acosh, 'acoshElem', (z: any) => _cxImpl()._cAcosh(z));
const atanhElem = _makeElementwiseUnop(Math.atanh, 'atanhElem', (z: any) => _cxImpl()._cAtanh(z));

// =====================================================================
// Closure of ARITH_OPS_N — pos / ifelse / link functions / casts
// =====================================================================
//
// The last ops still flowing through the legacy ARITH_OPS_N broadcast
// path; with these registered as variants, ARITH_OPS_N can retire.

// Pure identity (spec §07: pos = unary +).
const posElem = _makeElementwiseUnop((x: any) => +x, 'posElem');

// Type-restrictor casts (spec §03 lattice booleans ⊂ integers ⊂ reals).
const booleanElem = _makeElementwiseUnop((x: any) => x ? 1 : 0, 'booleanElem');
const integerElem = _makeElementwiseUnop((x: any) => Math.trunc(x), 'integerElem');

// Link functions (spec §07 GLM family helpers). Implementations
// mirror sampler.ARITH_OPS — kept in sync via the test suite.
// Avoiding a `require('./sampler.ts')` here because value-ops loads
// inside sampler.ts's require cycle; this module must not eagerly
// pull sampler back.
const _SQRT2 = Math.SQRT2;
function _erf(x: number): number {
  // Abramowitz-Stegun 7.1.26 approximation — accurate to ~1e-7,
  // matches what stdlib's erf returns to within the JS Math
  // function precision. Used by probit/invprobit elementwise; the
  // single-point sampler.ARITH_OPS variants use stdlib for higher
  // precision but the values agree to ~1e-7.
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t)
    + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
const logitElem     = _makeElementwiseUnop(
  (p: any) => Math.log(p / (1 - p)), 'logitElem');
const invlogitElem  = _makeElementwiseUnop(
  (x: any) => 1 / (1 + Math.exp(-x)), 'invlogitElem');
const probitElem    = _makeElementwiseUnop(
  (p: any) => _SQRT2 * _erfInv(2 * p - 1), 'probitElem');
const invprobitElem = _makeElementwiseUnop(
  (x: any) => 0.5 * (1 + _erf(x / _SQRT2)), 'invprobitElem');

// Inverse error function via Winitzki's approximation (max relative
// error ~1.3e-4 over the whole range, much better near the centre).
// Single-point sampler uses stdlib's erfcinv for full precision;
// elementwise broadcast paths use this approximation when atomCount
// is large enough that per-element stdlib calls would dominate.
function _erfInv(x: number): number {
  const a = 0.147;
  const ln = Math.log(1 - x * x);
  const term = 2 / (Math.PI * a) + ln / 2;
  const sign = x < 0 ? -1 : 1;
  return sign * Math.sqrt(Math.sqrt(term * term - ln / a) - term);
}

// ifelse(cond, then, else) — three-arg elementwise. Doesn't fit the
// _makeElementwiseBinop pattern; hand-rolled over flat data.
function ifelseElem(cond: any, thn: any, els: any): any {
  // Determine output shape by max-rank; broadcast singletons.
  const shapes = [cond, thn, els].map((v: any) =>
    isValue(v) ? v.shape : (typeof v === 'number' || typeof v === 'boolean'
      ? [] : null));
  if (shapes.some((s: any) => s == null)) {
    throw new Error('ifelseElem: all operands must be scalar or Value');
  }
  // Pick the output shape = the first non-empty shape; require any
  // others to match or be scalar (rank-0). Singleton-axis broadcasting
  // not supported in v1 here (matches the other elementwise impls).
  let outShape: number[] = [];
  for (const s of shapes) {
    if ((s as number[]).length > outShape.length) outShape = s as number[];
  }
  for (const s of shapes) {
    const sa = s as number[];
    if (sa.length === 0) continue;            // scalar OK
    if (sa.length !== outShape.length) {
      throw new Error('ifelseElem: rank mismatch ' + JSON.stringify(sa)
        + ' vs ' + JSON.stringify(outShape));
    }
    for (let k = 0; k < outShape.length; k++) {
      if (sa[k] !== outShape[k]) {
        throw new Error('ifelseElem: shape mismatch ' + JSON.stringify(sa)
          + ' vs ' + JSON.stringify(outShape));
      }
    }
  }
  const len = outShape.reduce((a: number, b: number) => a * b, 1) || 1;
  function readAt(v: any, i: number) {
    if (typeof v === 'number' || typeof v === 'boolean') return +v;
    return v.shape.length === 0 ? v.data[0] : v.data[i];
  }
  const out = new Float64Array(len);
  for (let i = 0; i < len; i++) {
    out[i] = readAt(cond, i) ? readAt(thn, i) : readAt(els, i);
  }
  return outShape.length === 0
    ? valueLib.scalar(out[0])
    : { shape: outShape.slice(), data: out };
}

// =====================================================================
// neg — pointwise negation
// =====================================================================
//
// Tag and shape are preserved; data is allocated fresh (caller may
// mutate the input independently after this returns).

function neg(a: Value): Value {
  if (!isValue(a)) throw new Error('value-ops.neg: argument must be a Value');
  if (valueLib.isDiagStored(a) && !a.im) {           // -diag stays diag
    const d = new Float64Array(a.data.length);
    for (let i = 0; i < d.length; i++) d[i] = -a.data[i];
    return valueLib.diagMatrix(d);
  }
  if (valueLib.isDiagStored(a)) a = valueLib.densify(a);
  // `a` widened to `any` via densify(); re-narrow via the predicate.
  const aA: any = a;
  if (isComplexValue(aA)) {
    // Negate raw stored buffers and keep the full Klein-4 tag — neg
    // commutes with transpose and conjugation, so no materialization is
    // needed (mirrors the real path: stored layout + tag preserved).
    const reO = new Float64Array(aA.data.length);
    const imO = new Float64Array(aA.im.length);
    for (let i = 0; i < reO.length; i++) reO[i] = -aA.data[i];
    for (let i = 0; i < imO.length; i++) imO[i] = -aA.im[i];
    const r: any = { shape: aA.shape.slice(), data: reO, im: imO, dtype: 'complex' };
    if (a.t && a.t !== 'N') r.t = a.t;
    return r;
  }
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = -a.data[i];
  const r: any = { shape: a.shape.slice(), data: out };
  if (a.t && a.t !== 'N') r.t = a.t;
  if (a.dtype) r.dtype = a.dtype;
  return r;
}

// =====================================================================
// Value ↔ nested-JS-array bridges
// =====================================================================
//
// The legacy linear-algebra implementations in sampler.js (Cholesky,
// LU, Gauss–Jordan, etc.) operate on nested JS arrays `M[i][j]`. To
// expose those ops on Values without re-implementing each algorithm
// on flat Float64Arrays (these are atom-
// indep one-shot ops where reallocation cost is negligible), we
// provide thin bridges: `_valueToNested` materialises a Value into
// nested form, honouring the Klein-4 transpose tag via index
// permutation. `_nestedToValue` packs the result back. The bridges
// support 1-D (vectors → flat JS array) and 2-D (matrices → nested
// JS array) shapes — the ranks the linalg ops accept.

function _valueToNested(v: any) {
  if (!isValue(v)) throw new Error('_valueToNested: not a Value');
  // Structured Values (vector-backed diag, future tri/sym) must be
  // materialized before nested indexing — the raw buffer is not a
  // dense row-major m×n layout. This single guard keeps every
  // nested-bridge linalg op (det / inv / linsolve / lower_cholesky /
  // trace / …) correct for structured inputs; O(n) fast-paths layer
  // on top at the op sites for the hot cases.
  if (v.struct !== undefined && (v.struct & valueLib.ST_OCC_MASK) !== valueLib.ST_DENSE) {
    v = valueLib.densify(v);
  }
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

function _nestedToValue(nested: any) {
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
// Atom-batched cross
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
// Deferred (uncommon today; lands when a use-case surfaces):
//   - shape=[N, m, n] × shape=[N, n]    (atom-batched matrix × vector)
//   - shape=[N] (batched scalar) ⊙ shape=[N, k]
//   - per-atom matrix × per-atom matrix

// Atom-batched matrix × matrix → atom-batched matrix. Per-atom
// matmul; supports three operand-batching patterns:
//   - A=[N,m,n] × B=[n,p]    → [N,m,p]
//   - A=[m,n]   × B=[N,n,p]  → [N,m,p]
//   - A=[N,m,n] × B=[N,n,p]  → [N,m,p]
// Tags on A or B (transpose / adjoint) are NOT supported in v1 —
// rank-3 transpose is a separate operation in spec §07 (transposes
// of rank>=3 swap the last two axes). The variant matcher rejects
// tagged operands at this rank.
function _matBatchedMatMul(A: any, B: any, N: any) {
  const aBatched = A.shape.length === 3 && A.shape[0] === N;
  const bBatched = B.shape.length === 3 && B.shape[0] === N;
  // Determine per-atom (m, n, p).
  const aShape = aBatched ? A.shape.slice(1) : A.shape;
  const bShape = bBatched ? B.shape.slice(1) : B.shape;
  if (aShape.length !== 2 || bShape.length !== 2) {
    throw new Error('mulN: matrix×matrix expected rank-2 per-atom shapes');
  }
  const [m, n] = aShape;
  const [bRows, p] = bShape;
  if (bRows !== n) {
    throw new Error(
      'mulN: matrix×matrix dimension mismatch per atom ('
      + JSON.stringify(aShape) + ' × ' + JSON.stringify(bShape) + ')');
  }
  if (isTransposeView(A) || isTransposeView(B)) {
    // Transposed rank-3 swaps the trailing two axes; the input would
    // already arrive untagged after the dispatcher resolves the tag.
    // Refuse here so the caller knows to densify-and-retry.
    throw new Error(
      'mulN: tagged matrix×matrix at rank-3 not supported '
      + '(densify the transposed operand first)');
  }
  const out = new Float64Array(N * m * p);
  const aStride = m * n;       // bytes per atom for A
  const bStride = n * p;       // bytes per atom for B
  const oStride = m * p;       // bytes per atom for output
  for (let atom = 0; atom < N; atom++) {
    const aBase = aBatched ? atom * aStride : 0;
    const bBase = bBatched ? atom * bStride : 0;
    const oBase = atom * oStride;
    for (let i = 0; i < m; i++) {
      const iRow = aBase + i * n;
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) {
          s += A.data[iRow + k] * B.data[bBase + k * p + j];
        }
        out[oBase + i * p + j] = s;
      }
    }
  }
  return { shape: [N, m, p], data: out };
}

// matrix(m, n) × shape=[N, n] → shape=[N, m]. Atom-major output;
// per-atom matvec with shared matrix. Tag on the matrix is honoured
// (BLAS gemm-flag style).
function _matBatchedVecMul(A: any, V: any, N: any) {
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
function _cxMatBatchedVecMul(A: any, V: any, N: any) {
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

// =====================================================================
// Atom-batched complex matrix × matrix (planar re/im storage)
// =====================================================================
//
// Three input patterns produce shape=[N, m, p] complex output:
//   - A=[N, m, n] × B=[n, p]    → shared B
//   - A=[m, n]    × B=[N, n, p] → shared A
//   - A=[N, m, n] × B=[N, n, p] → both per-atom
//
// Adjoint tags on A or B route through readComplex's conjugate
// folding (BLAS-gemm flag style); transpose flag on rank-3 inputs is
// not supported in v1 (matches the real twin _matBatchedMatMul).
function _cxMatBatchedMatMul(A: any, B: any, N: any) {
  const aBatched = A.shape.length === 3 && A.shape[0] === N;
  const bBatched = B.shape.length === 3 && B.shape[0] === N;
  const aShape = aBatched ? A.shape.slice(1) : A.shape;
  const bShape = bBatched ? B.shape.slice(1) : B.shape;
  if (aShape.length !== 2 || bShape.length !== 2) {
    throw new Error('mulN: complex matrix×matrix expected rank-2 per-atom shapes');
  }
  const [m, n] = aShape;
  const [bRows, p] = bShape;
  if (bRows !== n) {
    throw new Error(
      'mulN: complex matrix×matrix dimension mismatch per atom ('
      + JSON.stringify(aShape) + ' × ' + JSON.stringify(bShape) + ')');
  }
  if (isTransposeView(A) || isTransposeView(B)) {
    throw new Error(
      'mulN: tagged complex matrix×matrix at rank-3 not supported '
      + '(densify the transposed operand first)');
  }
  const cA = readComplex(A), cB = readComplex(B);
  const aStride = m * n, bStride = n * p, oStride = m * p;
  const re = new Float64Array(N * oStride), im = new Float64Array(N * oStride);
  for (let atom = 0; atom < N; atom++) {
    const aBase = aBatched ? atom * aStride : 0;
    const bBase = bBatched ? atom * bStride : 0;
    const oBase = atom * oStride;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let sr = 0, si = 0;
        for (let k = 0; k < n; k++) {
          const aIdx = aBase + i * n + k;
          const bIdx = bBase + k * p + j;
          const ar = cA.re[aIdx], aim = cA.im[aIdx];
          const br = cB.re[bIdx], bim = cB.im[bIdx];
          sr += ar * br - aim * bim;
          si += ar * bim + aim * br;
        }
        const oIdx = oBase + i * p + j;
        re[oIdx] = sr; im[oIdx] = si;
      }
    }
  }
  return _packCx(re, im, [N, m, p], false);
}

// Atom-indep value broadcast over the leading N axis of an atom-batched
// value, applied via a binary scalar fn. `batched` has shape=[N, ...rest];
// `indep` must have shape=rest (same orientation). Result has the
// batched shape.
function _atomBroadcastBinop(scalarFn: any, batched: any, indep: any, N: any, swapArgs: any, opName: any) {
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
  const r: any = { shape: batched.shape.slice(), data: out };
  if (batched.t && batched.t !== 'N') r.t = batched.t;
  if (batched.dtype) r.dtype = batched.dtype;
  return r;
}

// Atom-batched marker: leading axis is the atom count AND there is a
// non-trivial per-atom shape (rank ≥ 2). Delegates to the P3 canonical
// `value.isAtomBatched(v, N)` and additionally requires a non-trivial
// per-atom shape (rules out shape=[N] which is a batched scalar — the
// scalar-broadcast path handles that, not the matrix-vector dispatch
// this predicate gates).
function _hasAtomAxis(v: any, N: any) {
  return valueLib.isAtomBatched(v, N) && v.shape.length >= 2;
}

// mulN: atom-aware multiplication. Routes the MvNormal-style
// matrix × shape=[N, n] case to _matBatchedVecMul; otherwise delegates
// to the atom-indep `mul` (which already handles scalar broadcast,
// matmul, matvec etc. correctly when neither operand has an atom axis).
function mulN(a: any, b: any, N: any) {
  if (!isValue(a) || !isValue(b)) {
    throw new Error('value-ops.mulN: both operands must be Values');
  }
  const aBatched = _hasAtomAxis(a, N);
  const bBatched = _hasAtomAxis(b, N);
  // matrix × shape=[N, n]: a is shape [m, n], b is shape [N, n].
  if (!aBatched && bBatched
      && a.shape.length === 2 && b.shape.length === 2) {
    // Diagonal matrix × per-atom vector: O(N·n) elementwise scale —
    // no densification, no gemm (the diagonal-covariance MvNormal
    // `mu + L·z` path). Complex diag densifies (rare; cov is real).
    if (valueLib.isDiagStored(a)) {
      if (a.im) { a = valueLib.densify(a); }
      else {
        const d = a.data, n = d.length;
        if (b.shape[1] !== n) {
          throw new Error('mulN: diag×batchedVector shape mismatch (' +
            JSON.stringify(a.shape) + ' × ' + JSON.stringify(b.shape) + ')');
        }
        if (isTransposeView(b)) {
          throw new Error('mulN: batched vector must be column-oriented');
        }
        const out = new Float64Array(N * n);
        for (let atom = 0; atom < N; atom++) {
          const base = atom * n;
          for (let i = 0; i < n; i++) out[base + i] = d[i] * b.data[base + i];
        }
        return { shape: [N, n], data: out };
      }
    }
    if (_isCx(a, b)) return _cxMatBatchedVecMul(a, b, N);
    return _matBatchedVecMul(a, b, N);
  }
  // Atom-indep case.
  if (!aBatched && !bBatched) return mul(a, b);
  // Other atom-batched cases land when needed.
  throw new Error(
    'mulN: unsupported atom-batched shape combination ' +
    JSON.stringify(a.shape) + ' × ' + JSON.stringify(b.shape) +
    ' with N=' + N);
}

// addN / subN: atom-aware. Handles the atom-indep + atom-batched
// broadcast that MvNormal's `mu + L*z` needs (mu is atom-indep,
// L*z is atom-batched).
function _makeAtomAwareBinop(scalarFn: any, atomIndepImpl: any, opName: any) {
  return function atomAwareBinop(a: any, b: any, N: any) {
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

const addN = _makeAtomAwareBinop((x: any, y: any) => x + y, add, 'add');
const subN = _makeAtomAwareBinop((x: any, y: any) => x - y, sub, 'sub');

// negN: pointwise — the atom-indep neg already iterates over the
// whole data buffer regardless of rank, so atom-batched values are
// handled correctly without extra plumbing.
function negN(a: any, _N: any) {
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
  // engine-concepts §20.1 — batched-elementwise primitives for the
  // canonical `broadcasted(<op>)` form. Used by the broadcast
  // dispatcher's fast path when the head is a known scalar primitive.
  mulElem,
  divElem,
  powElem,
  modElem,
  expElem,
  logElem,
  sqrtElem,
  sinElem,
  cosElem,
  tanElem,
  absElem,
  abs2Elem,
  log10Elem,
  log1pElem,
  expm1Elem,
  floorElem,
  ceilElem,
  roundElem,
  // P9 additions — broadcast variants for the remaining ARITH_OPS_N
  // ops. Each is a one-liner over the shared _makeElementwise*
  // factories; collectively retire the broadcast1/2/3 dispatch
  // fallback for these ops in favour of the unified variant
  // registry.
  minElem, maxElem,
  ltElem, leElem, gtElem, geElem, equalElem, unequalElem,
  isfiniteElem, isinfElem, isnanElem, iszeroElem,
  landElem, lorElem, lxorElem, lnotElem,
  atan2Elem,
  asinElem, acosElem, atanElem,
  sinhElem, coshElem, tanhElem,
  asinhElem, acoshElem, atanhElem,
  // Closure of ARITH_OPS_N coverage (ifelse / link functions / casts).
  posElem,
  booleanElem, integerElem,
  logitElem, invlogitElem, probitElem, invprobitElem,
  ifelseElem,
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
  _matBatchedMatMul,
  _cxMatBatchedMatMul,
  _atomBroadcastBinop,
  _scalarBroadcastMul,
  _scalarBroadcastBinop,
  _matIdxN,
  _matIdxT,
};
