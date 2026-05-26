'use strict';

// =====================================================================
// ops-declarations.ts — per-op `OpDecl` registrations
// =====================================================================
//
// Single point where ops register themselves with `ops.ts`. As the
// unified declaration model matures (engine-concepts §17.x), more ops
// migrate here from their per-file ARITH_OPS entries / signature
// factories.
//
// Phase 1 scope: `cross` only (proof-of-concept). The existing
// `ARITH_OPS.cross` stays as-is; the declaration here exposes the
// same impl via `ops.dispatch('cross', [a, b])` and the conformance
// suite (test/ops-conformance.test.ts) verifies the two paths agree
// on random inputs.
//
// Phase 2 will move the natural home of each op's impl into its
// declaration and have `evaluateCall` route declared ops through
// `ops.dispatch`, retiring the per-file entries.

const ops = require('./ops.ts');
const valueLib = require('./value.ts');
const valueOps = require('./value-ops.ts');
const linalg = require('./sampler-linalg.ts');

// Type-AST builders (re-implemented locally to avoid a require cycle
// with types.ts; the shapes match what types.SIGNATURE_FACTORIES
// produces). Once types.ts learns to consult ops.signatureOf, this
// duplication retires.
function _array(rank: number, shape: any[], elem: any) {
  return { kind: 'array', rank, shape, elem };
}
const _REAL = { kind: 'scalar', prim: 'real' };

// =====================================================================
// cross(a, b) — 3-D vector cross product (spec §07)
// =====================================================================
//
// Logical shape: vec3 × vec3 → vec3 (rank 1 inputs, rank 1 output).
// The atom-batched form (shape=[N, 3]) is handled by the dispatcher's
// per-atom fallback — no `batched` fast-path yet (Phase 3 follow-up).
// Logical impl mirrors `ARITH_OPS.cross` exactly; the conformance
// suite pins equivalence.

function _crossLogical(a: any, b: any): any {
  const aIsVal = valueLib.isValue(a);
  const bIsVal = valueLib.isValue(b);
  const wantValue = aIsVal || bIsVal;
  function asLen3(v: any, isVal: boolean): { re: any; im: any | null } {
    if (isVal) {
      const D = valueLib.densify(v);
      if (D.shape.length !== 1 || D.shape[0] !== 3) {
        throw new Error('cross: argument must be a length-3 vector, got shape='
          + JSON.stringify(D.shape));
      }
      const c = valueLib.readComplex(D);
      return { re: c.re, im: valueLib.isComplexValue(D) ? c.im : null };
    }
    if (!v || typeof v.length !== 'number' || v.length !== 3) {
      throw new Error('cross: argument must be a length-3 vector');
    }
    let anyComplex = false;
    for (let k = 0; k < 3; k++) {
      const e = v[k];
      if (e != null && typeof e === 'object'
          && typeof e.re === 'number' && typeof e.im === 'number') {
        anyComplex = true; break;
      }
    }
    const re = new Float64Array(3);
    const im = anyComplex ? new Float64Array(3) : null;
    for (let k = 0; k < 3; k++) {
      const e = v[k];
      if (e != null && typeof e === 'object'
          && typeof e.re === 'number' && typeof e.im === 'number') {
        re[k] = e.re; if (im) im[k] = e.im;
      } else {
        re[k] = +e;
      }
    }
    return { re, im };
  }
  const A = asLen3(a, aIsVal);
  const B = asLen3(b, bIsVal);
  const hasComplex = A.im !== null || B.im !== null;
  if (hasComplex) {
    const aR = A.re, aI = A.im || new Float64Array(3);
    const bR = B.re, bI = B.im || new Float64Array(3);
    const cR = new Float64Array(3);
    const cI = new Float64Array(3);
    const idx: Array<[number, number]> = [[1, 2], [2, 0], [0, 1]];
    for (let k = 0; k < 3; k++) {
      const [u, v] = idx[k];
      const p_re = aR[u] * bR[v] - aI[u] * bI[v];
      const p_im = aR[u] * bI[v] + aI[u] * bR[v];
      const q_re = aR[v] * bR[u] - aI[v] * bI[u];
      const q_im = aR[v] * bI[u] + aI[v] * bR[u];
      cR[k] = p_re - q_re;
      cI[k] = p_im - q_im;
    }
    return valueLib.complexValue(cR, cI, [3]);
  }
  const aR = A.re, bR = B.re;
  const out = new Float64Array(3);
  out[0] = aR[1] * bR[2] - aR[2] * bR[1];
  out[1] = aR[2] * bR[0] - aR[0] * bR[2];
  out[2] = aR[0] * bR[1] - aR[1] * bR[0];
  if (wantValue) return { shape: [3], data: out };
  return out;
}

// Batched fast-path for cross: real-only, atom-major Float64Array
// loop over `[N, 3]` buffers. ~N× faster than the per-atom fallback
// (one JS call instead of N). Complex inputs and non-atom-batched
// shapes drop back to the dispatcher's per-atom fallback; the
// conformance harness pins both paths agree.
function _crossBatched(args: any[], N: number): any {
  const [a, b] = args;
  // Only handle the common case: both args are real Values of shape
  // [N, 3] or [3]. Anything else falls back to per-atom via the
  // dispatcher (this returns null sentinel below to signal fallback).
  function asAtomMajor(v: any): { data: Float64Array; isBatched: boolean } | null {
    if (!valueLib.isValue(v)) return null;
    if (v.dtype === 'complex') return null;             // complex → fallback
    if (v.shape.length === 1 && v.shape[0] === 3) {
      return { data: v.data, isBatched: false };
    }
    if (v.shape.length === 2 && v.shape[0] === N && v.shape[1] === 3) {
      return { data: v.data, isBatched: true };
    }
    return null;
  }
  const A = asAtomMajor(a);
  const B = asAtomMajor(b);
  if (!A || !B) {
    // Signal fallback — dispatcher would otherwise have called us
    // unconditionally. Easiest way is to do the per-atom work here
    // ourselves; matches the dispatcher's stitching semantics.
    return null;
  }
  // Tight Float64Array loop. Reads:
  //   a stride: 3 if batched, else 0 (broadcast)
  //   b stride: 3 if batched, else 0
  const aS = A.isBatched ? 3 : 0;
  const bS = B.isBatched ? 3 : 0;
  const aD = A.data, bD = B.data;
  const out = new Float64Array(N * 3);
  for (let i = 0; i < N; i++) {
    const ao = i * aS, bo = i * bS;
    out[i * 3 + 0] = aD[ao + 1] * bD[bo + 2] - aD[ao + 2] * bD[bo + 1];
    out[i * 3 + 1] = aD[ao + 2] * bD[bo + 0] - aD[ao + 0] * bD[bo + 2];
    out[i * 3 + 2] = aD[ao + 0] * bD[bo + 1] - aD[ao + 1] * bD[bo + 0];
  }
  return { shape: [N, 3], data: out };
}

// Wrapper that calls the fast-path when applicable, falls back to
// the dispatcher's per-atom path otherwise. The dispatcher itself
// invokes `batched` unconditionally for atom-batched inputs, so we
// handle "not eligible for fast-path" by running the per-atom loop
// inline (same as what the dispatcher would do without `batched`).
function _crossBatchedOrFallback(args: any[], N: number): any {
  const fast = _crossBatched(args, N);
  if (fast !== null) return fast;
  // Fallback: per-atom logical N times, stack. Same machinery the
  // dispatcher uses; we invoke directly here.
  const perAtom: any[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const sliced = args.map((v: any) => {
      if (valueLib.isValue(v) && v.shape.length === 2 && v.shape[0] === N) {
        const tailLen = v.shape[1];
        const subData = v.data.subarray(i * tailLen, (i + 1) * tailLen);
        const sub: any = { shape: [tailLen], data: subData };
        if (v.dtype === 'complex' && v.im) {
          sub.dtype = 'complex';
          sub.im = v.im.subarray(i * tailLen, (i + 1) * tailLen);
        }
        return sub;
      }
      return v;
    });
    perAtom[i] = _crossLogical(sliced[0], sliced[1]);
  }
  // Stack to shape=[N, 3]. Complex outputs preserve im part.
  const first = perAtom[0];
  const isComplex = valueLib.isComplexValue(first);
  const out = new Float64Array(N * 3);
  const outIm = isComplex ? new Float64Array(N * 3) : null;
  for (let i = 0; i < N; i++) {
    out.set(perAtom[i].data, i * 3);
    if (outIm && perAtom[i].im) outIm.set(perAtom[i].im, i * 3);
  }
  if (outIm) return valueLib.complexValue(out, outIm, [N, 3]);
  return { shape: [N, 3], data: out };
}

ops.register({
  name: 'cross',
  signature: {
    args: [_array(1, [3], _REAL), _array(1, [3], _REAL)],
    kwargs: {},
    result: _array(1, [3], _REAL),
  },
  argRanks: [1, 1],
  logical: _crossLogical,
  batched: _crossBatchedOrFallback,
});

// =====================================================================
// self_outer(v) — vᵀ × v outer product (spec §07)
// =====================================================================
//
// Logical shape: vec(n) → mat(n, n) (rank 1 input, rank 2 output).
// Tests the abstraction's "logical-rank-+-1 = atom-batched" detection
// at a higher rank than `cross` (where the logical output is also
// non-scalar — exercises `_stackPerAtom`'s shape-preservation).

function _selfOuterLogical(v: any): any {
  if (valueLib.isValue(v)) {
    const D = valueLib.densify(v);
    if (D.shape.length !== 1) {
      throw new Error('self_outer: argument must be a rank-1 vector, got shape='
        + JSON.stringify(D.shape));
    }
    const n = D.shape[0];
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) out[i * n + j] = D.data[i] * D.data[j];
    }
    return { shape: [n, n], data: out };
  }
  const n = v.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) row[j] = v[i] * v[j];
    out[i] = row;
  }
  return out;
}

ops.register({
  name: 'self_outer',
  signature: {
    args: [_array(1, ['%dynamic'], _REAL)],
    kwargs: {},
    result: _array(2, ['%dynamic', '%dynamic'], _REAL),
  },
  argRanks: [1],
  logical: _selfOuterLogical,
});

// =====================================================================
// trace(M) — sum of diagonal entries (square matrix → scalar)
// =====================================================================
//
// Phase 2 migration. Logical rank: 2 → 0. Diag-stored fast path
// retained: trace of a vector-backed diagonal Value is the sum of
// the diagonal vector (O(n)) without densification.

function _traceLogical(M: any): any {
  if (valueLib.isDiagStored(M) && !M.im) {
    let s = 0; for (let i = 0; i < M.data.length; i++) s += M.data[i];
    return valueLib.scalar(s);
  }
  if (valueLib.isValue(M)) {
    const D = valueLib.densify(M);
    const n = D.shape[0];
    if (n !== D.shape[1]) throw new Error('trace: argument must be a square matrix');
    let s = 0;
    for (let i = 0; i < n; i++) s += D.data[i * n + i];
    return valueLib.scalar(s);
  }
  if (!Array.isArray(M)) throw new Error('trace: argument must be a matrix');
  const n = M.length;
  if (n === 0 || M[0].length !== n) {
    throw new Error('trace: argument must be a square matrix');
  }
  let s = 0;
  for (let i = 0; i < n; i++) s += M[i][i];
  return s;
}

ops.register({
  name: 'trace',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _REAL,
  },
  argRanks: [2],
  logical: _traceLogical,
});

// =====================================================================
// diagmat(v) — diagonal-structured matrix from a vector
// =====================================================================
//
// Phase 2 migration. Logical rank: 1 → 2. Produces a vector-backed
// `diag` structured Value (O(n) storage); every diag-aware op
// fast-paths it, anything else densify()s. Complex diagonal carries
// its imaginary part on the same vector.

function _diagmatLogical(v: any): any {
  if (valueLib.isValue(v)) {
    if (v.shape.length !== 1) {
      throw new Error('diagmat: argument must be a rank-1 vector, got shape='
        + JSON.stringify(v.shape));
    }
    return valueLib.diagMatrix(v.data, v.im);
  }
  return valueLib.diagMatrix(v instanceof Float64Array ? v : Float64Array.from(v));
}

ops.register({
  name: 'diagmat',
  signature: {
    args: [_array(1, ['%dynamic'], _REAL)],
    kwargs: {},
    result: _array(2, ['%dynamic', '%dynamic'], _REAL),
  },
  argRanks: [1],
  logical: _diagmatLogical,
});

// =====================================================================
// det(A) — determinant of a square matrix (LU with partial pivoting)
// =====================================================================
//
// Phase 2 migration. Logical rank: 2 → 0. Diag-stored fast path:
// product of the diagonal (O(n)).

function _detLogical(A: any): any {
  if (valueLib.isDiagStored(A) && !A.im) {
    let p = 1; for (let i = 0; i < A.data.length; i++) p *= A.data[i];
    return valueLib.scalar(p);
  }
  if (valueLib.isValue(A)) {
    return valueLib.scalar(linalg._detLUValue(valueLib.densify(A)));
  }
  if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
    throw new Error('det: argument must be a non-empty square matrix');
  }
  return linalg._detLU(A);
}

ops.register({
  name: 'det',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _REAL,
  },
  argRanks: [2],
  logical: _detLogical,
});

// =====================================================================
// logabsdet(A) — log|det(A)| via LU (numerically stable on near-singular)
// =====================================================================

function _logabsdetLogical(A: any): any {
  if (valueLib.isDiagStored(A) && !A.im) {
    let s = 0;
    for (let i = 0; i < A.data.length; i++) s += Math.log(Math.abs(A.data[i]));
    return valueLib.scalar(s);
  }
  if (valueLib.isValue(A)) {
    return valueLib.scalar(linalg._logAbsDetLUValue(valueLib.densify(A)));
  }
  if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
    throw new Error('logabsdet: argument must be a non-empty square matrix');
  }
  return linalg._logAbsDetLU(A);
}

ops.register({
  name: 'logabsdet',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _REAL,
  },
  argRanks: [2],
  logical: _logabsdetLogical,
});

// =====================================================================
// inv(A) — matrix inverse via LU + back-substitution against I
// =====================================================================
//
// Phase 2 migration. Diag-stored fast path: reciprocal of the
// diagonal (O(n)). Throws on singular matrices.

function _invLogical(A: any): any {
  if (valueLib.isDiagStored(A) && !A.im) {
    const d = new Float64Array(A.data.length);
    for (let i = 0; i < d.length; i++) {
      if (A.data[i] === 0) throw new Error('inv: singular diagonal matrix');
      d[i] = 1 / A.data[i];
    }
    return valueLib.diagMatrix(d);
  }
  if (valueLib.isValue(A)) {
    return linalg._invValue(valueLib.densify(A));
  }
  if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
    throw new Error('inv: argument must be a non-empty square matrix');
  }
  return linalg._invGaussJordan(A);
}

// Batched inv: per-atom LU + back-sub on an atom-major [N, n, n]
// Float64Array. Skips the per-atom JS dispatch overhead; the LU
// factorisation itself is the same algorithm `_invValue` uses.
// Diag-stored Values fall through to per-atom (the diag-fast-path is
// O(n) per atom and rarely the bottleneck).
function _invBatched(args: any[], N: number): any {
  const [A] = args;
  if (!valueLib.isValue(A) || A.shape.length !== 3
      || A.shape[0] !== N || A.dtype === 'complex'
      || valueLib.isDiagStored(A)) {
    return null;  // signal fallback
  }
  const n = A.shape[1];
  if (n !== A.shape[2]) {
    throw new Error('inv: per-atom matrix must be square, got shape='
      + JSON.stringify(A.shape));
  }
  const out = new Float64Array(N * n * n);
  const stride = n * n;
  // Per-atom slice → linalg._invValue, copy into output buffer.
  for (let i = 0; i < N; i++) {
    const slice = A.data.subarray(i * stride, (i + 1) * stride);
    const atomA = { shape: [n, n], data: slice };
    const atomInv = linalg._invValue(atomA);
    out.set(atomInv.data, i * stride);
  }
  return { shape: [N, n, n], data: out };
}

function _invBatchedOrFallback(args: any[], N: number): any {
  const fast = _invBatched(args, N);
  if (fast !== null) return fast;
  // Per-atom fallback inline (diag-stored Values, complex, anything
  // non-batched-Value).
  const perAtom: any[] = new Array(N);
  const A = args[0];
  for (let i = 0; i < N; i++) {
    let atomA: any = A;
    if (valueLib.isValue(A) && A.shape.length === 3 && A.shape[0] === N) {
      const stride = A.shape[1] * A.shape[2];
      const subData = A.data.subarray(i * stride, (i + 1) * stride);
      atomA = { shape: [A.shape[1], A.shape[2]], data: subData };
    }
    perAtom[i] = _invLogical(atomA);
  }
  // All results should be matrices of consistent shape; densify any
  // diag-stored entries (rare, but possible if diag fast-path fired
  // per atom) before stacking.
  const tailShape = valueLib.isValue(perAtom[0])
    ? valueLib.densify(perAtom[0]).shape
    : [perAtom[0].length, perAtom[0][0].length];
  const tailLen = tailShape.reduce((a: number, b: number) => a * b, 1);
  const out = new Float64Array(N * tailLen);
  for (let i = 0; i < N; i++) {
    const d = valueLib.isValue(perAtom[i])
      ? valueLib.densify(perAtom[i]).data
      : perAtom[i];
    out.set(d, i * tailLen);
  }
  return { shape: [N, ...tailShape], data: out };
}

ops.register({
  name: 'inv',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _array(2, ['%dynamic', '%dynamic'], _REAL),
  },
  argRanks: [2],
  logical: _invLogical,
  batched: _invBatchedOrFallback,
});

// =====================================================================
// lower_cholesky(A) — lower-triangular L with A = L Lᵀ (PD A)
// =====================================================================
//
// Phase 2 migration. Diag-stored PD A: L = √diag (still diagonal,
// still lower-triangular). O(n). Throws if A is not PD.

function _lowerCholeskyLogical(A: any): any {
  if (valueLib.isDiagStored(A) && !A.im) {
    const d = new Float64Array(A.data.length);
    for (let i = 0; i < d.length; i++) {
      if (!(A.data[i] > 0)) {
        throw new Error('lower_cholesky: matrix is not positive definite');
      }
      d[i] = Math.sqrt(A.data[i]);
    }
    return valueLib.diagMatrix(d);
  }
  if (valueLib.isValue(A)) {
    return linalg._choleskyValue(valueLib.densify(A));
  }
  if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
    throw new Error('lower_cholesky: argument must be a non-empty square matrix');
  }
  return linalg._cholesky(A);
}

// Batched lower_cholesky: per-atom Cholesky-Banachiewicz on
// [N, n, n] atom-major Float64Array. Same skip-criteria as inv.
function _lowerCholeskyBatched(args: any[], N: number): any {
  const [A] = args;
  if (!valueLib.isValue(A) || A.shape.length !== 3
      || A.shape[0] !== N || A.dtype === 'complex'
      || valueLib.isDiagStored(A)) {
    return null;
  }
  const n = A.shape[1];
  if (n !== A.shape[2]) {
    throw new Error('lower_cholesky: per-atom matrix must be square, got shape='
      + JSON.stringify(A.shape));
  }
  const out = new Float64Array(N * n * n);
  const stride = n * n;
  for (let i = 0; i < N; i++) {
    const slice = A.data.subarray(i * stride, (i + 1) * stride);
    const atomA = { shape: [n, n], data: slice };
    const L = linalg._choleskyValue(atomA);
    out.set(L.data, i * stride);
  }
  return { shape: [N, n, n], data: out };
}

function _lowerCholeskyBatchedOrFallback(args: any[], N: number): any {
  const fast = _lowerCholeskyBatched(args, N);
  if (fast !== null) return fast;
  const perAtom: any[] = new Array(N);
  const A = args[0];
  for (let i = 0; i < N; i++) {
    let atomA: any = A;
    if (valueLib.isValue(A) && A.shape.length === 3 && A.shape[0] === N) {
      const stride = A.shape[1] * A.shape[2];
      const subData = A.data.subarray(i * stride, (i + 1) * stride);
      atomA = { shape: [A.shape[1], A.shape[2]], data: subData };
    }
    perAtom[i] = _lowerCholeskyLogical(atomA);
  }
  const tailShape = valueLib.isValue(perAtom[0])
    ? valueLib.densify(perAtom[0]).shape
    : [perAtom[0].length, perAtom[0][0].length];
  const tailLen = tailShape.reduce((a: number, b: number) => a * b, 1);
  const out = new Float64Array(N * tailLen);
  for (let i = 0; i < N; i++) {
    const d = valueLib.isValue(perAtom[i])
      ? valueLib.densify(perAtom[i]).data
      : perAtom[i];
    out.set(d, i * tailLen);
  }
  return { shape: [N, ...tailShape], data: out };
}

ops.register({
  name: 'lower_cholesky',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _array(2, ['%dynamic', '%dynamic'], _REAL),
  },
  argRanks: [2],
  logical: _lowerCholeskyLogical,
  batched: _lowerCholeskyBatchedOrFallback,
});

// =====================================================================
// row_gram(A) = A · Aᵀ   /   col_gram(A) = Aᵀ · A   (Hermitian Gram)
// =====================================================================
//
// Phase 2 migration. For real A the conj bit is a numerical no-op
// (so this matches the older transpose form); for complex A the
// adjoint is required for the Gram to be Hermitian. Value path
// folds the Klein-4 conj+swap tags at value-ops.mul dispatch (no
// transpose/conjugate materialisation). Useful for LKJ ↔
// LKJCholesky conversions and Gram-matrix priors.

function _rowGramLogical(A: any): any {
  if (valueLib.isValue(A)) return valueOps.mul(A, valueLib.adjoint(A));
  // Nested-array path uses ARITH_OPS.transpose, which isn't migrated
  // to ops.ts yet. Lazy-require to avoid a module-load cycle.
  const samplerMod = require('./sampler.ts');
  return linalg._matmul(A, samplerMod._internal.ARITH_OPS.transpose(A));
}

ops.register({
  name: 'row_gram',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _array(2, ['%dynamic', '%dynamic'], _REAL),
  },
  argRanks: [2],
  logical: _rowGramLogical,
});

function _colGramLogical(A: any): any {
  if (valueLib.isValue(A)) return valueOps.mul(valueLib.adjoint(A), A);
  const samplerMod = require('./sampler.ts');
  return linalg._matmul(samplerMod._internal.ARITH_OPS.transpose(A), A);
}

ops.register({
  name: 'col_gram',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _array(2, ['%dynamic', '%dynamic'], _REAL),
  },
  argRanks: [2],
  logical: _colGramLogical,
});

// =====================================================================
// Rank-polymorphic ops (Phase 5a — engine-concepts §18.7)
// =====================================================================
//
// transpose / adjoint accept vector OR matrix; linsolve(A, b) takes b
// as vector OR matrix. The argRanks-driven atom-batch detection
// can't disambiguate these from atom-batched lower-rank inputs at
// the runtime level, so the dispatcher's contract for `kind:
// 'rank-polymorphic'` is "call logical with the inputs as-is" —
// atom-batched semantics over a rank-polymorphic op require explicit
// `broadcast(fn(op(_)), …)` wrapping by the caller.
//
// The signatures remain in `types.SIGNATURE_FACTORIES` (they use the
// `special: '<op>'` marker that typeinfer special-cases for output-
// shape inference based on the actual input type). No `signature`
// field on the OpDecl — `signatureOf` falls through to the legacy
// table for these names.

function _transposeLogical(M: any): any {
  if (valueLib.isValue(M)) return valueLib.transpose(M);  // O(1) tag flip
  if (!Array.isArray(M) || M.length === 0) return [];
  const rows = M.length, cols = M[0].length;
  const out = new Array(cols);
  for (let j = 0; j < cols; j++) {
    const row = new Array(rows);
    for (let i = 0; i < rows; i++) row[i] = M[i][j];
    out[j] = row;
  }
  return out;
}

ops.register({
  name: 'transpose',
  kind: 'rank-polymorphic',
  logical: _transposeLogical,
});

function _adjointLogical(M: any): any {
  if (valueLib.isValue(M)) return valueLib.adjoint(M);    // O(1) tag flip
  // Real-only nested-array path falls back to transpose (no complex
  // numbers in nested arrays at this layer).
  return _transposeLogical(M);
}

ops.register({
  name: 'adjoint',
  kind: 'rank-polymorphic',
  logical: _adjointLogical,
});

// linsolve(A, b): A is square matrix (rank 2); b is vector (rank 1)
// or matrix (rank 2). Diagonal A: vector b → reciprocal × b (O(n)).
// Otherwise: LU + back-substitution.
function _linsolveLogical(A: any, b: any): any {
  if (valueLib.isDiagStored(A) && !A.im) {
    const d = A.data, m = d.length;
    const bv = valueLib.isValue(b) ? b : null;
    const bd = bv ? bv.data : b;
    if (bd && bd.length === m && (!bv || bv.shape.length === 1)) {
      const x = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        if (d[i] === 0) throw new Error('linsolve: singular diagonal matrix');
        x[i] = bd[i] / d[i];
      }
      return bv ? valueLib.vector(x) : Array.from(x);
    }
  }
  if (valueLib.isValue(A) || valueLib.isValue(b)) {
    const aV = valueLib.isValue(A) ? valueLib.densify(A) : valueLib.asValue(A);
    const bV = valueLib.isValue(b) ? valueLib.densify(b) : valueLib.asValue(b);
    return linalg._linsolveLUValue(aV, bV);
  }
  if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
    throw new Error('linsolve: A must be a non-empty square matrix');
  }
  return linalg._linsolveLU(A, b);
}

ops.register({
  name: 'linsolve',
  kind: 'rank-polymorphic',
  logical: _linsolveLogical,
});

// =====================================================================
// Variadic ops (Phase 5b — engine-concepts §18.7)
// =====================================================================
//
// vector / cat take a variable number of positional args. The
// dispatcher's variadic kind just forwards all args to `logical`;
// no atom-batch detection (variadic + batching is a Phase 5c+
// problem). Same delegation pattern as the rank-polymorphic ops.

function _vectorLogical(...xs: any[]): any {
  if (xs.length === 0) return { shape: [0], data: new Float64Array(0) };
  let allScalar = true;
  for (let i = 0; i < xs.length; i++) {
    const t = typeof xs[i];
    if (t !== 'number' && t !== 'boolean') { allScalar = false; break; }
  }
  if (!allScalar) return xs;
  const data = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    data[i] = xs[i] === true ? 1 : xs[i] === false ? 0 : +xs[i];
  }
  return { shape: [xs.length], data: data };
}

ops.register({
  name: 'vector',
  kind: 'variadic',
  logical: _vectorLogical,
});

function _catLogical(...xs: any[]): any {
  if (xs.length === 0) return { shape: [0], data: new Float64Array(0) };
  const first = xs[0];
  // cat(scalar, scalar, ...) → rank-1 Value (spec §07).
  if (typeof first === 'number' || typeof first === 'boolean') {
    const data = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      data[i] = xs[i] === true ? 1 : xs[i] === false ? 0 : +xs[i];
    }
    return { shape: [xs.length], data };
  }
  // cat(vector, vector, ...) → rank-1 Value (concatenation along
  // the only axis).
  if (valueLib.isValue(first)
      || (first && first.BYTES_PER_ELEMENT !== undefined)
      || Array.isArray(first)) {
    let total = 0;
    for (let j = 0; j < xs.length; j++) {
      const v = xs[j];
      total += valueLib.isValue(v) ? v.data.length : v.length;
    }
    const out = new Float64Array(total);
    let pos = 0;
    for (let j = 0; j < xs.length; j++) {
      const v = xs[j];
      const src = valueLib.isValue(v) ? v.data : v;
      for (let i = 0; i < src.length; i++) out[pos++] = +src[i];
    }
    return { shape: [total], data: out };
  }
  // cat(record, record, ...) → merged record (duplicate keys are a
  // static error per spec §07).
  if (first && typeof first === 'object') {
    const out: Record<string, any> = {};
    for (let j = 0; j < xs.length; j++) {
      const r = xs[j];
      for (const k in r) {
        if (!Object.prototype.hasOwnProperty.call(r, k)) continue;
        if (k in out) {
          throw new Error("cat: duplicate field '" + k + "'");
        }
        out[k] = r[k];
      }
    }
    return out;
  }
  throw new Error('cat: unsupported argument shape (got '
    + (typeof first) + ')');
}

ops.register({
  name: 'cat',
  kind: 'variadic',
  logical: _catLogical,
});

// =====================================================================
// Higher-order ops (Phase 5c — engine-concepts §18.8)
// =====================================================================
//
// reduce / scan / filter take a callable + data inputs. Their
// `logical` receives raw IR args plus a `ctx` carrying the engine's
// env + evaluateExpr + resolveFn (the higher-order dispatch contract,
// see ops.ts). The op resolves the callable, iterates over the data,
// and evaluates the body in an env extended per iteration.
//
// Atom-batched semantics for these ops are not handled
// automatically — `reduce` / `scan` / `filter` over an atom-batched
// array would require explicit per-atom broadcasting in the caller.
// Today's `_perAtomFallback` in `evaluateExprN` provides this for
// callers that need it.

function _reduceLogical(ir: any, ctx: any): any {
  const irArgs = ir.args || [];
  if (irArgs.length !== 2) {
    throw new Error('reduce: expected 2 args (function, xs), got ' + irArgs.length);
  }
  const fn = ctx.resolveFn(irArgs[0], ctx.env);
  if (!fn || fn.params.length !== 2) {
    throw new Error('reduce: function arg must be a binary function');
  }
  const xsRaw: any = ctx.evaluateExpr(irArgs[1], ctx.env);
  const xs: any = valueLib.isValue(xsRaw) ? xsRaw.data : xsRaw;
  if (!Array.isArray(xs) && !(xs && xs.BYTES_PER_ELEMENT)) {
    throw new Error('reduce: xs must be a vector');
  }
  if (xs.length === 0) {
    throw new Error('reduce: empty vector has no initial value');
  }
  const elemEnv = Object.assign({}, ctx.env);
  let acc: any = xs[0];
  for (let i = 1; i < xs.length; i++) {
    elemEnv[fn.params[0]] = acc;
    elemEnv[fn.params[1]] = xs[i];
    acc = ctx.evaluateExpr(fn.body, elemEnv);
  }
  return acc;
}

ops.register({
  name: 'reduce',
  kind: 'higher-order',
  logical: _reduceLogical,
});

function _scanLogical(ir: any, ctx: any): any {
  const irArgs = ir.args || [];
  if (irArgs.length !== 3) {
    throw new Error('scan: expected 3 args (function, init, xs), got ' + irArgs.length);
  }
  const fn = ctx.resolveFn(irArgs[0], ctx.env);
  if (!fn || fn.params.length !== 2) {
    throw new Error('scan: function arg must be a binary function');
  }
  const init = ctx.evaluateExpr(irArgs[1], ctx.env);
  const xsRaw: any = ctx.evaluateExpr(irArgs[2], ctx.env);
  const xs: any = valueLib.isValue(xsRaw) ? xsRaw.data : xsRaw;
  if (!Array.isArray(xs) && !(xs && xs.BYTES_PER_ELEMENT)) {
    throw new Error('scan: xs must be a vector');
  }
  const n = xs.length;
  const out: Float64Array = new Float64Array(n);
  const elemEnv = Object.assign({}, ctx.env);
  let acc = init;
  for (let i = 0; i < n; i++) {
    elemEnv[fn.params[0]] = acc;
    elemEnv[fn.params[1]] = xs[i];
    acc = ctx.evaluateExpr(fn.body, elemEnv);
    out[i] = acc === true ? 1 : acc === false ? 0 : +acc;
  }
  return { shape: [n], data: out };
}

ops.register({
  name: 'scan',
  kind: 'higher-order',
  logical: _scanLogical,
});

function _filterLogical(ir: any, ctx: any): any {
  const irArgs = ir.args || [];
  if (irArgs.length !== 2) {
    throw new Error('filter: expected 2 args (predicate, data), got ' + irArgs.length);
  }
  const fn = ctx.resolveFn(irArgs[0], ctx.env);
  if (!fn || fn.params.length !== 1) {
    throw new Error('filter: predicate must be a unary function');
  }
  const dataRaw = ctx.evaluateExpr(irArgs[1], ctx.env);
  const data = valueLib.isValue(dataRaw) ? dataRaw.data : dataRaw;
  if (!Array.isArray(data) && !(data && data.BYTES_PER_ELEMENT)) {
    throw new Error('filter: data must be a vector (got '
      + (data === null ? 'null' : typeof data) + ')');
  }
  const elemEnv = Object.assign({}, ctx.env);
  const kept: any[] = [];
  for (let i = 0; i < data.length; i++) {
    elemEnv[fn.params[0]] = data[i];
    const keep = ctx.evaluateExpr(fn.body, elemEnv);
    if (keep) kept.push(data[i]);
  }
  const out = new Float64Array(kept.length);
  for (let i = 0; i < kept.length; i++) {
    out[i] = kept[i] === true ? 1 : kept[i] === false ? 0 : +kept[i];
  }
  return { shape: [kept.length], data: out };
}

ops.register({
  name: 'filter',
  kind: 'higher-order',
  logical: _filterLogical,
});

// =====================================================================
// broadcast(f, args…) — Phase 5c remaining
// =====================================================================
//
// broadcast applies f elementwise over arrays. Two surface shapes:
//   broadcast(f, A, B, …)                — positional
//   broadcast(f, x = A, y = B, …)        — kwargs naming f's params
// Each array must have the same length (no auto-broadcast at this
// layer; spec §04's leading-axis singleton-expansion is the
// aggregate / valueOps job). Stochastic-broadcast (kernel f) is
// NOT handled here — the materialiser owns that path. The OpDecl
// logical mirrors the engine's existing `_broadcastApply` exactly.
//
// Migrating broadcast to higher-order required passing the full
// `ir` (not just `ir.args`) so kwargs are visible — see ops.ts
// dispatchHigherOrder signature.

function _broadcastLogical(ir: any, ctx: any): any {
  const args   = ir.args   || [];
  const kwargs = ir.kwargs || {};
  if (args.length < 1) throw new Error('broadcast: no function argument');
  const fn = ctx.resolveFn(args[0], ctx.env);
  if (!fn) throw new Error('broadcast: first arg must be a function');
  const kwargKeys = Object.keys(kwargs);
  const sources = new Array(fn.params.length);
  if (kwargKeys.length > 0) {
    // kwargs form: match by surface kwarg name first
    // (paramKwargs), fall back to internal placeholder name (params).
    for (let i = 0; i < fn.params.length; i++) {
      const surface = (fn.paramKwargs && fn.paramKwargs[i]) || fn.params[i];
      if (kwargs[surface] != null) sources[i] = kwargs[surface];
      else if (kwargs[fn.params[i]] != null) sources[i] = kwargs[fn.params[i]];
      else throw new Error('broadcast: no argument for parameter '
        + (surface || fn.params[i]));
    }
  } else {
    const posArgs = args.slice(1);
    if (posArgs.length !== fn.params.length) {
      throw new Error('broadcast: expected ' + fn.params.length
        + ' positional arrays, got ' + posArgs.length);
    }
    for (let i = 0; i < fn.params.length; i++) sources[i] = posArgs[i];
  }
  const inputs: any = sources.map((s: any) => ctx.evaluateExpr(s, ctx.env));
  // Delegate the per-element iteration to the engine's existing
  // `_broadcastApply` — Phase 5c keeps the impl shared between the
  // dedicated dispatch and the OpDecl path. Lazy require to avoid
  // a module-load cycle with sampler.ts (ops-declarations is
  // required during sampler.ts module load).
  const samplerMod = require('./sampler.ts');
  return samplerMod._internal._broadcastApply(fn, inputs, ctx.env);
}

ops.register({
  name: 'broadcast',
  kind: 'higher-order',
  logical: _broadcastLogical,
});

// =====================================================================
// aggregate(f_reduction, output_axes, expr) — Phase 5c remaining
// =====================================================================
//
// aggregate dispatches via the AGGREGATE_PATTERNS specialiser table
// (matmul, matvec, outer, etc.) with a broadcast-reduce default for
// everything the specialisers don't catch (engine-concepts §16).
// The whole machinery lives in `sampler-aggregate.ts`; the OpDecl
// `logical` just delegates to `_evalAggregate(ir, env)` — the same
// entry the dedicated dispatch uses.
//
// Aggregate is the most distinctive higher-order op (axis-name IR,
// pattern table, multi-mode dispatch). Subsuming the specialiser
// machinery directly into OpDecl's `batched` slot would be a larger
// refactor; the delegation approach single-sources the impl via the
// existing sampler-aggregate.ts file while bringing aggregate into
// the OpDecl framework for consistency.

function _aggregateLogical(ir: any, ctx: any): any {
  const aggregateMod = require('./sampler-aggregate.ts');
  return aggregateMod._evalAggregate(ir, ctx.env);
}

ops.register({
  name: 'aggregate',
  kind: 'higher-order',
  logical: _aggregateLogical,
});

module.exports = {
  // Re-export for tests that want to call the logical impls directly
  // (rather than through `ops.dispatch`).
  _crossLogical,
  _selfOuterLogical,
  _traceLogical,
  _diagmatLogical,
  _detLogical,
  _logabsdetLogical,
  _invLogical,
  _lowerCholeskyLogical,
  _rowGramLogical,
  _colGramLogical,
  _transposeLogical,
  _adjointLogical,
  _linsolveLogical,
  _vectorLogical,
  _catLogical,
  _reduceLogical,
  _scanLogical,
  _filterLogical,
  _broadcastLogical,
  _aggregateLogical,
};
