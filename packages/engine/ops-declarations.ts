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

ops.register({
  name: 'cross',
  signature: {
    args: [_array(1, [3], _REAL), _array(1, [3], _REAL)],
    kwargs: {},
    result: _array(1, [3], _REAL),
  },
  argRanks: [1, 1],
  logical: _crossLogical,
  // batched: optional fast-path; Phase 3 will add a tight loop over
  // [N, 3] Float64Array buffers. Until then the dispatcher uses the
  // per-atom fallback — semantically equivalent, just N JS calls.
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

ops.register({
  name: 'inv',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _array(2, ['%dynamic', '%dynamic'], _REAL),
  },
  argRanks: [2],
  logical: _invLogical,
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

ops.register({
  name: 'lower_cholesky',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _array(2, ['%dynamic', '%dynamic'], _REAL),
  },
  argRanks: [2],
  logical: _lowerCholeskyLogical,
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
};
