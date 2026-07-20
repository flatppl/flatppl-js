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
// The atom-batched form (shape=[N, 3]) takes the real-only
// `_crossBatched` tight loop below; complex / non-atom-batched
// shapes fall back to the inline per-atom loop in
// `_crossBatchedOrFallback`. Logical impl mirrors `ARITH_OPS.cross`
// exactly; the conformance suite pins equivalence.

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
    if (valueLib.isAtomBatched(v, N) && v.shape.length === 2 && v.shape[1] === 3) {
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
      if (valueLib.isAtomBatched(v, N) && v.shape.length === 2) {
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

// Batched fast-path: atom-batched vector `[N, k]` → atom-batched
// matrix `[N, k, k]` with each [k, k] slice = v[atom] · v[atom]^†.
// Vectorised tight loop on flat row-major storage; complex falls
// back to per-atom (rare).
function _selfOuterBatchedOrFallback(args: any[], N: number): any | null {
  const v = args[0];
  if (!valueLib.isValue(v)) return null;
  if (v.dtype === 'complex') return null;
  if (v.shape.length !== 2) return null;
  const k = v.shape[1];
  const out = new Float64Array(N * k * k);
  for (let atom = 0; atom < N; atom++) {
    const vBase = atom * k;
    const oBase = atom * k * k;
    for (let i = 0; i < k; i++) {
      const vi = v.data[vBase + i];
      for (let j = 0; j < k; j++) {
        out[oBase + i * k + j] = vi * v.data[vBase + j];
      }
    }
  }
  return { shape: [N, k, k], data: out };
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
  batched: _selfOuterBatchedOrFallback,
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

// Batched fast-path: atom-batched matrix `[N, m, m]` → atom-batched
// scalar `[N]`. Iterates once over the flat row-major buffer reading
// only diagonal positions (i*m + i, atom-strided). Complex falls
// back to per-atom slicing (rare; trace of complex matrices isn't
// part of the hot loop today).
function _traceBatchedOrFallback(args: any[], N: number): any | null {
  const M = args[0];
  if (!valueLib.isValue(M)) return null;
  if (M.dtype === 'complex') return null;
  if (valueLib.isDiagStored(M)) return null;   // logical handles diag
  if (M.shape.length !== 3) return null;
  const m = M.shape[1];
  if (M.shape[2] !== m) return null;
  // Densify if any structural tag (lower / upper / sym / posdef)
  // is set — the diagonal is still in `data[i*m + i]` after
  // densification, but the structured-bit fast paths aren't
  // applicable to a reduction.
  const D = valueLib.densify(M);
  const stride = m * m;
  const out = new Float64Array(N);
  for (let atom = 0; atom < N; atom++) {
    const base = atom * stride;
    let s = 0;
    for (let i = 0; i < m; i++) s += D.data[base + i * m + i];
    out[atom] = s;
  }
  return { shape: [N], data: out };
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
  batched: _traceBatchedOrFallback,
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

// Shared atom-batched slice helper: given an atom-batched Value v
// (shape=[N, ...inner]), return the i-th per-atom Value
// (shape=[...inner]) as a subarray view into the flat buffer.
// Avoids per-atom allocations of the data buffer.
function _atomSlice(v: any, i: number, N: number): any {
  if (!valueLib.isValue(v) || v.shape.length === 0 || v.shape[0] !== N) return v;
  const inner = v.shape.slice(1);
  const innerLen = inner.reduce((a: number, b: number) => a * b, 1) || 1;
  const sub: any = {
    shape: inner,
    data: v.data.subarray(i * innerLen, (i + 1) * innerLen),
  };
  if (v.dtype === 'complex' && v.im) {
    sub.dtype = 'complex';
    sub.im = v.im.subarray(i * innerLen, (i + 1) * innerLen);
  }
  return sub;
}

// Pack N per-atom Float64Array results (each of length k) into one
// atom-batched Value shape=[N, ...inner]. Helper for batched linalg
// outputs.
function _packAtoms(perAtom: any[], N: number, innerShape: number[]): any {
  const innerLen = innerShape.reduce((a, b) => a * b, 1) || 1;
  const out = new Float64Array(N * innerLen);
  for (let i = 0; i < N; i++) {
    const r = perAtom[i];
    const src = valueLib.isValue(r) ? r.data : (typeof r === 'number' ? null : r);
    if (typeof r === 'number') out[i] = r;
    else if (src) out.set(src, i * innerLen);
  }
  return { shape: [N].concat(innerShape), data: out };
}

// diagmat batched: input [N, k] → output [N, k, k]. Per-atom
// constructs a diagonal matrix; uses diag-stored representation
// (struct = ST_DIAG) when the inner logical produces one.
function _diagmatBatched(args: any[], N: number): any {
  const v = args[0];
  if (!valueLib.isValue(v) || v.shape.length !== 2 || v.shape[0] !== N) {
    // Not atom-batched in the canonical sense — defer to per-atom.
    const perAtom: any[] = new Array(N);
    for (let i = 0; i < N; i++) perAtom[i] = _diagmatLogical(_atomSlice(v, i, N));
    return _packAtoms(perAtom, N, perAtom[0].shape);
  }
  const k = v.shape[1];
  // Output: [N, k, k] dense. (Could store diag-tagged, but the
  // variant dispatcher's downstream consumers may densify anyway;
  // dense is the safe baseline.)
  const out = new Float64Array(N * k * k);
  for (let i = 0; i < N; i++) {
    const oBase = i * k * k;
    const dBase = i * k;
    for (let j = 0; j < k; j++) {
      out[oBase + j * k + j] = v.data[dBase + j];
    }
  }
  return { shape: [N, k, k], data: out };
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
  batched: _diagmatBatched,
});

// =====================================================================
// det(A) — determinant of a square matrix (LU with partial pivoting)
// =====================================================================
//
// Phase 2 migration. Logical rank: 2 → 0. Diag-stored fast path:
// product of the diagonal (O(n)).

function _detLogical(A: any): any {
  if (valueLib.isValue(A)) valueLib.requireMatrix(A, 'det');
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

// Batched fast-path: input [N, m, m] → output [N] scalars. Per-atom
// _detLogical with subarray views into the flat buffer; avoids
// allocating a new Value per atom.
function _detBatched(args: any[], N: number): any {
  const A = args[0];
  if (!valueLib.isValue(A) || A.shape.length !== 3 || A.shape[0] !== N) {
    // Per-atom fallback (rare; the dispatcher invokes batched
    // unconditionally for atom-batched inputs).
    const perAtom = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const r = _detLogical(_atomSlice(A, i, N));
      perAtom[i] = valueLib.isValue(r) ? r.data[0] : r;
    }
    return { shape: [N], data: perAtom };
  }
  const out = new Float64Array(N);
  for (let atom = 0; atom < N; atom++) {
    const sub = _atomSlice(A, atom, N);
    const r = _detLogical(sub);
    out[atom] = valueLib.isValue(r) ? r.data[0] : r;
  }
  return { shape: [N], data: out };
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
  batched: _detBatched,
});

// =====================================================================
// logabsdet(A) — log|det(A)| via LU (numerically stable on near-singular)
// =====================================================================

function _logabsdetLogical(A: any): any {
  if (valueLib.isValue(A)) valueLib.requireMatrix(A, 'logabsdet');
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

// Batched fast-path mirrors _detBatched.
function _logabsdetBatched(args: any[], N: number): any {
  const A = args[0];
  if (!valueLib.isValue(A) || A.shape.length !== 3 || A.shape[0] !== N) {
    const perAtom = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const r = _logabsdetLogical(_atomSlice(A, i, N));
      perAtom[i] = valueLib.isValue(r) ? r.data[0] : r;
    }
    return { shape: [N], data: perAtom };
  }
  const out = new Float64Array(N);
  for (let atom = 0; atom < N; atom++) {
    const r = _logabsdetLogical(_atomSlice(A, atom, N));
    out[atom] = valueLib.isValue(r) ? r.data[0] : r;
  }
  return { shape: [N], data: out };
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
  batched: _logabsdetBatched,
});

// =====================================================================
// inv(A) — matrix inverse via LU + back-substitution against I
// =====================================================================
//
// Phase 2 migration. Diag-stored fast path: reciprocal of the
// diagonal (O(n)). Throws on singular matrices.

function _invLogical(A: any): any {
  if (valueLib.isValue(A)) valueLib.requireMatrix(A, 'inv');
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
    if (valueLib.isAtomBatched(A, N) && A.shape.length === 3) {
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
// _ms_check_symmetric(A) — engine-internal metricsum runtime guard
// =====================================================================
//
// Validating passthrough: returns its input unchanged after asserting
// it's a rank-2 square matrix that's approximately symmetric. Emitted
// by lift.inlineMetricsumLift to wrap the metric argument once per
// metricsum call (engine-concepts §23). Spec §sec:metricsum mandates
// symmetric metrics; squareness comes for free via inv()'s own runtime
// check, but symmetry was unchecked under both Form-A and the initial
// Form-B landings — the wrapper closes that gap with a metricsum-
// attributed error message (instead of opaque downstream NaNs).
//
// Tolerance: mixed absolute + relative (NumPy `allclose` convention).
//   |A[i,j] - A[j,i]| ≤ ATOL + RTOL · max(|A[i,j]|, |A[j,i]|)
// Defaults are baked in (not user-tunable) to keep the op's arity at 1.
const _MS_SYM_ATOL = 1e-12;
const _MS_SYM_RTOL = 1e-9;

function _msCheckSymmetricLogical(A: any): any {
  // Densify to a canonical Value so the shape + data are uniform
  // regardless of input format (nested JS array, nested Float64Array,
  // diag-stored Value, dense Value).
  const v = valueLib.densify(valueLib.asValue(A));
  // Spec §sec:metricsum: metric is a rank-2 array of scalars. Refuse a
  // vector-of-vectors per §03 — `requireMatrix` emits the rowstack hint.
  valueLib.requireMatrix(v, 'metricsum: metric');
  if (!Array.isArray(v.shape) || v.shape.length !== 2) {
    throw new Error('metricsum: metric must be a rank-2 matrix, got shape ['
      + (Array.isArray(v.shape) ? v.shape.join(',') : '?') + ']');
  }
  const m = v.shape[0];
  const n = v.shape[1];
  if (m !== n) {
    throw new Error('metricsum: metric must be square, got shape ['
      + m + ',' + n + ']');
  }
  const data = v.data;
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < n; j++) {
      const aij = data[i * n + j];
      const aji = data[j * n + i];
      const diff = Math.abs(aij - aji);
      const tol = _MS_SYM_ATOL + _MS_SYM_RTOL * Math.max(
        Math.abs(aij), Math.abs(aji));
      if (diff > tol) {
        throw new Error('metricsum: metric is not symmetric: A['
          + i + ',' + j + '] = ' + aij + ' but A['
          + j + ',' + i + '] = ' + aji);
      }
    }
  }
  // Return the original input — Value-identity preservation matters
  // for caches that key off binding handles, and we want zero overhead
  // beyond the symmetry walk itself.
  return A;
}

function _msCheckSymmetricBatched(args: any[], N: number): any {
  const A = args[0];
  // Atom-batched [N, m, n] matrix: validate each slice. The dispatcher
  // calls this when atomN is set; if A's shape doesn't fit the [N, m, n]
  // expectation we return null so dispatch falls back to per-atom.
  if (!valueLib.isValue(A) || !Array.isArray(A.shape) || A.shape.length !== 3
      || A.shape[0] !== N) {
    return null;
  }
  const m = A.shape[1];
  const n = A.shape[2];
  if (m !== n) {
    throw new Error('metricsum: metric must be square per atom, got shape ['
      + A.shape.join(',') + ']');
  }
  const stride = m * n;
  for (let atom = 0; atom < N; atom++) {
    const base = atom * stride;
    for (let i = 0; i < m; i++) {
      for (let j = i + 1; j < n; j++) {
        const aij = A.data[base + i * n + j];
        const aji = A.data[base + j * n + i];
        const diff = Math.abs(aij - aji);
        const tol = _MS_SYM_ATOL + _MS_SYM_RTOL * Math.max(
          Math.abs(aij), Math.abs(aji));
        if (diff > tol) {
          throw new Error('metricsum: metric is not symmetric at atom '
            + atom + ': A[' + i + ',' + j + '] = ' + aij
            + ' but A[' + j + ',' + i + '] = ' + aji);
        }
      }
    }
  }
  return A;
}

ops.register({
  name: '_ms_check_symmetric',
  signature: {
    args: [_array(2, ['%dynamic', '%dynamic'], _REAL)],
    kwargs: {},
    result: _array(2, ['%dynamic', '%dynamic'], _REAL),
  },
  argRanks: [2],
  logical: _msCheckSymmetricLogical,
  batched: _msCheckSymmetricBatched,
});

// =====================================================================
// lower_cholesky(A) — lower-triangular L with A = L Lᵀ (PD A)
// =====================================================================
//
// Phase 2 migration. Diag-stored PD A: L = √diag (still diagonal,
// still lower-triangular). O(n). Throws if A is not PD.

function _lowerCholeskyLogical(A: any): any {
  if (valueLib.isValue(A)) valueLib.requireMatrix(A, 'lower_cholesky');
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
    if (valueLib.isAtomBatched(A, N) && A.shape.length === 3) {
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
  if (valueLib.isValue(A)) {
    valueLib.requireMatrix(A, 'row_gram');
    return valueOps.mul(A, valueLib.adjoint(A));
  }
  // Nested-array path uses ARITH_OPS.transpose, which isn't migrated
  // to ops.ts yet. Lazy-require to avoid a module-load cycle.
  const samplerMod = require('./sampler.ts');
  return linalg._matmul(A, samplerMod._internal.ARITH_OPS.transpose(A));
}

// Batched gram fast-paths: per-atom row_gram (A · Aᵀ) / col_gram
// (Aᵀ · A). Each atom produces a square matrix; output is
// [N, m, m] (row_gram) or [N, n, n] (col_gram).
function _rowGramBatched(args: any[], N: number): any {
  const A = args[0];
  if (!valueLib.isValue(A) || A.shape.length !== 3 || A.shape[0] !== N) {
    // Generic per-atom dispatch.
    const perAtomResults: any[] = [];
    for (let i = 0; i < N; i++) perAtomResults.push(_rowGramLogical(_atomSlice(A, i, N)));
    return _packAtoms(perAtomResults, N, perAtomResults[0].shape);
  }
  const m = A.shape[1];
  const perAtomResults: any[] = [];
  for (let atom = 0; atom < N; atom++) {
    perAtomResults.push(_rowGramLogical(_atomSlice(A, atom, N)));
  }
  return _packAtoms(perAtomResults, N, [m, m]);
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
  batched: _rowGramBatched,
});

function _colGramLogical(A: any): any {
  if (valueLib.isValue(A)) {
    valueLib.requireMatrix(A, 'col_gram');
    return valueOps.mul(valueLib.adjoint(A), A);
  }
  const samplerMod = require('./sampler.ts');
  return linalg._matmul(samplerMod._internal.ARITH_OPS.transpose(A), A);
}

function _colGramBatched(args: any[], N: number): any {
  const A = args[0];
  if (!valueLib.isValue(A) || A.shape.length !== 3 || A.shape[0] !== N) {
    const perAtomResults: any[] = [];
    for (let i = 0; i < N; i++) perAtomResults.push(_colGramLogical(_atomSlice(A, i, N)));
    return _packAtoms(perAtomResults, N, perAtomResults[0].shape);
  }
  const n = A.shape[2];
  const perAtomResults: any[] = [];
  for (let atom = 0; atom < N; atom++) {
    perAtomResults.push(_colGramLogical(_atomSlice(A, atom, N)));
  }
  return _packAtoms(perAtomResults, N, [n, n]);
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
  batched: _colGramBatched,
});

// =====================================================================
// Rank-polymorphic ops (Phase 5a — engine-concepts §18)
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
  if (valueLib.isValue(A)) valueLib.requireMatrix(A, 'linsolve');
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

// linsolve atom-aware variant: A=rank-2 square × b=rank-1 vector,
// atom-batched b → per-atom solve with shared A. Output shape=[N, m].
//
// The most common Bayesian use case: whiten a per-atom vector by a
// shared (atom-indep) precision/covariance Cholesky factor. Without
// this variant, the rank-polymorphic dispatch path called
// `_linsolveLogical` with rank-1 expectation and would error on
// rank-2 atom-batched b.
//
// A atom-batched is NOT covered (would need a rank-3 A pattern);
// users for whom that's a hot path can use `broadcast(fn(linsolve
// (_, b)), A_per_atom)`.
function _linsolveBatchedVec(args: any[], N: number): any {
  const A = args[0], b = args[1];
  if (!valueLib.isValue(A) || A.shape.length !== 2) {
    throw new Error('linsolve.batched(rank-2, rank-1): A must be rank-2');
  }
  if (!valueLib.isAtomBatched(b, N) || b.shape.length !== 2) {
    throw new Error('linsolve.batched(rank-2, rank-1): b must be atom-batched rank-1 (shape=[N, m])');
  }
  const m = A.shape[1];
  if (b.shape[1] !== m) {
    throw new Error('linsolve.batched: A is ' + JSON.stringify(A.shape)
      + ', b is ' + JSON.stringify(b.shape) + ' — incompatible inner dim');
  }
  const out = new Float64Array(N * m);
  for (let atom = 0; atom < N; atom++) {
    const subData = b.data.subarray(atom * m, (atom + 1) * m);
    const subB: any = { shape: [m], data: subData };
    const x = _linsolveLogical(A, subB);
    // _linsolveLogical may return a Value or a plain array; normalise.
    const xData = valueLib.isValue(x) ? x.data : x;
    for (let j = 0; j < m; j++) out[atom * m + j] = xData[j];
  }
  return { shape: [N, m], data: out };
}
ops.registerVariant('linsolve', {
  argPatterns: [
    { rank: 2, dtype: 'real' },
    { rank: 1, dtype: 'real' },
  ],
  impl: (vs: any[]) => _linsolveLogical(vs[0], vs[1]),
  batched: _linsolveBatchedVec,
  label: 'linsolve(rank-2, rank-1) atom-aware → per-atom solve, shared A',
});

// =====================================================================
// Variadic ops (Phase 5b — engine-concepts §18)
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
  if (allScalar) {
    const data = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      data[i] = xs[i] === true ? 1 : xs[i] === false ? 0 : +xs[i];
    }
    return { shape: [xs.length], data: data };
  }
  // Nested-vector path (`vector(V1, V2, …)` where each Vi is itself
  // a Value or vector-like). When all elements share the same inner
  // shape, collapse into a flat shape-explicit Value with an
  // explicit `outerRank=1` tag — the §2.1 engine-wide storage
  // convention. The tag distinguishes a length-N nested-vector
  // (outerRank=1, inner shape S) from a flat rank-(1+|S|) tensor
  // (no outerRank tag → every axis is a loop axis) per spec §03's
  // "vectors of vectors are not interpreted as matrices implicitly".
  //
  // Falls back to the legacy JS-array-of-Values form when shapes
  // don't agree, or when an element isn't array-like (records,
  // tuples, kernels, mixed scalar+non-scalar) — downstream
  // consumers (`rowstack`, `colstack`, matrix linalg) still
  // accept both representations via `valueLib.asVectorOfVectors`.
  let innerShape: number[] | null = null;
  // Track each arg's own explicit outerRank tag so the wrapper's
  // outerRank can STACK on top: `[[C]]` (= vector(vector(C))) has
  // outerRank=2 because the inner `vector(C)` is itself a tagged
  // nested vector (outerRank=1). The new wrapping adds one more
  // outer axis ON TOP of the existing nesting.
  //
  // Untagged args (no `outerRank` field — flat vectors, scalars,
  // typed-arrays, JS arrays) are treated as a SINGLE cell. New
  // outerRank = 1, cell = arg.shape. The user's `[C]` lifts C
  // into a length-1 nested vector with C as the sole cell.
  let stackOuterRank: number | null = null;
  for (let i = 0; i < xs.length; i++) {
    const v = xs[i];
    let s: number[] | null = null;
    let nestedTag: number = 0;
    if (v && Array.isArray(v.shape) && v.data instanceof Float64Array) {
      s = v.shape;
      // Only an EXPLICIT tag stacks; untagged Values are cells.
      nestedTag = (typeof v.outerRank === 'number') ? v.outerRank : 0;
    } else if (v && v.BYTES_PER_ELEMENT !== undefined && typeof v.length === 'number') {
      s = [v.length];
    } else if (Array.isArray(v)) {
      s = [v.length];
    } else if (typeof v === 'number' || typeof v === 'boolean') {
      return xs;
    } else {
      return xs;
    }
    // s is provably non-null here: every branch above either assigned
    // it a number[] or returned. tsc can't carry that narrowing past
    // the chain, so assert it (type-only, no runtime line).
    if (innerShape === null) { innerShape = s; stackOuterRank = nestedTag; }
    else {
      if (s!.length !== innerShape.length) return xs;
      for (let a = 0; a < s!.length; a++) if (s![a] !== innerShape[a]) return xs;
      // Ragged stacking across args ⇒ fall back to JS-array form.
      if (nestedTag !== stackOuterRank) return xs;
    }
  }
  if (innerShape === null) return xs;
  const innerLen = innerShape.reduce((a: number, b: number) => a * b, 1);
  const out = new Float64Array(xs.length * innerLen);
  for (let i = 0; i < xs.length; i++) {
    const v = xs[i];
    if (v && Array.isArray(v.shape) && v.data instanceof Float64Array) {
      out.set(v.data, i * innerLen);
    } else if (v && v.BYTES_PER_ELEMENT !== undefined) {
      for (let k = 0; k < innerLen; k++) out[i * innerLen + k] = +(v as any)[k];
    } else if (Array.isArray(v)) {
      for (let k = 0; k < innerLen; k++) out[i * innerLen + k] = +v[k];
    }
  }
  return {
    shape: [xs.length].concat(innerShape as number[]),
    data: out,
    outerRank: 1 + (stackOuterRank || 0),
  };
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
// Higher-order ops (Phase 5c — engine-concepts §18.1)
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

// engine-concepts §20.1 — `broadcasted(<scalar_op>)` engine primitives.
// Each op registers a variant with `wrappingOp: 'broadcast'` on the
// ops.ts shape-pattern registry (engine-concepts §18.2). The variant
// matches when the dispatcher is called with `opts.wrappingOp ===
// 'broadcast'`; argPatterns are empty constraints (the impls accept
// any shape — value-ops' elementwise impls handle same-shape,
// rank-0 × rank-N, AND spec §04 singleton-axis expansion via stride-0
// reads; the shared `_broadcastOutShape` combiner owns the rule).
//
// Set membership: every scalar primitive whose `broadcasted(<op>)`
// engine primitive exists. For ops whose spec semantics ARE
// elementwise at any rank (add/sub/neg per spec §07 "arrays of same
// shape"), the spec primitive doubles as the batched primitive —
// `valueOps.add(A, B)` is already elementwise. For ops whose spec
// form has different semantics on rank ≥ 1 (mul = matmul) or is
// scalar-only at spec (unary scalar maths), the engine primitive is
// a separate `<op>Elem` impl in value-ops.
//
// Lazy require: value-ops depends on value.ts only (no cycle with
// ops.ts), but the registration runs at module load — keeping the
// table as a one-shot cache avoids re-resolving value-ops on every
// dispatch. The vo binding initialises on first call to
// _ensureBroadcastedRegistered() (the eager call at the bottom of
// this module, alongside the mul-direct and atom-batched ensures).
// Atom-indep `logical` impls for the scalar primitives (engine-concepts
// §18). The logical-rank-0 (scalar) math; the batched path is the op's
// broadcast variant (value-ops *Elem, registered above). `kind:
// 'rank-polymorphic'` hands the scalar arg to `logical` as-is — exactly
// the legacy `ARITH_OPS` entry's behavior, so the migration is
// behavior-preserving (conformance-pinned). Migrating a family =
// extending the tables here.
//
// The impls below mirror `sampler.ARITH_OPS` VERBATIM (the conformance
// suite pins exact equivalence). Migrated: every scalar primitive EXCEPT
// the Value-aware arithmetic `add` / `sub` / `neg` / `mul`. Those four
// carry direct / atom-batched variants whose dispatch interacts with a
// logical fallback in load-bearing ways — e.g. attaching `add`'s logical
// changes a rank-mismatched atom-batched add from "no variant matched"
// to the logical's "rank mismatch", and `mul`'s matmul/matvec variants
// match scalar inputs and throw. They need a deliberate
// variant-fallthrough pass (decide the refusal contract), kept as one
// unit (TODO §18).
function _registerScalarLogicals(): void {
  const cx = require('./sampler-complex.ts');
  const _isC = cx._isComplex;
  const stdlibGamma   = require('@stdlib/math-base-special-gamma');
  const stdlibGammaln = require('@stdlib/math-base-special-gammaln');
  const stdlibErfc    = require('@stdlib/math-base-special-erfc');
  const stdlibErfcinv = require('@stdlib/math-base-special-erfcinv');

  // Family 1: pure-real unary elementary math (`Math.X`).
  const REAL_UNARY: Record<string, (a: number) => number> = {
    log10: Math.log10, log1p: Math.log1p, expm1: Math.expm1,
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    asin: Math.asin, acos: Math.acos, atan: Math.atan,
    sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
    asinh: Math.asinh, acosh: Math.acosh, atanh: Math.atanh,
    floor: Math.floor, ceil: Math.ceil, round: Math.round,
  };

  // The full logical table (rank-polymorphic — args handed to the impl
  // as-is, exactly as the legacy ARITH_OPS entry receives them).
  const LOGICALS: Record<string, (...a: any[]) => any> = {
    ...REAL_UNARY,
    // Family 2: complex-aware unary + accessors (sampler-complex leaf).
    abs:  (a: any) => _isC(a) ? cx._cAbs(a)  : Math.abs(a),
    abs2: (a: any) => _isC(a) ? cx._cAbs2(a) : a * a,
    exp:  (a: any) => _isC(a) ? cx._cExp(a)  : Math.exp(a),
    log:  (a: any) => _isC(a) ? cx._cLog(a)  : Math.log(a),
    sqrt: (a: any) => _isC(a) ? cx._cSqrt(a) : Math.sqrt(a),
    real: (z: any) => _isC(z) ? z.re : +z,
    imag: (z: any) => _isC(z) ? z.im : 0,
    conj: (z: any) => _isC(z) ? cx._cConj(z) : +z,
    cis:  (theta: any) => ({ re: Math.cos(+theta), im: Math.sin(+theta) }),
    complex: (re: any, im: any) => {
      if (im === undefined) {
        if (_isC(re)) return re;
        if (typeof re === 'number') return { re: re, im: 0 };
        throw new Error('complex: single-arg restrictor requires real or complex input');
      }
      return { re: +re, im: +im };
    },
    pos: (a: any) => _isC(a) ? a : +a,
    // Family 3: complex-aware binary (only broadcast variants — safe).
    divide: (a: any, b: any) => (_isC(a) || _isC(b))
      ? cx._cDiv(cx._toComplex(a), cx._toComplex(b)) : a / b,
    pow: (a: any, b: any) => (_isC(a) || _isC(b))
      ? cx._cPow(cx._toComplex(a), cx._toComplex(b)) : Math.pow(a, b),
    // Family 4: pure-real binary / pairwise.
    div: (a: any, b: any) => Math.floor(a / b),       // spec §07 ⌊a/b⌋
    mod: (a: any, b: any) => a - b * Math.floor(a / b), // spec §07 a − b·⌊a/b⌋ (floor-mod)
    min: (a: any, b: any) => Math.min(a, b),
    max: (a: any, b: any) => Math.max(a, b),
    atan2: (y: any, x: any) => Math.atan2(y, x),
    // Family 5: gamma family + link functions (stdlib already bundled
    // via value-ops's *Elem broadcast impls).
    gamma:     (a: any) => stdlibGamma(a),
    loggamma:  (a: any) => stdlibGammaln(a),
    logit:     (p: any) => Math.log(p / (1 - p)),
    invlogit:  (x: any) => 1 / (1 + Math.exp(-x)),
    probit:    (p: any) => -Math.SQRT2 * stdlibErfcinv(2 * p),
    invprobit: (x: any) => 0.5 * stdlibErfc(-x / Math.SQRT2),
    // Family 6: comparisons / predicates / logic / conditional / casts.
    lt: (a: any, b: any) => a < b,
    le: (a: any, b: any) => a <= b,
    gt: (a: any, b: any) => a > b,
    ge: (a: any, b: any) => a >= b,
    equal:   (a: any, b: any) => a === b,
    unequal: (a: any, b: any) => a !== b,
    isfinite: (a: any) => Number.isFinite(a),
    isinf:    (a: any) => !Number.isNaN(a) && !Number.isFinite(a),
    isnan:    (a: any) => Number.isNaN(a),
    iszero:   (a: any) => a === 0,
    land: (a: any, b: any) => a && b,
    lor:  (a: any, b: any) => a || b,
    lxor: (a: any, b: any) => a !== b,
    lnot: (a: any) => !a,
    ifelse: (c: any, a: any, b: any) => c ? a : b,
    boolean: (x: any) => {
      if (x === true || x === false) return x;
      if (x === 0) return false;
      if (x === 1) return true;
      throw new Error('boolean: value ' + x + ' is not a boolean');
    },
    integer: (x: any) => {
      if (Number.isInteger(x)) return x;
      throw new Error('integer: value ' + x + ' is not an integer');
    },
  };

  for (const name in LOGICALS) {
    ops.attachLogical(name, LOGICALS[name], 'rank-polymorphic');
  }
}

let _BCAST_VARIANTS_REGISTERED = false;
function _ensureBroadcastedRegistered(): void {
  if (_BCAST_VARIANTS_REGISTERED) return;
  _BCAST_VARIANTS_REGISTERED = true;
  const vo = require('./value-ops.ts');
  // (opName, arity, implFn) — implFn takes the same Values the
  // legacy `impl` did. Arity is recorded in argPatterns.length so
  // the variant matcher rejects ill-shaped calls.
  const BCAST_TABLE: Array<[string, number, (vs: any[]) => any]> = [
    // Binary additive (spec elementwise; value-ops impl IS batched).
    ['add',    2, (vs) => vo.add(vs[0], vs[1])],
    ['sub',    2, (vs) => vo.sub(vs[0], vs[1])],
    // Binary multiplicative (spec has matrix semantics on rank ≥ 1;
    // engine primitives are separate elementwise impls).
    ['mul',    2, (vs) => vo.mulElem(vs[0], vs[1])],
    // `div` is integer floor-division (spec §07 ⌊a/b⌋) → floorDivElem;
    // `divide` is real division → divElem. Distinct ops, kept separate.
    ['div',    2, (vs) => vo.floorDivElem(vs[0], vs[1])],
    ['divide', 2, (vs) => vo.divElem(vs[0], vs[1])],
    ['pow',    2, (vs) => vo.powElem(vs[0], vs[1])],
    ['mod',    2, (vs) => vo.modElem(vs[0], vs[1])],
    // Unary negation (spec elementwise; value-ops impl IS batched).
    ['neg',    1, (vs) => vo.neg(vs[0])],
    // Unary scalar maths (spec is scalar-only; engine primitive is
    // pointwise application of the JS scalar fn over flat data).
    ['exp',    1, (vs) => vo.expElem(vs[0])],
    ['log',    1, (vs) => vo.logElem(vs[0])],
    ['sqrt',   1, (vs) => vo.sqrtElem(vs[0])],
    ['sin',    1, (vs) => vo.sinElem(vs[0])],
    ['cos',    1, (vs) => vo.cosElem(vs[0])],
    ['tan',    1, (vs) => vo.tanElem(vs[0])],
    ['abs',    1, (vs) => vo.absElem(vs[0])],
    ['abs2',   1, (vs) => vo.abs2Elem(vs[0])],
    ['log10',  1, (vs) => vo.log10Elem(vs[0])],
    ['log1p',  1, (vs) => vo.log1pElem(vs[0])],
    ['expm1',  1, (vs) => vo.expm1Elem(vs[0])],
    ['floor',  1, (vs) => vo.floorElem(vs[0])],
    ['ceil',   1, (vs) => vo.ceilElem(vs[0])],
    ['round',  1, (vs) => vo.roundElem(vs[0])],
    // P9 additions — finish the §18.2 keystone migration so every
    // ARITH_OPS_N scalar primitive flows through the variant registry.
    ['min',    2, (vs) => vo.minElem(vs[0], vs[1])],
    ['max',    2, (vs) => vo.maxElem(vs[0], vs[1])],
    ['lt',     2, (vs) => vo.ltElem(vs[0], vs[1])],
    ['le',     2, (vs) => vo.leElem(vs[0], vs[1])],
    ['gt',     2, (vs) => vo.gtElem(vs[0], vs[1])],
    ['ge',     2, (vs) => vo.geElem(vs[0], vs[1])],
    ['equal',  2, (vs) => vo.equalElem(vs[0], vs[1])],
    ['unequal', 2, (vs) => vo.unequalElem(vs[0], vs[1])],
    ['isfinite', 1, (vs) => vo.isfiniteElem(vs[0])],
    ['isinf',  1, (vs) => vo.isinfElem(vs[0])],
    ['isnan',  1, (vs) => vo.isnanElem(vs[0])],
    ['iszero', 1, (vs) => vo.iszeroElem(vs[0])],
    ['land',   2, (vs) => vo.landElem(vs[0], vs[1])],
    ['lor',    2, (vs) => vo.lorElem(vs[0], vs[1])],
    ['lxor',   2, (vs) => vo.lxorElem(vs[0], vs[1])],
    ['lnot',   1, (vs) => vo.lnotElem(vs[0])],
    ['atan2',  2, (vs) => vo.atan2Elem(vs[0], vs[1])],
    ['asin',   1, (vs) => vo.asinElem(vs[0])],
    ['acos',   1, (vs) => vo.acosElem(vs[0])],
    ['atan',   1, (vs) => vo.atanElem(vs[0])],
    ['sinh',   1, (vs) => vo.sinhElem(vs[0])],
    ['cosh',   1, (vs) => vo.coshElem(vs[0])],
    ['tanh',   1, (vs) => vo.tanhElem(vs[0])],
    ['asinh',  1, (vs) => vo.asinhElem(vs[0])],
    ['acosh',  1, (vs) => vo.acoshElem(vs[0])],
    ['atanh',  1, (vs) => vo.atanhElem(vs[0])],
    // Closure of ARITH_OPS_N coverage — pos / link fns / casts /
    // ifelse. With these, every entry in _SCALAR_PRIM_ARITY flows
    // through the variant registry; ARITH_OPS_N becomes a thin
    // legacy facade that future work can retire.
    ['pos',     1, (vs) => vo.posElem(vs[0])],
    ['boolean', 1, (vs) => vo.booleanElem(vs[0])],
    ['integer', 1, (vs) => vo.integerElem(vs[0])],
    ['logit',     1, (vs) => vo.logitElem(vs[0])],
    ['invlogit',  1, (vs) => vo.invlogitElem(vs[0])],
    ['probit',    1, (vs) => vo.probitElem(vs[0])],
    ['invprobit', 1, (vs) => vo.invprobitElem(vs[0])],
    ['gamma',     1, (vs) => vo.gammaElem(vs[0])],
    ['loggamma',  1, (vs) => vo.loggammaElem(vs[0])],
    ['ifelse',  3, (vs) => vo.ifelseElem(vs[0], vs[1], vs[2])],
    // Phase 3.2: complex constructor / accessors elementwise.
    // `complex.(re, im)` / `real.(z)` / `imag.(z)` / `conj.(z)` /
    // `cis.(theta)` lower to `broadcast(<op>, args)` and route here.
    ['complex', 2, (vs) => vo.complexElem(vs[0], vs[1])],
    ['real',    1, (vs) => vo.realElem(vs[0])],
    ['imag',    1, (vs) => vo.imagElem(vs[0])],
    ['conj',    1, (vs) => vo.conjElem(vs[0])],
    ['cis',     1, (vs) => vo.cisElem(vs[0])],
  ];
  for (const [opName, arity, impl] of BCAST_TABLE) {
    // Arity is anchored to the ONE scalar-primitive source of truth
    // (`ops.SCALAR_PRIM_ARITY`, engine-concepts §18) — drift between this
    // table and the batched/compile consumers fails loudly at load.
    const canonical = (ops.SCALAR_PRIM_ARITY as any)[opName];
    if (canonical !== arity) {
      throw new Error(`BCAST_TABLE: '${opName}' arity ${arity} disagrees with `
        + `ops.SCALAR_PRIM_ARITY (${canonical}) — update the one source of truth`);
    }
    const argPatterns = new Array(arity).fill(null).map(() => ({}));
    ops.registerVariant(opName, {
      argPatterns,
      wrappingOp: 'broadcast',
      impl,
      label: 'broadcasted(' + opName + ')',
    });
  }
}

// =====================================================================
// Atom-batched fast-path variants (P1 follow-up; engine-concepts §18.2)
// =====================================================================
//
// The §2.1 leading-axis-batch convention: a Value of shape=[N, …rest]
// where N is the atom count carries an atom-batched rank-(rank+1)
// payload. The asymmetric case "atom-indep rank-r + atom-batched
// rank-r" (e.g. MvNormal's `mu + L·z` where mu is shape=[k] and L·z
// is shape=[N, k]) needs a per-cell elementwise broadcast loop that
// the symmetric value-ops elementwise impl doesn't handle (rank-
// mismatch throw).
//
// This block lifts the existing `value-ops.addN` / `subN` / `negN` /
// `mulN` fast-paths into variant `batched` slots on the corresponding
// op. The variant matcher recognises atom-batched args via the
// `_pickVariantAtomAware` path (shape rank == pattern.rank + 1,
// shared N across all atom-batched args); when it picks a matching
// variant with a `batched` impl, the dispatcher routes to it directly
// without per-atom slicing.
//
// Migration scope (this commit):
//   - add / sub: rank-1 + rank-1 with one or both atom-batched.
//     Symmetric same-shape case stays on the existing broadcasted
//     variant (rank-N elementwise add is just vo.add). The new
//     atom-batched variant handles the ASYMMETRIC case via the
//     existing _atomBroadcastBinop helper.
//   - neg: rank-1 unary. The atom-indep neg already iterates over
//     flat data so the batched impl is a trivial passthrough; the
//     variant registration makes routing uniform.
//   - mul: matrix × rank-1 vec (rank-2 + rank-1). When the second
//     arg is atom-batched (shape=[N, n] meaning [N] vectors of
//     length n), routes to _matBatchedVecMul for the MvNormal `L·z`
//     hot path (complex matvec stays in value-ops.mulN —
//     _cxMatBatchedVecMul is internal there). Diag-stored matrix arg
//     stays as a value-ops pre-check (the diag fast-path's null-
//     fallthrough doesn't fit variant matching).
//
// Mostly real-only; the one complex variant below (rank-2 × rank-2
// batched matmul, both-operands-complex) is the precedent the broader
// complex migration extends. Everything else complex stays on the
// value-ops pre-check path in `value-ops.mulN`; the full migration is
// a separate item (parallel complex variants + an either-operand
// dtype pattern form — see the TODO's complex-mul migration entry).

let _ATOM_BATCHED_VARIANTS_REGISTERED = false;
function _ensureAtomBatchedRegistered(): void {
  if (_ATOM_BATCHED_VARIANTS_REGISTERED) return;
  _ATOM_BATCHED_VARIANTS_REGISTERED = true;
  const vo = require('./value-ops.ts');
  const valueLib = require('./value.ts');

  // -------------------------------------------------------------------
  // add / sub: rank-1 + rank-1, atom-batched broadcast
  // -------------------------------------------------------------------
  //
  // The exact-rank match (both rank-1) calls vo.add / vo.sub directly.
  // Atom-aware match (one or both rank-2) routes to `batched`, which
  // detects which arg is atom-batched and dispatches:
  //   - both atom-batched same shape → vo.add (elementwise on flat data)
  //   - one atom-batched, one atom-indep → _atomBroadcastBinop
  //
  // Mirror the value-ops _makeAtomAwareBinop logic.
  function _makeBatchedAddLike(scalarFn: any, atomIndepImpl: any, opName: string) {
    return function batchedAddLike(args: any[], N: number) {
      const a = args[0], b = args[1];
      // P3 follow-up: use the canonical isAtomBatched predicate.
      // This variant is registered for `argPatterns: [{rank: 1}, {rank: 1}]`
      // (atom-aware → rank-2 with leading dim N), so the rank check
      // is implicit; `isAtomBatched` formalises the §2.1 contract.
      const aBatched = valueLib.isAtomBatched(a, N) && a.shape.length === 2;
      const bBatched = valueLib.isAtomBatched(b, N) && b.shape.length === 2;
      if (aBatched && bBatched) {
        // Same atom-batched shape: elementwise on flat data works.
        return atomIndepImpl(a, b);
      }
      if (aBatched && !bBatched) {
        return vo._atomBroadcastBinop(scalarFn, a, b, N, false, opName);
      }
      // !aBatched && bBatched
      return vo._atomBroadcastBinop(scalarFn, b, a, N, true, opName);
    };
  }
  ops.registerVariant('add', {
    argPatterns: [{ rank: 1 }, { rank: 1 }],
    impl: (vs: any[]) => vo.add(vs[0], vs[1]),
    batched: _makeBatchedAddLike((x: any, y: any) => x + y, vo.add, 'add'),
    label: 'add(rank-1, rank-1) atom-aware',
  });
  ops.registerVariant('sub', {
    argPatterns: [{ rank: 1 }, { rank: 1 }],
    impl: (vs: any[]) => vo.sub(vs[0], vs[1]),
    batched: _makeBatchedAddLike((x: any, y: any) => x - y, vo.sub, 'sub'),
    label: 'sub(rank-1, rank-1) atom-aware',
  });

  // -------------------------------------------------------------------
  // neg: rank-1 unary, atom-batched
  // -------------------------------------------------------------------
  //
  // The atom-indep `neg` iterates over flat data regardless of rank;
  // batched value-ops.neg is just neg unchanged. The variant exists
  // primarily so atom-aware dispatch routes uniformly through the
  // registry rather than via the value-ops.negN side door.
  ops.registerVariant('neg', {
    argPatterns: [{ rank: 1 }],
    impl: (vs: any[]) => vo.neg(vs[0]),
    batched: (args: any[], _N: number) => vo.neg(args[0]),
    label: 'neg(rank-1) atom-aware',
  });

  // -------------------------------------------------------------------
  // mul: matrix × rank-1 vec, atom-batched on the vector
  // -------------------------------------------------------------------
  //
  // The MvNormal `L · z` hot path: L is shape=[n, n] (atom-indep
  // matrix), z is shape=[N, n] (atom-batched rank-1 vector). The
  // existing `_matBatchedVecMul` runs the per-atom gemv loop in one
  // pass. The argPatterns declare NO struct constraint, so a
  // diag-stored L matches too — the tripwire throw below catches a
  // caller that should have gone through value-ops.mulN's diag
  // pre-check (the diag fast-path stays there until a separate
  // complex+diag migration). Real-only via dtype: 'real'.
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 2, dtype: 'real' },
      { rank: 1, dtype: 'real' },
    ],
    impl: (vs: any[]) => vo._matVecMul(vs[0], vs[1]),
    batched: (args: any[], N: number) => {
      const A = args[0], v = args[1];
      // This impl handles exactly the atom-indep matrix × atom-
      // batched [N, n] vector pattern. NOTE the atom-aware matcher
      // would ALSO accept an atom-batched A=[N, m, n] (each arg
      // independently matches exact-rank OR rank+1 with leading N) —
      // `_matBatchedVecMul` only reads A as [m, n], so guard
      // explicitly rather than misread [N, m] as the matrix dims.
      if (A.shape.length !== 2) {
        throw new Error(
          'ops.mul.batched(rank-2 × atom-batched rank-1): atom-batched '
          + 'matrix operand (shape=[N, m, n]) is not implemented — this '
          + 'variant handles an atom-indep matrix × per-atom vector only');
      }
      if (valueLib.isDiagStored && valueLib.isDiagStored(A)) {
        // Diag-stored matrices belong on value-ops.mulN's diag
        // pre-check path (O(n) scale per atom). Reaching this impl
        // with one means a caller bypassed that pre-check — throw
        // loudly so the missed caller surfaces instead of paying the
        // densified gemv silently.
        throw new Error(
          'ops.mul.batched(rank-2 × atom-batched rank-1): diag-stored matrix ' +
          'should route through value-ops.mulN (diag fast-path), not the variant registry');
      }
      return vo._matBatchedVecMul(A, v, N);
    },
    label: 'mul(rank-2, rank-1) atom-aware → matBatchedVecMul',
  });

  // -----------------------------------------------------------------
  // P6 follow-up: rank-2 × rank-2 atom-batched matmul
  // -----------------------------------------------------------------
  //
  // Three input patterns produce the same shape=[N, m, p] output:
  //   - A=[N, m, n] × B=[n, p]    → per-atom matmul, shared B
  //   - A=[m, n]    × B=[N, n, p] → per-atom matmul, shared A
  //   - A=[N, m, n] × B=[N, n, p] → per-atom matmul, both per-atom
  // The variant matcher dispatches via atom-aware rank classification:
  // `argRanks: [2, 2]` means each operand is "logical rank 2 OR
  // rank-3 with leading dim N". `_matBatchedMatMul` inspects whether
  // each operand has the [N, ...] shape and dispatches accordingly.
  //
  // Routes Bayesian per-atom linear-algebra hot paths (X · per_atom_B
  // for hierarchical models; per_atom_A · per_atom_B for time-series
  // state-space models) into one batched-gemm call.
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 2, dtype: 'real' },
      { rank: 2, dtype: 'real' },
    ],
    impl: (vs: any[]) => vo._matMatMul(vs[0], vs[1]),
    batched: (args: any[], N: number) => {
      const A = args[0], B = args[1];
      // Refuse diag-stored — keep on the densify-and-retry pre-check.
      if ((valueLib.isDiagStored && valueLib.isDiagStored(A))
          || (valueLib.isDiagStored && valueLib.isDiagStored(B))) {
        throw new Error(
          'ops.mul.batched(rank-2, rank-2): diag-stored operand should '
          + 'route through value-ops.mulN densify-and-retry');
      }
      return vo._matBatchedMatMul(A, B, N);
    },
    label: 'mul(rank-2, rank-2) atom-aware → matBatchedMatMul',
  });

  // Complex counterpart — per-atom complex matmul over [N, m, n] ×
  // [N, n, p]. Routes through _cxMatBatchedMatMul (planar re/im).
  // The per-arg dtype matcher is strict: this variant fires only when
  // BOTH operands are complex Values. A mixed real × complex pair
  // matches NEITHER this variant nor the real one — dispatch throws
  // "no variant matched" and the batched-aggregate harness catches it
  // as fall-through to the generic lowering (correct, just not
  // vectorised). An either-operand dtype pattern form would close
  // that; it's the concrete motivation noted in the TODO's
  // complex-mul migration item.
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 2, dtype: 'complex' },
      { rank: 2, dtype: 'complex' },
    ],
    impl: (vs: any[]) => vo.mul(vs[0], vs[1]),     // delegate to dense mul
    batched: (args: any[], N: number) => {
      const A = args[0], B = args[1];
      if ((valueLib.isDiagStored && valueLib.isDiagStored(A))
          || (valueLib.isDiagStored && valueLib.isDiagStored(B))) {
        throw new Error(
          'ops.mul.batched(complex rank-2, complex rank-2): diag-stored '
          + 'operand should route through value-ops.mulN densify-and-retry');
      }
      return vo._cxMatBatchedMatMul(A, B, N);
    },
    label: 'mul(complex rank-2, complex rank-2) atom-aware → cxMatBatchedMatMul',
  });
}

// =====================================================================
// `mul` direct-wrapping variants (engine-concepts §18.2)
// =====================================================================
//
// Spec §07 mul has shape-dependent semantics: scalar broadcast,
// inner/outer products via Klein-4 tag, matvec, vecmat, matmul.
// The existing `valueOps.mul` switch handles these via a sequence of
// `if (sa.length === ...)` clauses; this block lifts that switch
// into the variant registry as data. Each variant covers one
// shape/tag combination and points at the corresponding helper.
//
// **Pre-dispatch filtering** stays in `valueOps.mul`:
//   - Diag-stored Values are fast-pathed via `_diagMul` BEFORE the
//     registry; on no-fast-path the diag is densified and falls
//     through. The diag fast-path's null-fallthrough semantics
//     doesn't fit cleanly into "match → impl → done" variant
//     dispatch, so it stays as a pre-hook for now.
//   - Complex Values are routed to the existing `_cx*Mul` helpers
//     via the `_isCx` branch in `valueOps.mul`. Migrating complex
//     to variants would require either a cross-arg "any complex"
//     pattern dimension or parallel complex variants for every
//     shape combo; both are larger refactors for a future session.
//
// Variants therefore declare `dtype: 'real'` to make this contract
// explicit: complex Values must be filtered out by the caller
// before the registry is consulted. The matcher rejects complex
// dtype, and the dispatcher would throw "no variant matched" —
// catching any code path that accidentally bypasses valueOps.mul's
// pre-check.
//
// **Klein-4 dispatch** for vec×vec is encoded as four variants:
//   - swapped × unswapped → inner product
//   - unswapped × swapped → outer product
//   - unswapped × unswapped → static error (col × col is undefined)
//   - swapped   × swapped   → static error (row × row is undefined)
// `tag: ['T', 'A']` matches "swapped" Klein-4 elements; `tag:
// ['N', 'C']` matches "unswapped". This factoring mirrors
// `isTransposeView()`'s swapped-bit semantics directly.

let _MUL_DIRECT_VARIANTS_REGISTERED = false;
function _ensureMulDirectRegistered(): void {
  if (_MUL_DIRECT_VARIANTS_REGISTERED) return;
  _MUL_DIRECT_VARIANTS_REGISTERED = true;
  const vo = require('./value-ops.ts');

  // (1) Scalar × anything (left scalar): broadcast the scalar over
  //     every cell of the array operand, preserving its tag.
  ops.registerVariant('mul', {
    argPatterns: [{ rank: 0, dtype: 'real' }, { dtype: 'real' }],
    wrappingOp: 'direct',
    impl: (args: any[]) => vo._scalarBroadcastMul(args[0].data[0], args[1]),
    label: 'mul(scalar, *) → scalar broadcast',
  });
  // (2) Anything × scalar (right scalar): symmetric to (1).
  ops.registerVariant('mul', {
    argPatterns: [{ dtype: 'real' }, { rank: 0, dtype: 'real' }],
    wrappingOp: 'direct',
    impl: (args: any[]) => vo._scalarBroadcastMul(args[1].data[0], args[0]),
    label: 'mul(*, scalar) → scalar broadcast',
  });
  // (3a) swapped × unswapped vec×vec → inner product (row × col).
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 1, tag: ['T', 'A'], dtype: 'real' },
      { rank: 1, tag: ['N', 'C'], dtype: 'real' },
    ],
    wrappingOp: 'direct',
    impl: (args: any[]) => vo._innerProduct(args[0], args[1]),
    label: 'mul(row vec, col vec) → inner product',
  });
  // (3b) unswapped × swapped vec×vec → outer product (col × row).
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 1, tag: ['N', 'C'], dtype: 'real' },
      { rank: 1, tag: ['T', 'A'], dtype: 'real' },
    ],
    wrappingOp: 'direct',
    impl: (args: any[]) => vo._outerProduct(args[0], args[1]),
    label: 'mul(col vec, row vec) → outer product',
  });
  // (3c) unswapped × unswapped vec×vec → static error (col × col).
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 1, tag: ['N', 'C'], dtype: 'real' },
      { rank: 1, tag: ['N', 'C'], dtype: 'real' },
    ],
    wrappingOp: 'direct',
    impl: () => {
      throw new Error(
        'mul: vector * vector is not defined; use transpose(v1) * v2 ' +
        'for inner product or v1 * transpose(v2) for outer product');
    },
    label: 'mul(col vec, col vec) → error',
  });
  // (3d) swapped × swapped vec×vec → static error (row × row).
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 1, tag: ['T', 'A'], dtype: 'real' },
      { rank: 1, tag: ['T', 'A'], dtype: 'real' },
    ],
    wrappingOp: 'direct',
    impl: () => {
      throw new Error(
        'mul: transpose(v1) * transpose(v2) is not defined (two row vectors)');
    },
    label: 'mul(row vec, row vec) → error',
  });
  // (4) Matrix × column vector → matvec product. The vector MUST be
  //     unswapped (column orientation); a transposed/row vector on
  //     the right is undefined per spec §07.
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 2, dtype: 'real' },
      { rank: 1, tag: ['N', 'C'], dtype: 'real' },
    ],
    wrappingOp: 'direct',
    impl: (args: any[]) => vo._matVecMul(args[0], args[1]),
    label: 'mul(mat, col vec) → matvec',
  });
  // (4-err) Matrix × row vector → static error. Wins over the
  //     general (rank-2, rank-1) match by tag specificity.
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 2, dtype: 'real' },
      { rank: 1, tag: ['T', 'A'], dtype: 'real' },
    ],
    wrappingOp: 'direct',
    impl: () => {
      throw new Error(
        'mul: matrix * (transposed/row vector) is not defined; ' +
        'mul requires a column vector on the right');
    },
    label: 'mul(mat, row vec) → error',
  });
  // (5) Row vector × matrix → vecmat product. The vector MUST be
  //     swapped (row orientation).
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 1, tag: ['T', 'A'], dtype: 'real' },
      { rank: 2, dtype: 'real' },
    ],
    wrappingOp: 'direct',
    impl: (args: any[]) => vo._vecMatMul(args[0], args[1]),
    label: 'mul(row vec, mat) → vecmat',
  });
  // (5-err) Column vector × matrix → static error.
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 1, tag: ['N', 'C'], dtype: 'real' },
      { rank: 2, dtype: 'real' },
    ],
    wrappingOp: 'direct',
    impl: () => {
      throw new Error(
        'mul: (column vector) * matrix is not defined; ' +
        'mul requires matrix on the left of a column vector or ' +
        'transpose(v) on the left of a matrix');
    },
    label: 'mul(col vec, mat) → error',
  });
  // (6) Matrix × matrix → matmul.
  ops.registerVariant('mul', {
    argPatterns: [
      { rank: 2, dtype: 'real' },
      { rank: 2, dtype: 'real' },
    ],
    wrappingOp: 'direct',
    impl: (args: any[]) => vo._matMatMul(args[0], args[1]),
    label: 'mul(mat, mat) → matmul',
  });
}

// Try the fast path: when the broadcast head names a known scalar
// primitive (either as a bare ref or as a synthesised functionof
// wrapping the op with the params in some order), dispatch directly
// to the corresponding value-ops elementwise primitive. Returns the
// result Value on success; null when no fast path applies (caller
// falls through to the per-cell `_broadcastApply` path).
//
// Recognises both lowered shapes:
//   1. `broadcast(<ref to op>, A, B, …)` — the form `broadcasted(<op>)
//      (A, B, …)` lowers to.
//   2. `broadcast(functionof(<op>(_arg1_, _arg2_, …)), A, B, …)` —
//      the form `A .* B` and `op.(A, B)` lower to (dotted-binary +
//      `f.(…)` surfaces, plus `broadcast(<op>, …)` direct call when
//      `<op>` is in BUILTIN_FUNCTIONS).
//
// Soundness gates:
//   - All evaluated broadcast args must be Values or coercible via
//     asValue — bare numbers / typed arrays / FLAT all-scalar JS
//     arrays (`_isFlatScalarArray`); nested JS arrays and
//     outerRank-tagged Values stay on the cold path (spec §03
//     nested-vector semantics must not elementwise-dispatch).
//   - Shapes must agree per the ONE shared combine rule,
//     `value-ops._broadcastOutShape` (spec §04): same rank among
//     rank ≥ 1 inputs, per-axis sizes equal or 1 (singleton axes
//     expand by stride-0 reads); rank-0 broadcasts against any rank.
//   - For ops with kwargs the head's `paramKwargs` (functionof case)
//     or arity-bound positional order applies; mismatches return
//     null (cold path takes over).
// A JS array every element of which is a plain scalar (number /
// boolean) — the only JS-array shape the broadcast fast path coerces
// via asValue. Mirrors `broadcast-shape.classifyAxisStructure`'s
// allFlatScalars test MINUS complex {re, im} scalars (planar complex
// coercion is the cold path's job).
function _isFlatScalarArray(v: any[]): boolean {
  for (let i = 0; i < v.length; i++) {
    const t = typeof v[i];
    if (t !== 'number' && t !== 'boolean') return false;
  }
  return true;
}

function _maybeFastBroadcasted(ir: any, ctx: any): any | null {
  _ensureBroadcastedRegistered();
  const args   = ir.args   || [];
  const kwargs = ir.kwargs || {};
  if (args.length < 1) return null;
  const head = args[0];
  if (!head) return null;

  // Resolve the broadcast head to (opName, expected arity, optional
  // body-param reorder). Two IR shapes lower into the broadcast head:
  //   1. `broadcast(<ref to op>, A, B, …)` — bare ref produced by
  //      `broadcasted(<op>)(args)`.
  //   2. `broadcast(functionof(<op>(_arg1_, _arg2_, …)), A, B, …)` —
  //      synthesised functionof produced by dotted-binary `.* ./ …`
  //      and `op.(…)` surfaces, plus direct `broadcast(<op>, …)`
  //      where `<op>` is a known builtin.
  // The variant matcher in ops.ts decides whether the op has a
  // `wrappingOp: 'broadcast'` variant matching the inputs; this
  // function just unpacks the IR shape so dispatch sees flat Values.
  let opName: string | null = null;
  let arity = 0;
  // For the functionof case, `paramReorder[i]` says: the i-th call
  // arg to the body comes from broadcast position `paramReorder[i]`.
  // For the bare-ref case it's identity (i.e., [0, 1, …, arity-1]).
  let paramReorder: number[] | null = null;
  let paramKwargs: string[] | null = null;

  if (head.kind === 'ref' && head.ns === 'self') {
    if (!ops.hasVariantFor(head.name, 'broadcast')) return null;
    opName = head.name;
    arity = -1;  // determined by remaining IR args after head
    paramReorder = null;
    paramKwargs = null;
  } else if (head.kind === 'call' && head.op === 'functionof'
             && Array.isArray(head.params)
             && head.body && head.body.kind === 'call' && head.body.op) {
    const body = head.body;
    if (body.kwargs && Object.keys(body.kwargs).length > 0) return null;
    if (Array.isArray(body.fields) && body.fields.length > 0) return null;
    const bodyArgs: any[] = body.args || [];
    if (bodyArgs.length !== head.params.length) return null;
    if (!ops.hasVariantFor(body.op, 'broadcast')) return null;
    // Body args must each be a ref to one of head.params — `%local`
    // for placeholder formals, `self` for identifier-bound boundary
    // params (spec-shaped bodies §11); the indexOf check disambiguates
    // a self ref naming a param from a closure capture.
    const order = new Array(bodyArgs.length);
    for (let i = 0; i < bodyArgs.length; i++) {
      const ba = bodyArgs[i];
      if (!ba || ba.kind !== 'ref'
          || (ba.ns !== '%local' && ba.ns !== 'self')) return null;
      const idx = head.params.indexOf(ba.name);
      if (idx < 0) return null;
      order[i] = idx;
    }
    opName = body.op;
    arity = bodyArgs.length;
    paramReorder = order;
    paramKwargs = Array.isArray(head.paramKwargs) ? head.paramKwargs : null;
  } else {
    return null;
  }

  // Resolve the broadcast call's positional / kwarg arg sources into
  // a positional array of length `arity` (the head's declared param
  // order — equivalent to `head.params` for functionof, or the op's
  // declared order for bare refs).
  const kwNames = Object.keys(kwargs);
  let sources: any[];
  if (kwNames.length === 0) {
    const posArgs = args.slice(1);
    if (arity === -1) arity = posArgs.length;
    if (posArgs.length !== arity) return null;
    sources = posArgs;
  } else if (args.length === 1) {
    // Pure kwargs form. Need paramKwargs (functionof case) — bare-ref
    // heads don't carry surface kwarg names at this layer.
    if (!paramKwargs || arity === -1 || paramKwargs.length !== arity) return null;
    sources = new Array(arity);
    for (let i = 0; i < arity; i++) {
      const name = paramKwargs[i];
      if (!(name in kwargs)) return null;
      sources[i] = kwargs[name];
    }
    // Reject extra kwargs the head doesn't declare.
    for (const n of kwNames) {
      if (paramKwargs.indexOf(n) === -1) return null;
    }
  } else {
    return null;  // mixed pos+kw — cold path
  }

  // Evaluate broadcast arg IRs to Values. The fast path requires
  // every evaluated arg to coerce to a Value (rank ≥ 0); anything
  // else (records, tables, function objects) bails to the cold path.
  const valueLib = require('./value.ts');
  const vo = require('./value-ops.ts');   // module-cached; cycle-safe
  const inputs: any[] = new Array(arity);
  for (let i = 0; i < arity; i++) {
    const v = ctx.evaluateExpr(sources[i], ctx.env);
    if (v == null) return null;
    if (valueLib.isValue(v)) {
      inputs[i] = v;
      continue;
    }
    if (typeof v === 'number' || typeof v === 'boolean'
        || v instanceof Float64Array
        || (ArrayBuffer.isView(v) && typeof (v as any).length === 'number')) {
      inputs[i] = valueLib.asValue(v);
      continue;
    }
    // Plain FLAT-SCALAR JS arrays coerce to rank-1 Values (the same
    // all-flat-scalars predicate `classifyAxisStructure` uses, so the
    // two paths agree on which arrays are flat collections). Anything
    // else — nested arrays (the Ref-wrap hold-constant idiom wraps
    // VALUES; nested numeric arrays carry classifyNestedJSArray's
    // outer-axis semantics), arrays of records, ragged input — stays
    // on the cold path, which owns those semantics and diagnostics.
    if (Array.isArray(v) && _isFlatScalarArray(v)) {
      inputs[i] = valueLib.asValue(v);
      continue;
    }
    return null;
  }

  // Reorder per the body's param order if needed (functionof case
  // with swizzled body args).
  let opInputs: any[];
  if (paramReorder) {
    opInputs = new Array(arity);
    for (let i = 0; i < arity; i++) opInputs[i] = inputs[paramReorder[i]];
  } else {
    opInputs = inputs;
  }

  // Shape compatibility gate — the fast path implements the spec §04
  // collection rule directly: all rank ≥ 1 inputs must share the same
  // rank (no implicit axis insertion — `addaxes` aligns), and along
  // each axis sizes must be equal or 1 (a singleton expands by
  // repetition; the value-ops elementwise impls realise it via
  // stride-0 reads — the shared `_broadcastOutShape` combiner is the
  // one owner of the rule). Rank-0 inputs broadcast trivially.
  //
  // Bails to the cold path (return null) on:
  //   - a Value with NESTED-VECTOR semantics (outerRank < rank): per
  //     cell the body sees an inner sub-Value WHOLE, not a scalar —
  //     `_broadcastApply` + `classifyAxisStructure` own that walk
  //     (and its §04 outer-axis rank check, which compares OUTER
  //     ranks, not storage ranks — elementwise dispatch here would be
  //     semantically wrong, not just slower);
  //   - incompatible ranks/sizes, so `_broadcastApply`'s spec-§04
  //     diagnostics (with the addaxes hint) fire instead of a
  //     value-ops-flavoured error.
  for (const v of opInputs) {
    if (typeof v.outerRank === 'number'
        && v.outerRank !== v.shape.length) {
      return null;
    }
  }
  try {
    vo._broadcastOutShape(opName!, opInputs.map((v: any) => v.shape));
  } catch (_e) {
    return null;
  }

  // Dispatch through the variant registry. The 'broadcast'-wrapping
  // variant for `opName` (registered above) maps to the corresponding
  // value-ops elementwise impl. Returns null if no variant matches
  // (e.g. opName has no broadcast variant — `head` was a non-
  // broadcastable builtin).
  return ops.dispatchVariant(opName!, opInputs, { wrappingOp: 'broadcast' });
}

// Read a bare-builtin / kernel-name from a head or argument IR node,
// accepting the three shapes `_resolveKernelName` (sampler.ts) accepts:
//   {kind:'ref', name}  ·  {kind:'call', op}  ·  {kind:'lit', value:'…'}
// Returns null for any other shape.
function _headNameOf(node: any): string | null {
  if (!node || typeof node !== 'object') return null;
  if (node.kind === 'ref' && typeof node.name === 'string') return node.name;
  if (node.kind === 'call' && typeof node.op === 'string') return node.op;
  if (node.kind === 'const' && typeof node.name === 'string') return node.name;
  if (node.kind === 'lit' && typeof node.value === 'string') return node.value;
  return null;
}

// Synthesize a `{params, paramKwargs, body}` fn descriptor for a
// broadcast whose head is a bare BUILTIN function (`_resolveFn` returned
// null — the head is not a user fn / functionof / module-aliased ref).
// The per-cell body applies the builtin to fresh params; the broadcast
// call's own positional/kwarg shape fixes the arity and, for ordered-
// named forms (`record`/`joint`/…), the field names. General over any
// builtin head; `broadcast(record, rate = arr)` and (the brief's bare
// form) `broadcast(builtin_logdensityof, K, kin, obs)` both land here.
function _synthBuiltinHeadFn(head: any, args: any[], kwargs: any): any {
  const name = _headNameOf(head);
  if (name == null) return null;
  const builtins = require('./builtins.ts');
  // FIELD_FORMS (record/joint/jointchain/cartprod/table) is lower.ts's
  // single source of truth for the ordered-named constructors that carry
  // their entries as a `fields:[{name,value}]` array rather than kwargs.
  const lower = require('./lower.ts');
  const isFieldForm = lower._internal.FIELD_FORMS.has(name);
  // Only builtins that evaluate to a VALUE in a call position: the
  // elementary / measure-eval builtin FUNCTIONS and the ordered-named
  // structural constructors (`record`/`joint`/…). Distributions,
  // measure ops, constants and sets are not value-callable heads.
  if (!builtins.BUILTIN_FUNCTIONS.has(name) && !isFieldForm) return null;

  const kwKeys = Object.keys(kwargs || {});
  if (kwKeys.length > 0) {
    // kwarg form — arity and surface names come from the broadcast
    // kwargs (kept in key order). `paramKwargs` carries the surface
    // names so the cold path's kwarg→source matching binds correctly.
    const params = kwKeys.map((_k, i) => '__b' + i);
    let body: any;
    if (isFieldForm) {
      body = {
        kind: 'call', op: name,
        fields: kwKeys.map((k, i) => ({
          name: k, value: { kind: 'ref', ns: '%local', name: params[i] },
        })),
      };
    } else {
      /* c8 ignore start -- general §04 path: a non-field-form builtin head
         with kwargs. The current determiniser emission uses the field-form
         (`record`) kwarg shape and the functionof-head form, both exercised
         in broadcast-builtin-eval.test.ts; this branch is retained for
         spec completeness (any builtin head). */
      const bk: any = {};
      for (let i = 0; i < kwKeys.length; i++) {
        bk[kwKeys[i]] = { kind: 'ref', ns: '%local', name: params[i] };
      }
      body = { kind: 'call', op: name, kwargs: bk };
    }
    /* c8 ignore stop */
    return { params, paramKwargs: kwKeys.slice(), body, paramName: params[0] };
  }
  /* c8 ignore start -- positional form: general §04 broadcast-over-builtin
     (arity from the data args). The determiniser emits the functionof-head
     (outer) + field-form record (inner) shapes, both exercised in
     broadcast-builtin-eval.test.ts. */
  const n = args.length - 1;
  if (n < 0) return null;
  const params: string[] = [];
  for (let i = 0; i < n; i++) params.push('__b' + i);
  const body = {
    kind: 'call', op: name,
    args: params.map((p) => ({ kind: 'ref', ns: '%local', name: p })),
  };
  return { params, paramKwargs: null, body, paramName: params[0] };
  /* c8 ignore stop */
}

// Hold symbolic kernel-NAME data args inline in the body. Per spec §07 a
// `builtin_logdensityof` / transport kernel arg is a bare kernel NAME,
// not an evaluable value: it must reach `_resolveKernelName` (sampler.ts)
// as a name-shaped IR node. A broadcast data-arg source that reads as a
// known built-in kernel is loop-invariant and can never evaluate as a
// value, so we substitute its IR directly into the body (replacing the
// param ref) and drop it from the params/sources the zip iterates. Fully
// general — fires for any head whose data arg is a bare kernel name,
// never for ordinary value args; returns the fn/sources unchanged when
// there is none.
function _inlineHeldKernelArgs(fn: any, sources: any[]): any {
  const densityPrims = require('./density-prims.ts');
  const subst = new Map<string, any>();
  const keep: number[] = [];
  for (let i = 0; i < fn.params.length; i++) {
    const kname = _headNameOf(sources[i]);
    if (kname != null && densityPrims.isBuiltinKernel(kname)) {
      subst.set(fn.params[i], sources[i]);
    } else {
      keep.push(i);
    }
  }
  if (subst.size === 0) return { fn, sources };
  const irWalk = require('./ir-walk.ts');
  const newBody = irWalk.mapIR(fn.body, (nd: any) => {
    if (nd && nd.kind === 'ref' && typeof nd.name === 'string'
        && (nd.ns === '%local' || nd.ns === 'self') && subst.has(nd.name)) {
      return subst.get(nd.name);
    }
    return nd;
  });
  const params = keep.map((i) => fn.params[i]);
  const paramKwargs = Array.isArray(fn.paramKwargs)
    ? keep.map((i) => fn.paramKwargs[i]) : fn.paramKwargs;
  return {
    fn: { params, paramKwargs, body: newBody, paramName: params[0] },
    sources: keep.map((i) => sources[i]),
  };
}

// =====================================================================
// Rank-d (nested) kernel-broadcast DENSITY — flattened evaluation
// =====================================================================
//
// The determiniser flattens a composed/nested `iid` density (spec §06:
// `logdensityof(iid(M,n),x) = Σᵢ logdensityof(M,xᵢ)`, recursing through
// nesting → Σ over ALL leaves, order-independent) to ONE
// `sum(broadcast(builtin_logdensityof, K, <params>, <obs>))` — no
// `functionof`/`get0` unroll. For a ONCE-nested `iid` the kernel params
// are a rank-1 size-1 `record`-array (`mu = [0.0]`) broadcasting
// against a rank-1 obs; that shape already worked via the generic
// per-cell `_broadcastApply` below. For a TWICE-or-more-nested `iid`
// the params/obs are rank-≥2 bracket literals (e.g. `mu = [[0.0]]`
// against a `[3,2]` obs) — per spec §03 "vectors of vectors are not
// matrices", such a literal carries an `outerRank` tag, and the
// generic broadcast loop (`classifyAxisStructure` / `_broadcastApply`)
// treats an outerRank-tagged collection as "loop the OUTER axis only,
// pass the inner block WHOLE per cell" — the right convention for a
// per-row VECTOR-valued kernel, but the WRONG one here: a univariate
// kernel's `x` must be a plain scalar per cell, and EVERY bracket-
// nesting level is an independent iid axis to loop over, not a block
// to hand the kernel whole. Left as-is, that mismatch either throws
// (`builtin_logdensityof(...): univariate kernel needs a scalar
// variate`) or silently mis-shapes, and since this runs inside the
// FIXED-PHASE resolver (`fixed-values.ts`), the exception is caught
// and turned into "unresolved" — surfacing many layers up as the
// unhelpful `no derivation for '<binding>'`, not a shape diagnostic.
//
// `_tryFlattenKernelBroadcastDensity` recognises the exact flattened
// shape (`builtin_logdensityof(<kernel>, <kernel_input>, <x>)` per
// cell, with `<kernel_input>` sourced from a bare
// `broadcast(record, field = …, …)`) and, ONLY when `<x>`'s evaluated
// shape is genuinely rank ≥ 2, computes the density directly: every
// field + `x` is read via its raw `.shape`/`.data` (the `outerRank`
// tag is ignored here on purpose — ALL axes are loop axes for this
// primitive), unified via the ordinary equal-or-1 broadcast rule
// (`value-ops._broadcastOutShape`, the same rule every other
// elementwise op already uses), and `builtin_logdensityof` is invoked
// once per fully-flattened leaf; the result is an ordinary (untagged)
// flat Value of the broadcast shape, which the enclosing `sum(...)`
// then reduces over ALL leaves exactly as spec §06 requires. Returns
// `null` — meaning "not this shape, use the existing generic path
// unchanged" — for anything that doesn't match, so the already-
// verified rank-1 (simple, once-nested `iid`) path never routes
// through here and keeps its exact prior behaviour.
function _tryFlattenKernelBroadcastDensity(fn: any, sources: any[], ctx: any): any {
  if (!fn || !fn.body || fn.body.kind !== 'call'
      || fn.body.op !== 'builtin_logdensityof') return null;
  const bodyArgs = fn.body.args;
  if (!Array.isArray(bodyArgs) || bodyArgs.length !== 3) return null;
  const densityPrims = require('./density-prims.ts');
  const kernelName = _headNameOf(bodyArgs[0]);
  if (kernelName == null || !densityPrims.isBuiltinKernel(kernelName)) return null;
  // Exactly 2 params should remain — kernel_input and x — once the
  // kernel NAME itself was held/inlined by `_inlineHeldKernelArgs`.
  if (!Array.isArray(fn.params) || fn.params.length !== 2
      || !Array.isArray(sources) || sources.length !== 2) {
    return null;
  }
  const kernelInputSourceIR = sources[0];
  const xSourceIR = sources[1];
  // Only the determiniser's flattened-iid params shape: a bare
  // `broadcast(record, field = …, …)` kwarg-form call. Anything else
  // (a positional kernel input, a plain record, a bare scalar) falls
  // through to the generic path unchanged — out of scope here.
  if (!kernelInputSourceIR || kernelInputSourceIR.kind !== 'call'
      || kernelInputSourceIR.op !== 'broadcast'
      || !Array.isArray(kernelInputSourceIR.args)
      || kernelInputSourceIR.args.length !== 1) return null;
  const recordHead = kernelInputSourceIR.args[0];
  if (!recordHead || recordHead.kind !== 'ref' || recordHead.name !== 'record') return null;
  const fieldKwargs = kernelInputSourceIR.kwargs;
  if (!fieldKwargs || Object.keys(fieldKwargs).length === 0) return null;

  const xVal = ctx.evaluateExpr(xSourceIR, ctx.env);
  // Bail unless `x` is genuinely rank ≥ 2 — the already-verified rank-1
  // (simple iid) case never reaches the flattened path.
  if (!valueLib.isValue(xVal) || xVal.shape.length < 2) return null;

  const fieldNames = Object.keys(fieldKwargs);
  const fieldVals: any[] = fieldNames.map(
    (k) => ctx.evaluateExpr(fieldKwargs[k], ctx.env));
  // Every field must be a plain scalar or a Value — anything else
  // (a nested JS array that never collapsed to a Value, a record, a
  // kernel/fn object) is outside this primitive's contract; defer to
  // the generic path so it can raise its own diagnostic.
  for (const v of fieldVals) {
    if (v != null && typeof v === 'object' && !valueLib.isValue(v)) return null;
  }

  // Gather every operand's FULL shape (ignore any `outerRank` tag —
  // every bracket-nesting level is a loop axis for this scoring
  // broadcast, spec §06 "Σ over all leaves").
  const shapes: number[][] = [xVal.shape];
  for (const v of fieldVals) if (valueLib.isValue(v)) shapes.push(v.shape);
  let outShape: number[];
  try {
    outShape = valueOps._broadcastOutShape('builtin_logdensityof', shapes).outShape;
  } catch (_e) {
    return null;   // shape mismatch — let the generic path raise its own diagnostic
  }
  const rank = outShape.length;
  function stridesFor(shape: number[]): number[] {
    if (shape.length === 0) return new Array(rank).fill(0);
    const st = new Array(rank);
    let acc = 1;
    for (let a = rank - 1; a >= 0; a--) {
      st[a] = (shape[a] === 1) ? 0 : acc;
      acc *= shape[a];
    }
    return st;
  }
  const xStrides = stridesFor(xVal.shape);
  const fieldStrides = fieldVals.map(
    (v) => (valueLib.isValue(v) ? stridesFor(v.shape) : null));

  const total = outShape.reduce((a: number, b: number) => a * b, 1);
  const out = new Float64Array(total);
  const idx = new Array(rank).fill(0);
  for (let flat = 0; flat < total; flat++) {
    let rem = flat;
    for (let a = rank - 1; a >= 0; a--) {
      idx[a] = rem % outShape[a];
      rem = Math.floor(rem / outShape[a]);
    }
    let xOff = 0;
    for (let a = 0; a < rank; a++) xOff += idx[a] * xStrides[a];
    const kernelInput: any = {};
    for (let f = 0; f < fieldNames.length; f++) {
      const v = fieldVals[f];
      if (!valueLib.isValue(v)) { kernelInput[fieldNames[f]] = v; continue; }
      let off = 0;
      const st = fieldStrides[f]!;
      for (let a = 0; a < rank; a++) off += idx[a] * st[a];
      kernelInput[fieldNames[f]] = v.data[off];
    }
    out[flat] = densityPrims.builtinLogdensityof(kernelName, kernelInput, xVal.data[xOff]);
  }
  return { shape: outShape, data: out };
}

function _broadcastLogical(ir: any, ctx: any): any {
  // Fast path: `broadcasted(<scalar_op>)(args)` and the equivalent
  // dotted-binary / `op.(…)` lowered forms dispatch directly to the
  // engine's batched-elementwise primitive (engine-concepts §20.1).
  // Bypasses per-cell iteration; uses value-ops' flat-data loops.
  // Returns null when the head doesn't match a known scalar primitive
  // OR the args don't evaluate to coercible Values — the per-cell
  // path below then runs as before.
  const fast = _maybeFastBroadcasted(ir, ctx);
  if (fast !== null) return fast;

  const args   = ir.args   || [];
  const kwargs = ir.kwargs || {};
  if (args.length < 1) throw new Error('broadcast: no function argument');
  // Head resolution: a user fn / functionof / module-aliased ref via
  // `_resolveFn`; otherwise a bare BUILTIN head (`record`,
  // `builtin_logdensityof`, …) synthesised from the call's own shape
  // (spec §04 value-level broadcast over a builtin head is legal).
  let fn = ctx.resolveFn(args[0], ctx.env);
  if (!fn) fn = _synthBuiltinHeadFn(args[0], args, kwargs);
  if (!fn) throw new Error('broadcast: first arg must be a function');
  const kwargKeys = Object.keys(kwargs);
  const sources = new Array(fn.params.length);
  if (kwargKeys.length > 0) {
    // kwargs form: match by surface kwarg name first
    // (paramKwargs), fall back to internal placeholder name (params).
    for (let i = 0; i < fn.params.length; i++) {
      const surface = (fn.paramKwargs && fn.paramKwargs[i]) || fn.params[i];
      if (kwargs[surface] != null) sources[i] = kwargs[surface];
      /* c8 ignore next 3 -- defensive: internal-placeholder-name fallback and the missing-argument guard (surface-name match is the exercised path) */
      else if (kwargs[fn.params[i]] != null) sources[i] = kwargs[fn.params[i]];
      else throw new Error('broadcast: no argument for parameter '
        + (surface || fn.params[i]));
    }
  } else {
    const posArgs = args.slice(1);
    /* c8 ignore next 4 -- defensive: positional arity guard */
    if (posArgs.length !== fn.params.length) {
      throw new Error('broadcast: expected ' + fn.params.length
        + ' positional arrays, got ' + posArgs.length);
    }
    for (let i = 0; i < fn.params.length; i++) sources[i] = posArgs[i];
  }
  // Held symbolic kernel-name args (e.g. `Poisson` in
  // `builtin_logdensityof.(Poisson, …)`) are inlined into the body so
  // they reach `_resolveKernelName` as IR, not as an evaluated value.
  const held = _inlineHeldKernelArgs(fn, sources);
  fn = held.fn;
  // Rank-d (nested, d≥2) kernel-broadcast DENSITY: try the flattened
  // evaluator first (see its header comment). It only fires for the
  // exact `builtin_logdensityof(<kernel>, <record-broadcast>, <x>)`
  // shape with a genuinely rank-≥2 `x`; anything else (including every
  // rank-1 case already covered by tests) returns null and falls
  // through to the unchanged generic path below.
  const flattened = _tryFlattenKernelBroadcastDensity(fn, held.sources, ctx);
  if (flattened !== null) return flattened;
  const inputs: any = held.sources.map((s: any) => ctx.evaluateExpr(s, ctx.env));
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

// Eagerly register the broadcasted-primitives variants at module
// load (engine-concepts §18.2 / §20.1). value-ops is already
// required at the top of this file so there's no cycle risk.
_ensureBroadcastedRegistered();

// Eagerly register the `mul` direct-wrapping variants — the
// rank-based shape switch that previously lived in valueOps.mul
// (engine-concepts §18.2).
_ensureMulDirectRegistered();

// Eagerly register the atom-batched fast-path variants for
// add/sub/neg/mul (engine-concepts §18.2 — P1 follow-up). Variant
// `batched` slots make `ops.dispatch` route atom-batched inputs
// uniformly through the registry; legacy `value-ops.addN/subN/
// negN/mulN` callers now have a registry path available.
_ensureAtomBatchedRegistered();

// Attach atom-indep `logical` impls for the scalar primitives, migrating
// their single-point eval onto `ops.dispatch` (engine-concepts §18 —
// "one source of truth per op"). Phased-additive: one family at a time,
// conformance-tested (test/ops-conformance.test.ts) against the legacy
// `ARITH_OPS` reference, which stays as the fallback until every family
// has migrated (then ARITH_OPS becomes a derived facade).
_registerScalarLogicals();

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
