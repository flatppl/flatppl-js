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

module.exports = {
  // Re-export for tests that want to call the logical impls directly
  // (rather than through `ops.dispatch`).
  _crossLogical,
  _selfOuterLogical,
};
