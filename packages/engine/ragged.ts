'use strict';

// =====================================================================
// ragged.ts — the ragged-per-atom value kind (engine-concepts §2.3)
// =====================================================================
//
// ONE owner of the ragged Value: a vector of variable-length per-atom
// arrays, stored flat (Julia `ArraysOfArrays.VectorOfArrays` / CSR /
// jagged layout). The sibling of the uniform shape-tagged Value (§2.1):
// equal-size inner ⇒ uniform `[N,…k]` Value (ArrayOfSimilarArrays);
// genuinely ragged ⇒ this kind (VectorOfArrays). Uniform is the special
// case `offsets[i] = i·k`.
//
//   { ragged: true,
//     data:    Float64Array,   // all atoms' elements concatenated
//     offsets: Int32Array,     // length N+1; atom i = data[offsets[i] : offsets[i+1]]
//     kernelShape: number[] }  // fixed leading per-element dims (ragged
//                              // in the LAST axis only): [] scalar points,
//                              // [d] for d-dim points (= VectorOfArrays.kernel_size)
//
// The load-bearing payoff (§2.3): a length-preserving elementwise map
// reuses the flat broadcast fast path verbatim with `offsets` carried as
// metadata (`raggedMapFlat`); a segmented reduction collapses to a
// uniform `[N]` Value (`raggedSegmentReduce`). Only length-changing /
// cross-boundary ops need offset-aware kernels.
//
// Dependency-free leaf — produces/consumes plain `{shape, data}` Value
// literals; no require cycle. Mirrors value-set.ts's "one vocabulary
// owner" pattern.

function isRagged(v: any): boolean {
  return v != null && typeof v === 'object' && v.ragged === true
    && v.data instanceof Float64Array && v.offsets instanceof Int32Array;
}

// Construct a ragged value from a flat buffer + offsets (+ optional
// kernelShape). Validates the offsets are monotonic and frame the data.
function ragged(data: Float64Array, offsets: Int32Array,
                kernelShape?: number[]): any {
  const ks = kernelShape || [];
  const stride = _prod(ks);
  if (offsets.length < 1) {
    throw new Error('ragged: offsets must have length N+1 (≥1)');
  }
  if (offsets[0] !== 0 || offsets[offsets.length - 1] !== data.length) {
    throw new Error('ragged: offsets must run 0 … data.length (got '
      + offsets[0] + ' … ' + offsets[offsets.length - 1]
      + ', data.length=' + data.length + ')');
  }
  for (let i = 1; i < offsets.length; i++) {
    if (offsets[i] < offsets[i - 1]) {
      throw new Error('ragged: offsets must be non-decreasing (at ' + i + ')');
    }
    if (stride > 1 && (offsets[i] - offsets[i - 1]) % stride !== 0) {
      throw new Error('ragged: segment ' + (i - 1) + ' length not a multiple '
        + 'of kernelShape stride ' + stride);
    }
  }
  return { ragged: true, data, offsets, kernelShape: ks };
}

// Build a ragged value from a JS array of per-atom arrays (each a JS
// array or Float64Array of scalars — the VectorOfVectors case). The
// common constructor for matPoissonProcess-style output.
function raggedFromArrays(arrs: Array<ArrayLike<number>>): any {
  const N = arrs.length;
  const offsets = new Int32Array(N + 1);
  let total = 0;
  for (let i = 0; i < N; i++) { total += arrs[i].length; offsets[i + 1] = total; }
  const data = new Float64Array(total);
  let p = 0;
  for (let i = 0; i < N; i++) {
    const a = arrs[i];
    for (let j = 0; j < a.length; j++) data[p++] = a[j];
  }
  return { ragged: true, data, offsets, kernelShape: [] };
}

// Number of atoms (= N).
function raggedCount(v: any): number {
  return v.offsets.length - 1;
}

// Flat-element span of atom i (offsets[i+1] − offsets[i]).
function raggedSpan(v: any, i: number): number {
  return v.offsets[i + 1] - v.offsets[i];
}

// Per-atom element COUNT (span / kernel stride): the number of points
// atom i drew (the ragged axis length), independent of kernelShape.
function raggedElemCount(v: any, i: number): number {
  const stride = _prod(v.kernelShape);
  return raggedSpan(v, i) / (stride || 1);
}

// Atom i as a (uniform) Value VIEW: shape `[count_i, …kernelShape]`
// (scalar points → `[count_i]`), backed by a subarray of `data` (no
// copy). The boundary handoff back to the §2.1 uniform path.
function raggedElem(v: any, i: number): any {
  const lo = v.offsets[i], hi = v.offsets[i + 1];
  const sub = v.data.subarray(lo, hi);
  const count = raggedElemCount(v, i);
  const shape = v.kernelShape.length === 0 ? [count] : [count, ...v.kernelShape];
  return { shape, data: sub };
}

// Materialise to a plain JS array of Float64Arrays (one per atom) — the
// display / densify fallback (scalar-points form; with a kernelShape the
// per-atom slice keeps the flat element order).
function raggedToNested(v: any): Float64Array[] {
  const out: Float64Array[] = [];
  for (let i = 0; i < raggedCount(v); i++) {
    out.push(v.data.slice(v.offsets[i], v.offsets[i + 1]));
  }
  return out;
}

// Two ragged values share structure iff identical offsets (+ kernelShape)
// — the ragged analog of "same shape" for a same-structure binary op.
function raggedSameStructure(a: any, b: any): boolean {
  if (a.offsets.length !== b.offsets.length) return false;
  if (_prod(a.kernelShape) !== _prod(b.kernelShape)) return false;
  for (let i = 0; i < a.offsets.length; i++) {
    if (a.offsets[i] !== b.offsets[i]) return false;
  }
  return true;
}

// Length-preserving elementwise map (§2.3): apply `flatFn` to the flat
// `data` as ONE uniform Value (shape `[data.length]`) and re-wrap with
// the SAME offsets. The reuse of the flat broadcast fast path — boundaries
// are irrelevant to an elementwise op, so `offsets` is pure metadata.
// `flatFn(flatValue) → flatValue'` must preserve length.
function raggedMapFlat(v: any, flatFn: (flat: any) => any): any {
  const flat = { shape: [v.data.length], data: v.data };
  const out = flatFn(flat);
  if (!(out && out.data instanceof Float64Array)) {
    throw new Error('raggedMapFlat: flatFn must return a Value with Float64Array data');
  }
  if (out.data.length !== v.data.length) {
    throw new Error('raggedMapFlat: flatFn must be length-preserving ('
      + out.data.length + ' ≠ ' + v.data.length + ')');
  }
  return { ragged: true, data: out.data, offsets: v.offsets, kernelShape: v.kernelShape };
}

// Segmented reduction (§2.3): reduce each atom's elements to one scalar,
// producing a UNIFORM `[N]` Value — the ragged→uniform collapse (e.g.
// `sum.(ragged)`). `reduce(acc, x) → acc`, seeded with `init`.
function raggedSegmentReduce(v: any, reduce: (acc: number, x: number) => number,
                             init: number): any {
  const N = raggedCount(v);
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let acc = init;
    for (let p = v.offsets[i]; p < v.offsets[i + 1]; p++) acc = reduce(acc, v.data[p]);
    out[i] = acc;
  }
  return { shape: [N], data: out };
}

// Merge n ragged values over the SAME N atoms into one: atom a becomes the
// concatenation of each part's atom-a slice. The point-process superposition
// union (spec §08) — by the superposition theorem a process with intensity
// Σ_i weighted(w_i, s_i) is the union of independent component processes, so
// matPoissonProcess samples each component then merges by atom. Order within
// an atom is irrelevant (point sets are unordered). All parts must share atom
// count N and kernelShape stride.
function raggedMerge(parts: any[]): any {
  if (parts.length === 0) throw new Error('raggedMerge: needs ≥ 1 part');
  if (parts.length === 1) return parts[0];
  const N = raggedCount(parts[0]);
  const ks = parts[0].kernelShape;
  const ksStride = _prod(ks);
  for (let i = 1; i < parts.length; i++) {
    if (raggedCount(parts[i]) !== N) {
      throw new Error('raggedMerge: atom-count mismatch (' + raggedCount(parts[i]) + ' ≠ ' + N + ')');
    }
    if (_prod(parts[i].kernelShape) !== ksStride) {
      throw new Error('raggedMerge: kernelShape stride mismatch');
    }
  }
  const offsets = new Int32Array(N + 1);
  let total = 0;
  for (let a = 0; a < N; a++) {
    for (let i = 0; i < parts.length; i++) total += parts[i].offsets[a + 1] - parts[i].offsets[a];
    offsets[a + 1] = total;
  }
  const data = new Float64Array(total);
  let p = 0;
  for (let a = 0; a < N; a++) {
    for (let i = 0; i < parts.length; i++) {
      const pi = parts[i];
      for (let x = pi.offsets[a]; x < pi.offsets[a + 1]; x++) data[p++] = pi.data[x];
    }
  }
  return ragged(data, offsets, ks);
}

function _prod(dims: number[]): number {
  let p = 1;
  for (let i = 0; i < dims.length; i++) p *= dims[i];
  return p;
}

module.exports = {
  isRagged,
  ragged,
  raggedFromArrays,
  raggedCount,
  raggedSpan,
  raggedElemCount,
  raggedElem,
  raggedToNested,
  raggedSameStructure,
  raggedMapFlat,
  raggedSegmentReduce,
  raggedMerge,
};
