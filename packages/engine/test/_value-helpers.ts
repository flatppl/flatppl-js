'use strict';

// Shared helpers for tests that need to compare engine outputs that
// may be returned as either shape-explicit Values (the spec §03 /
// engine-concepts §2.1 contract) or legacy nested-JS-array form.
//
// After the §2.1 migration the producers (vector, fill, zeros, ones,
// onehot, linspace, extlinspace, cumsum, cumprod, softmax,
// logsoftmax, l1unit, l2unit, cat, partition, bincounts, array,
// rowstack, colstack, eye) emit Values. Existing tests that used
// `assert.deepEqual(ev(...), [1, 2, 3])` need a normaliser to keep
// passing while still being readable.
//
// `toJS(v)` returns the JS-form (nested arrays of numbers) for any
// Value / Float64Array / nested-array input. Use it on either side
// of deepEqual, OR wrap only the produced side and keep the literal
// expectation as-is.

const valueLib = require('../value.ts');

function toJS(v: any): any {
  if (v == null) return v;
  if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') return v;
  if (valueLib.isValue(v)) {
    let d = valueLib.densify(v);
    // densify() preserves a transpose tag (`t: 'T'`) as a view; apply
    // it physically here so the JS view matches the LOGICAL shape.
    if (d.t === 'T' || d.t === 'A') {
      d = _applyTransposeView(d);
    }
    return _reshapeFlat(Array.from(d.data), d.shape, 0);
  }
  if (v instanceof Float64Array || v instanceof Int32Array || v instanceof Uint8Array) {
    return Array.from(v);
  }
  if (Array.isArray(v)) {
    return v.map(toJS);
  }
  if (typeof v === 'object') {
    // record-like — recurse fields
    const out: any = {};
    for (const k in v) {
      if (Object.prototype.hasOwnProperty.call(v, k)) out[k] = toJS(v[k]);
    }
    return out;
  }
  return v;
}

// Physically materialise a transposed view (Klein-4 'T' tag) so the
// data layout matches the logical (already-transposed) shape.
function _applyTransposeView(v: any): any {
  if (v.shape.length < 2) return v;   // rank-1 transpose: tag only
  const rank = v.shape.length;
  // Spec: transpose swaps the LAST two axes regardless of rank.
  const m = v.shape[rank - 2];
  const n = v.shape[rank - 1];
  // For rank=2, the source had logical [n, m] (after swap = [m, n]).
  // The source DATA is in source-shape row-major: index s_ij = i*n + j
  // for [n, m]. We want dest[i*n + j] = source at logical (i, j) of
  // the [m, n] view = source at swapped (j, i) physical.
  if (rank === 2) {
    const src = v.data;
    const out = new Float64Array(m * n);
    // src has physical shape [n, m]: src[a*m + b] for a in [0,n), b in [0,m).
    // The view (after applying transpose tag) at logical (i, j) for i in [0,m), j in [0,n)
    // is src[j*m + i].
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        out[i * n + j] = src[j * m + i];
      }
    }
    return { shape: [m, n], data: out };
  }
  // Higher-rank: only the last two axes swap; leading axes preserved.
  // Treat the leading axes as a flat batch, then loop matrix transposes.
  let lead = 1;
  for (let i = 0; i < rank - 2; i++) lead *= v.shape[i];
  const src = v.data;
  const out = new Float64Array(lead * m * n);
  // Source physical shape: leading × [n, m] (since view is [m, n]).
  for (let l = 0; l < lead; l++) {
    const sBase = l * m * n;   // same numel as dest per batch slice
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        out[sBase + i * n + j] = src[sBase + j * m + i];
      }
    }
  }
  return { shape: v.shape.slice(), data: out };
}

function _reshapeFlat(flat: number[], shape: number[], depth: number): any {
  if (shape.length - depth <= 1) {
    if (depth >= shape.length) return flat[0];
    return flat.slice();
  }
  const n = shape[depth];
  const tail = shape.slice(depth + 1);
  const innerLen = tail.reduce((a, b) => a * b, 1);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const slice = flat.slice(i * innerLen, (i + 1) * innerLen);
    out[i] = _reshapeFlat(slice, shape, depth + 1);
  }
  return out;
}

module.exports = { toJS };
