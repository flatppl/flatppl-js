'use strict';

// =====================================================================
// broadcast-shape — value-level axis classification for broadcast
// =====================================================================
//
// Maps a heterogeneous JS runtime value (Value, typed array, JS array,
// scalar, function/kernel object, …) onto a uniform "slot descriptor"
// the broadcast loop consumes. The descriptor either marks the input
// as loop-invariant (`coll: false`, hold-constant `val`) or describes
// its OUTER axis structure: how many axes the broadcast iterates over,
// what their sizes are, and how to extract the per-cell value.
//
// Two collection flavours:
//
//   - `flat` — a shape-explicit Value of rank ≥ 1. Every axis is a
//     loop axis; per cell the body sees a scalar (or {re, im} for
//     complex dtype). Row-major strides; size-1 axes carry stride 0
//     so they auto-repeat under the broadcast index.
//
//   - `nested` — a JS array whose innermost leaves are Values (or
//     scalars). The JS-array nesting depth is the OUTER rank; per
//     cell the body sees the innermost leaf WHOLE. This is the
//     Julia-`Ref(C)`-style hold-constant idiom: `[C]` is a length-1
//     outer collection wrapping the rank-1 C; per cell, C is passed
//     intact. Multi-level nesting (`[[C]]`, `[[V1, V2], [V3, V4]]`)
//     is supported; rectangularity is enforced at every depth.
//
// Spec compliance: §04's "all collection arguments must have the same
// number of axes" rule applies to OUTER rank. A flat shape-[2, 3]
// Value (outerRank=2) and a nested-vector `[[V1, V2]]` (outerRank=2)
// compose; a flat shape-[3] Value and a nested-vector `[[V1, V2]]`
// (outerRank 1 vs 2) do not. Singleton expansion is per-axis as in
// the spec.
//
// Why this lives in its own module: the classifier is used by the
// broadcast loop in sampler.ts but conceptually wants to be re-used
// by any future broadcast consumer (kernel-broadcast lowering,
// atom-batched fast paths, aggregate's outer-axis classification at
// `get(arr, .i, .j)` sites). Moving it out of sampler.ts puts the
// API in one place and lets sampler.ts stay focused on the iteration
// loop and per-cell evaluation.
//
// Output stacking is also here: `tryStackBroadcastCells` takes a flat
// list of per-cell results plus the broadcast outer shape and packs
// them into a single shape-explicit Value when shapes line up
// (gufunc-style stacking), or returns null so the broadcast loop can
// fall back to a nested JS array.

const valueLib = require('./value.ts');

// ---------------------------------------------------------------------
// Slot descriptors
// ---------------------------------------------------------------------
//
// The classifier returns one of three slot shapes. All are plain
// JS objects so the iteration loop can switch on `coll` / `kind`
// without type-narrowing tooling.
//
//   HoldConstantSlot:  { coll: false, val }
//     `val` is whatever the body sees on each call.
//
//   FlatSlot:          { coll: true, kind: 'flat', V,
//                        outerRank, outerShape, getCell(idx) }
//     A flat shape-explicit Value; every axis is a loop axis.
//     `getCell(idx)` returns a scalar (or {re, im} for complex).
//
//   NestedSlot:        { coll: true, kind: 'nested', elements,
//                        outerRank, outerShape, getCell(idx) }
//     A nested JS array; JS-array nesting depth = outer rank. Per
//     cell `getCell(idx)` returns the innermost element WHOLE.
//     Rectangularity is validated; raggedness throws.

// ---------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------

// Classify one broadcast input. The single source of truth for
// "what's the outer-axis structure of this value, and how do I read
// one cell?". Used by `_broadcastApply`; intended to be the entry
// point for any future broadcast-shape consumer.
function classifyAxisStructure(v: any): any {
  // Table (spec §03): when a table is passed to broadcast, it is
  // traversed row-wise; each row is passed to the callable as a
  // record over the table's columns. Outer rank = 1 (only the row
  // axis is iterated); the cell is a record built on demand.
  if (v && v.__table__ === true && v.columns) {
    const nrows = v.nrows || 0;
    return {
      coll: true, kind: 'table', table: v,
      outerRank: 1, outerShape: [nrows],
      getCell(idx: number[]) {
        // Row-wise: each cell is the row record (table-valued columns become
        // nested records via valueLib.tableRow).
        return valueLib.tableRow(v, (nrows === 1) ? 0 : idx[0]);
      },
    };
  }
  if (valueLib.isValue(v)) {
    if (v.shape.length === 0) return { coll: false, val: v.data[0] };
    const V = v;
    const shape = V.shape;
    const rank = shape.length;
    // Engine-concepts §2.1 outerRank tag (C7). When set by the
    // constructor (`vector(...)` of Values), it splits the shape
    // into LOOP axes (the leading `outerRank`) and CELL axes (the
    // remaining trailing axes). Per cell the callable sees a Value
    // of the inner shape, not a scalar. Default `outerRank ===
    // shape.length` (flat tensor — every axis is a loop axis,
    // cell = scalar).
    const outerRank = (typeof V.outerRank === 'number') ? V.outerRank : rank;
    if (outerRank < rank) {
      // Nested-vector path: cell is a sub-Value of the inner shape.
      // Per cell we copy out the inner block into a fresh Float64Array
      // — the callable may otherwise see overlapping subarray views
      // and accidentally mutate.
      const innerShape = shape.slice(outerRank);
      const innerLen = innerShape.reduce((a: number, b: number) => a * b, 1);
      const outerShape = shape.slice(0, outerRank);
      // Row-major strides over the OUTER axes; size-1 axes get
      // stride 0 so they auto-repeat.
      const outerStrides = new Array(outerRank);
      let acc = innerLen;
      for (let a = outerRank - 1; a >= 0; a--) {
        outerStrides[a] = (outerShape[a] === 1) ? 0 : acc;
        acc *= outerShape[a];
      }
      return {
        coll: true, kind: 'nested', V,
        outerRank, outerShape,
        getCell(idx: number[]) {
          let off = 0;
          for (let a = 0; a < outerRank; a++) off += idx[a] * outerStrides[a];
          const sub = new Float64Array(innerLen);
          for (let k = 0; k < innerLen; k++) sub[k] = V.data[off + k];
          return { shape: innerShape.slice(), data: sub };
        },
      };
    }
    // Flat tensor: every axis is a loop axis.
    const strides = new Array(rank);
    let acc = 1;
    for (let a = rank - 1; a >= 0; a--) {
      strides[a] = (shape[a] === 1) ? 0 : acc;
      acc *= shape[a];
    }
    return {
      coll: true, kind: 'flat', V,
      outerRank: rank, outerShape: shape,
      getCell(idx: number[]) {
        let off = 0;
        for (let a = 0; a < rank; a++) off += idx[a] * strides[a];
        return (V.dtype === 'complex' && V.im)
          ? { re: V.data[off], im: V.im[off] }
          : V.data[off];
      },
    };
  }
  if (v != null && v.BYTES_PER_ELEMENT !== undefined
      && typeof v.length === 'number') {
    return classifyAxisStructure(valueLib.asValue(v));
  }
  if (Array.isArray(v)) {
    // Distinguish a JS array that lifts to a flat numeric Value (the
    // legacy `vector(1, 2, 3)` path → JS array of numbers) from a
    // "nested-vector" — a JS array whose elements are themselves
    // Values (or further-nested arrays) i.e. the `vector(C)`-style
    // path where C is a non-scalar.
    let allFlatScalars = true;
    for (let i = 0; i < v.length; i++) {
      const el = v[i];
      const t = typeof el;
      if (t === 'number' || t === 'boolean') continue;
      if (el && el.re !== undefined && el.im !== undefined
          && typeof el.re === 'number' && typeof el.im === 'number') continue;
      allFlatScalars = false;
      break;
    }
    if (allFlatScalars) {
      // Legacy flat-numeric JS array → coerce to a flat rank-1 Value.
      return classifyAxisStructure(valueLib.asValue(v));
    }
    // Nested-vector. May be rank-1 (`[C]`) or deeper (`[[C]]`,
    // `[[V1, V2], [V3, V4]]`, …).
    return classifyNestedJSArray(v);
  }
  if (typeof v === 'number' || typeof v === 'boolean'
      || (v && typeof v === 'object'
          && typeof v.re === 'number' && typeof v.im === 'number')) {
    return { coll: false, val: v };                       // scalar / complex
  }
  if (v && typeof v === 'object'
      && v.body !== undefined && Array.isArray(v.params)) {
    return { coll: false, val: v };                       // fn / kernel
  }
  if (v && typeof v === 'object') {
    throw new Error('broadcast: records and tuples are not allowed as '
      + 'broadcast inputs (spec §04)');
  }
  return { coll: false, val: v };
}

// Walk a (possibly multi-level) nested JS array, returning a slot
// descriptor with outerRank = nesting depth, outerShape = the per-
// level lengths, and getCell(idx) that indexes idx[0..outerRank-1]
// into the chain. The innermost element is whatever the engine
// produces (typically a Value); the broadcast loop passes it whole
// to the callable.
//
// Exported for the rare case where the input is known to be a
// nested-vector ahead of time and the broader classifier dispatch
// is unnecessary; ordinary callers should use `classifyAxisStructure`.
function classifyNestedJSArray(v: any[]): any {
  // Discover the nesting depth + per-level lengths by descending
  // along the [0] chain until we hit a non-JS-array.
  const depths: number[] = [];
  let cur: any = v;
  while (Array.isArray(cur)) {
    depths.push(cur.length);
    if (cur.length === 0) break;
    cur = cur[0];
  }
  // Rectangularity check at every level. Validate that at each
  // level, every sibling is a JS array of the same length as the
  // first, all the way down. Ragged ⇒ throw (matches the asValue /
  // vector(...) discipline).
  function _checkRect(node: any[], level: number) {
    if (level >= depths.length - 1) return;
    const expected = depths[level + 1];
    for (let i = 0; i < node.length; i++) {
      const ch = node[i];
      if (!Array.isArray(ch) || ch.length !== expected) {
        throw new Error('broadcast: nested-vector argument is ragged '
          + 'at depth ' + (level + 1) + ' (expected length ' + expected
          + ', got ' + (Array.isArray(ch) ? ch.length : 'non-array')
          + '); broadcast nested-vector args must be rectangular '
          + '(spec §04 — no ragged arrays)');
      }
      _checkRect(ch, level + 1);
    }
  }
  _checkRect(v, 0);
  const outerRank = depths.length;
  const outerShape = depths.slice();
  return {
    coll: true, kind: 'nested', elements: v,
    outerRank, outerShape,
    getCell(idx: number[]) {
      let node: any = v;
      for (let a = 0; a < outerRank; a++) {
        const sz = outerShape[a];
        const i = (sz === 1) ? 0 : idx[a];
        node = node[i];
      }
      return node;
    },
  };
}

// ---------------------------------------------------------------------
// Cell stacking
// ---------------------------------------------------------------------

// Stack a flat (row-major over the broadcast index) list of per-cell
// results into a single shape-explicit Value when possible. Returns
// null if the cells aren't stackable (mixed shapes, non-numeric
// content); callers fall back to a nested JS array.
//
// Two stack shapes are supported (gufunc-style):
//   - every cell is a scalar number / boolean ⇒ Value shape=bshape.
//   - every cell is a Value with the SAME inner shape S ⇒ Value
//     shape=[...bshape, ...S], row-major over the broadcast index
//     with each cell's data laid out contiguously.
function tryStackBroadcastCells(cells: any[], bshape: number[]): any {
  if (cells.length === 0) {
    return { shape: bshape.slice(), data: new Float64Array(0) };
  }
  // Try scalar-stack first.
  let allScalar = true;
  for (let i = 0; i < cells.length; i++) {
    const t = typeof cells[i];
    if (t !== 'number' && t !== 'boolean') { allScalar = false; break; }
  }
  if (allScalar) {
    const out = new Float64Array(cells.length);
    for (let i = 0; i < cells.length; i++) {
      out[i] = cells[i] === true ? 1 : cells[i] === false ? 0 : +cells[i];
    }
    return { shape: bshape.slice(), data: out };
  }
  // Try Value-stack (same inner shape).
  let innerShape: number[] | null = null;
  for (let i = 0; i < cells.length; i++) {
    if (!valueLib.isValue(cells[i])) return null;
    const s = cells[i].shape;
    if (innerShape === null) innerShape = s;
    else {
      if (s.length !== innerShape.length) return null;
      for (let a = 0; a < s.length; a++) if (s[a] !== innerShape[a]) return null;
    }
  }
  if (innerShape === null) return null;
  const innerLen = innerShape.reduce((a: number, b: number) => a * b, 1);
  const out = new Float64Array(cells.length * innerLen);
  for (let i = 0; i < cells.length; i++) {
    out.set(cells[i].data, i * innerLen);
  }
  return { shape: bshape.concat(innerShape), data: out };
}

module.exports = {
  classifyAxisStructure,
  classifyNestedJSArray,
  tryStackBroadcastCells,
};
