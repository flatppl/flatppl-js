'use strict';
// Pure §04 broadcast-axis resolver. Shared single source of truth for the
// sampler executor (mat-broadcast) and density (walkBroadcast) so the two
// paths agree by construction. Operates on axis SIZES only — no value objects,
// no engine deps — so it is independently unit-testable.

// resolveBroadcastAxes(args): align each arg's collection axes per spec §04
// "Stochastic broadcast" — all args same axis-count; per axis equal-or-singular;
// a size-1 axis expands by repetition (stride 0). Returns the output axis
// vector + per-arg row-major strides over it. Throws a §04 error otherwise.
function resolveBroadcastAxes(args: Array<{ name: string; collectionAxes: number[] }>) {
  const ranked = args.filter((a) => a.collectionAxes.length > 0);
  if (ranked.length === 0) {
    // All scalar — a non-collection broadcast (K=1). One implicit cell.
    const strides: Record<string, number[]> = {};
    for (const a of args) strides[a.name] = [];
    return { axes: [], strides };
  }
  const m = ranked[0].collectionAxes.length;
  for (const a of ranked) {
    if (a.collectionAxes.length !== m) {
      throw new Error('broadcast: all collection arguments must have the same '
        + 'number of axes (spec §04); arg \'' + a.name + '\' has '
        + a.collectionAxes.length + ', expected ' + m);
    }
  }
  // Output size along each axis: the max (== the non-singular size). Validate
  // every arg is equal-or-singular there.
  const axes = new Array(m).fill(1);
  for (let k = 0; k < m; k++) {
    for (const a of ranked) {
      const s = a.collectionAxes[k];
      if (s !== 1) {
        if (axes[k] === 1) axes[k] = s;
        else if (axes[k] !== s) {
          throw new Error('broadcast: incompatible collection sizes on axis '
            + k + ' (' + axes[k] + ' vs ' + s + ') for arg \'' + a.name
            + '\' — sizes must be equal or 1 (spec §04)');
        }
      }
    }
  }
  // Row-major strides over `axes`, with stride 0 where the arg is singular (or
  // scalar, i.e. no collection axes) on that output axis.
  const strides: Record<string, number[]> = {};
  for (const a of args) {
    const ca = a.collectionAxes;
    const st = new Array(m).fill(0);
    if (ca.length === m) {
      let acc = 1;
      for (let k = m - 1; k >= 0; k--) {
        st[k] = (ca[k] === 1) ? 0 : acc;
        acc *= ca[k];
      }
    }
    strides[a.name] = st;
  }
  return { axes, strides };
}

function coordToOffset(coord: number[], strides: number[]) {
  let off = 0;
  for (let k = 0; k < strides.length; k++) off += coord[k] * strides[k];
  return off;
}

// Flat row-major cell index → coordinate over `axes`.
function cellToCoord(cell: number, axes: number[]) {
  const c = new Array(axes.length);
  let rem = cell;
  for (let k = axes.length - 1; k >= 0; k--) { c[k] = rem % axes[k]; rem = Math.floor(rem / axes[k]); }
  return c;
}

/** Select one existing §04 collection cell without consuming its inner axes. */
function cellAccessor(v: any, usesAtom: boolean, N: number): {
  axes: number[]; accessor: (i: number, j: number) => any;
} {
  const V = require('./value.ts');
  const batched = usesAtom && (V.isAtomBatched(v, N) || (Array.isArray(v) && v.length === N));
  const atom = (i: number) => (!batched ? v : V.isValue(v) ? V._sliceRow(v, i) : v[i]);
  const first = atom(0);
  const axesOf = (x: any): number[] => {
    if (x?.__table__ === true) return [x.nrows];
    if (V.isValue(x)) {
      const levels = V.levelAxesOf(x);
      return levels.length ? x.shape.slice(0, levels[0]) : x.shape.slice();
    }
    return Array.isArray(x) || x?.BYTES_PER_ELEMENT !== undefined ? [x.length] : [];
  };
  const axes = axesOf(first);
  if (axes.length === 0) return { axes, accessor: (i) => atom(i) };
  const cellOf = (x: any, j: number) => {
    if (x?.__table__ === true) return V.tableRow(x, j);
    if (Array.isArray(x) || x?.BYTES_PER_ELEMENT !== undefined) return x[j];
    const dense = V._logicalDense(x);
    const cellShape = dense.shape.slice(axes.length);
    const size = cellShape.reduce((a: number, b: number) => a * b, 1);
    if (cellShape.length === 0) return dense.data[j];
    const cell: any = { shape: cellShape, data: dense.data.subarray(j * size, (j + 1) * size) };
    return V.setLevels(cell, V.levelAxesOf(dense).slice(1));
  };
  if (!batched) return { axes, accessor: (_i, j) => cellOf(first, j) };
  return { axes, accessor: (i, j) => cellOf(atom(i), j) };
}

module.exports = { resolveBroadcastAxes, coordToOffset, cellToCoord, cellAccessor };
