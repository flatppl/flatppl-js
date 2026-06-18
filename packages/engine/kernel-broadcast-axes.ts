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

module.exports = { resolveBroadcastAxes, coordToOffset, cellToCoord };
