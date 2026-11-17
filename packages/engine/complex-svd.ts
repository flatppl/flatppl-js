'use strict';

// §09 tall complex SVD, using one-sided cyclic Jacobi rotations. Column norms
// and inner products are scaled separately, retaining tiny singular values
// beside large orthogonal columns. V accumulates the same unitary rotations.
// This is the column-rotation construction used by the Jacobi SVD family
// described in LAPACK Working Notes169/170; it never forms AᴴA.
const V = require('./value.ts');
const EPS = Number.EPSILON;

function columnNorm(re: Float64Array, im: Float64Array, m: number, n: number, j: number): number {
  let norm = 0;
  for (let i = 0; i < m; i++) norm = Math.hypot(norm, re[i * n + j], im[i * n + j]);
  return norm;
}

/** Keep a unit vector accurate when its norm rounds on the subnormal grid. */
function subnormalUnit(re: Float64Array, im: Float64Array, m: number, n: number, j: number): [number, number] {
  let scale = 0, norm = 0;
  for (let i = 0; i < m; i++) scale = Math.max(scale, Math.abs(re[i * n + j]), Math.abs(im[i * n + j]));
  for (let i = 0; i < m; i++) norm = Math.hypot(norm, re[i * n + j] / scale, im[i * n + j] / scale);
  return [scale, norm];
}

/** A unitary two-column rotation, with rho the phase of column pᴴ column q. */
function rotate(re: Float64Array, im: Float64Array, rows: number, cols: number,
    p: number, q: number, cs: number, sn: number, pr: number, pi: number): void {
  for (let i = 0; i < rows; i++) {
    const a = i * cols + p, b = i * cols + q;
    const ar = re[a], ai = im[a], br = re[b], bi = im[b];
    re[a] = cs * ar + sn * (pr * br + pi * bi);
    im[a] = cs * ai + sn * (pr * bi - pi * br);
    re[b] = -sn * (pr * ar - pi * ai) + cs * br;
    im[b] = -sn * (pr * ai + pi * ar) + cs * bi;
  }
}

function svd(input: any): any {
  const a = V._logicalDense(input), [m, n] = a.shape;
  if (m < n) throw new Error('Complex Jacobi SVD requires a tall matrix');
  const br = new Float64Array(a.data), bi = new Float64Array(a.im || m * n);
  const vr = new Float64Array(n * n), vi = new Float64Array(n * n);
  for (let i = 0; i < n; i++) vr[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let rotated = false;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const np = columnNorm(br, bi, m, n, p), nq = columnNorm(br, bi, m, n, q);
      if (np === 0 || nq === 0) continue;
      let ps = np, pu = 1, qs = nq, qu = 1;
      if (np < 2.2250738585072014e-308) [ps, pu] = subnormalUnit(br, bi, m, n, p);
      if (nq < 2.2250738585072014e-308) [qs, qu] = subnormalUnit(br, bi, m, n, q);
      let gr = 0, gi = 0;
      for (let i = 0; i < m; i++) {
        const ar = br[i * n + p] / ps / pu, ai = bi[i * n + p] / ps / pu;
        const cr = br[i * n + q] / qs / qu, ci = bi[i * n + q] / qs / qu;
        gr += ar * cr + ai * ci; gi += ar * ci - ai * cr;
      }
      const gamma = Math.hypot(gr, gi);
      if (gamma <= 32 * EPS) continue;
      const scale = Math.max(np, nq), x = np / scale, y = nq / scale;
      const cross = x * y * gamma, difference = x * x - y * y;
      const theta = difference / (2 * cross);
      const t = difference === 0 ? 1 : Math.abs(theta) > 1e150 ? cross / difference
        : Math.sign(theta) / (Math.abs(theta) + Math.hypot(theta, 1));
      if (t === 0) {
        // A rotation angle below the representable range can still change
        // a tiny column. Apply its limiting projection in column units;
        // the corresponding change to unitary V rounds to zero.
        const small = np < nq ? p : q, large = np < nq ? q : p;
        const ss = np < nq ? ps : qs, su = np < nq ? pu : qu;
        const ls = np < nq ? qs : ps, lu = np < nq ? qu : pu;
        const projectionI = np < nq ? -gi : gi;
        for (let i = 0; i < m; i++) {
          const ar = br[i * n + large] / ls / lu, ai = bi[i * n + large] / ls / lu;
          br[i * n + small] -= ss * (su * (ar * gr - ai * projectionI));
          bi[i * n + small] -= ss * (su * (ar * projectionI + ai * gr));
        }
        rotated = true; continue;
      }
      const cs = 1 / Math.hypot(1, t), sn = cs * t;
      rotate(br, bi, m, n, p, q, cs, sn, gr / gamma, gi / gamma);
      rotate(vr, vi, n, n, p, q, cs, sn, gr / gamma, gi / gamma);
      rotated = true;
    }
    if (!rotated) break;
  }
  const norms = Array.from({ length: n }, (_, j) => columnNorm(br, bi, m, n, j));
  const order = Array.from({ length: n }, (_, j) => j).sort((p, q) => norms[q] - norms[p]);
  const singular = Float64Array.from(order, j => norms[j]);
  const ur = new Float64Array(m * n), ui = new Float64Array(m * n);
  const rr = new Float64Array(n * n), ri = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const old = order[j], norm = singular[j];
    let scale = norm, unit = 1;
    if (norm > 0 && norm < 2.2250738585072014e-308) [scale, unit] = subnormalUnit(br, bi, m, n, old);
    if (norm > 0) for (let i = 0; i < m; i++) {
      ur[i * n + j] = br[i * n + old] / scale / unit; ui[i * n + j] = bi[i * n + old] / scale / unit;
    }
    for (let i = 0; i < n; i++) { rr[i * n + j] = vr[i * n + old]; ri[i * n + j] = vi[i * n + old]; }
  }
  // Fill only exact null columns. Sorted nonzero columns are all present
  // before coordinate selection, so null vectors cannot duplicate them.
  for (let j = 0; j < n; j++) {
    if (singular[j] !== 0) continue;
    let best = 0, residual = -Infinity;
    for (let i = 0; i < m; i++) {
      let r = 1;
      for (let k = 0; k < j; k++) r -= ur[i * n + k] ** 2 + ui[i * n + k] ** 2;
      if (r > residual) { residual = r; best = i; }
    }
    ur[best * n + j] = 1;
    for (let pass = 0; pass < 2; pass++) for (let k = 0; k < j; k++) {
      let dr = 0, di = 0;
      for (let i = 0; i < m; i++) {
        const a = i * n + k, b = i * n + j;
        dr += ur[a] * ur[b] + ui[a] * ui[b]; di += ur[a] * ui[b] - ui[a] * ur[b];
      }
      for (let i = 0; i < m; i++) {
        const a = i * n + k, b = i * n + j;
        ur[b] -= ur[a] * dr - ui[a] * di; ui[b] -= ur[a] * di + ui[a] * dr;
      }
    }
    const norm = columnNorm(ur, ui, m, n, j);
    for (let i = 0; i < m; i++) { ur[i * n + j] /= norm; ui[i * n + j] /= norm; }
  }
  // A normalized Gram certificate detects nonconvergence independently of
  // the sweep count; the same guard includes zero-column basis completion.
  let error = 0;
  const factors: Array<[Float64Array, Float64Array, number]> = [[ur, ui, m], [rr, ri, n]];
  for (const [re, im, rows] of factors) {
    for (let p = 0; p < n; p++) for (let q = p; q < n; q++) {
      let gr = p === q ? -1 : 0, gi = 0;
      for (let i = 0; i < rows; i++) {
        const a = i * n + p, b = i * n + q;
        gr += re[a] * re[b] + im[a] * im[b]; gi += re[a] * im[b] - im[a] * re[b];
      }
      error = Math.max(error, Math.hypot(gr, gi));
    }
  }
  if (!Number.isFinite(error) || error > 128 * EPS * Math.max(m, n)) throw new Error('svd: unitary factors did not converge');
  return { U: V.complexValue(ur, ui, [m, n]), S: { shape: [n], data: singular },
    V: V.complexValue(rr, ri, [n, n]) };
}

module.exports = { svd };
