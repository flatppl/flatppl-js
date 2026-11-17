'use strict';

// §09 general right eigenvectors, through complex unitary Schur iteration.
// A shifted QR step applies the full similarity T←QᴴTQ and Z←ZQ. Deflation
// checks the entire active last row (the input need not be Hessenberg).
// Wilkinson shifts use the bottom 2×2 spectrum; exceptional shifts escape
// stalled symmetric cycles. See LAPACK Working Note153, §2, for this strategy.
const V = require('./value.ts');
const { qr } = require('./complex-qr.ts');
const EPS = Number.EPSILON;

function sqrtComplex(re: number, im: number): [number, number] {
  const size = Math.hypot(re, im);
  if (size === 0) return [0, 0];
  if (re >= 0) {
    const r = Math.sqrt((size + re) / 2);
    return [r, im / (2 * r)];
  }
  const i = (im < 0 ? -1 : 1) * Math.sqrt((size - re) / 2);
  return [Math.abs(im / (2 * i)), i];
}

/** Scale the discriminant before squaring, retaining small complex shifts. */
function shift(t: any, n: number, last: number): [number, number] {
  const indices = [(last - 1) * n + last - 1, (last - 1) * n + last,
    last * n + last - 1, last * n + last];
  let scale = 0;
  for (const i of indices) scale = Math.max(scale, Math.abs(t.data[i]), Math.abs(t.im[i]));
  if (scale === 0) return [0, 0];
  const [a, b, c, d] = indices;
  const ar = t.data[a] / scale, ai = t.im[a] / scale;
  const br = t.data[b] / scale, bi = t.im[b] / scale;
  const cr = t.data[c] / scale, ci = t.im[c] / scale;
  const dr = t.data[d] / scale, di = t.im[d] / scale;
  const xr = (ar - dr) / 2, xi = (ai - di) / 2;
  const [yr, yi] = sqrtComplex(xr * xr - xi * xi + br * cr - bi * ci,
    2 * xr * xi + br * ci + bi * cr);
  const mr = (ar + dr) / 2, mi = (ai + di) / 2;
  const sign = Math.hypot(mr + yr - dr, mi + yi - di) <= Math.hypot(mr - yr - dr, mi - yi - di) ? 1 : -1;
  return [scale * (mr + sign * yr), scale * (mi + sign * yi)];
}

/** Embed the active Q in identity, retaining couplings to deflated columns. */
function step(t: any, z: any, q: any, n: number, m: number): [any, any] {
  const sr = new Float64Array(t.data), si = new Float64Array(t.im);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    let re = 0, im = 0;
    for (let k = 0; k < m; k++) {
      const a = k * m + i, b = k * n + j;
      re += q.data[a] * t.data[b] + q.im[a] * t.im[b];
      im += q.data[a] * t.im[b] - q.im[a] * t.data[b];
    }
    sr[i * n + j] = re; si[i * n + j] = im;
  }
  const tr = new Float64Array(sr), ti = new Float64Array(si);
  const zr = new Float64Array(z.data), zi = new Float64Array(z.im);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    let re = 0, im = 0, vr = 0, vi = 0;
    for (let k = 0; k < m; k++) {
      const a = i * n + k, b = k * m + j;
      re += sr[a] * q.data[b] - si[a] * q.im[b];
      im += sr[a] * q.im[b] + si[a] * q.data[b];
      vr += z.data[a] * q.data[b] - z.im[a] * q.im[b];
      vi += z.data[a] * q.im[b] + z.im[a] * q.data[b];
    }
    tr[i * n + j] = re; ti[i * n + j] = im;
    zr[i * n + j] = vr; zi[i * n + j] = vi;
  }
  return [V.complexValue(tr, ti, [n, n]), V.complexValue(zr, zi, [n, n])];
}

/** Backsolve triangular Schur eigenvectors. A repeated eigenvalue may have
 * fewer independent vectors; a norm-scaled denominator perturbation returns
 * its limiting right eigenvectors without promising a nonexistent basis. */
function vectors(t: any, z: any, n: number): any {
  const vr = new Float64Array(n * n), vi = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const wr = new Float64Array(n), wi = new Float64Array(n); wr[j] = 1;
    const lr = t.data[j * n + j], li = t.im[j * n + j];
    for (let i = j - 1; i >= 0; i--) {
      let rr = 0, ri = 0, rowScale = Math.max(Math.hypot(t.data[i * n + i], t.im[i * n + i]), Math.hypot(lr, li));
      for (let k = i + 1; k <= j; k++) {
        const a = i * n + k;
        rr -= t.data[a] * wr[k] - t.im[a] * wi[k];
        ri -= t.data[a] * wi[k] + t.im[a] * wr[k];
        rowScale = Math.max(rowScale, Math.hypot(t.data[a], t.im[a]));
      }
      let dr = t.data[i * n + i] - lr, di = t.im[i * n + i] - li;
      const size = Math.hypot(dr, di), minimum = Math.max(Number.MIN_VALUE, 8 * EPS * rowScale);
      if (size < minimum) {
        const ratio = size === 0 ? 0 : minimum / size;
        dr = size === 0 ? minimum : dr * ratio; di *= ratio;
      }
      const scale = Math.max(Math.abs(dr), Math.abs(di)), ar = dr / scale, ai = di / scale;
      const denominator = ar * ar + ai * ai;
      wr[i] = (rr / scale * ar + ri / scale * ai) / denominator;
      wi[i] = (ri / scale * ar - rr / scale * ai) / denominator;
      let norm = 0;
      for (let k = i; k <= j; k++) norm = Math.hypot(norm, wr[k], wi[k]);
      for (let k = i; k <= j; k++) { wr[k] /= norm; wi[k] /= norm; }
    }
    let norm = 0;
    for (let i = 0; i < n; i++) {
      let re = 0, im = 0;
      for (let k = 0; k <= j; k++) {
        const a = i * n + k;
        re += z.data[a] * wr[k] - z.im[a] * wi[k];
        im += z.data[a] * wi[k] + z.im[a] * wr[k];
      }
      vr[i * n + j] = re; vi[i * n + j] = im; norm = Math.hypot(norm, re, im);
    }
    for (let i = 0; i < n; i++) { vr[i * n + j] /= norm; vi[i * n + j] /= norm; }
  }
  return V.complexValue(vr, vi, [n, n]);
}

function eigen(input: any): any {
  const a = V._logicalDense(input), n = a.shape[0];
  if (a.shape.length !== 2 || a.shape[1] !== n) throw new Error('eigen: A must be a square matrix');
  let t = V.complexValue(new Float64Array(a.data), new Float64Array(a.im || n * n), [n, n]);
  let z = V.complexValue(new Float64Array(n * n), new Float64Array(n * n), [n, n]);
  for (let i = 0; i < n; i++) z.data[i * n + i] = 1;
  let active = n, iterations = 0;
  while (active > 1) {
    const last = active - 1;
    let off = 0;
    for (let j = 0; j < last; j++) off = Math.max(off, Math.hypot(t.data[last * n + j], t.im[last * n + j]));
    const scale = Math.max(Math.hypot(t.data[last * n + last], t.im[last * n + last]),
      Math.hypot(t.data[(last - 1) * n + last - 1], t.im[(last - 1) * n + last - 1]));
    if (off === 0 || off <= 64 * EPS * scale) {
      for (let j = 0; j < last; j++) { t.data[last * n + j] = 0; t.im[last * n + j] = 0; }
      active--; iterations = 0; continue;
    }
    if (++iterations > 2000 * n) throw new Error('eigen: Schur iteration did not converge');
    let [muR, muI] = shift(t, n, last);
    if (iterations % 32 === 0) { muR = t.data[last * n + last] + .75 * off; muI = t.im[last * n + last] + .25 * off; }
    const br = new Float64Array(active * active), bi = new Float64Array(active * active);
    for (let i = 0; i < active; i++) for (let j = 0; j < active; j++) {
      br[i * active + j] = t.data[i * n + j] - (i === j ? muR : 0);
      bi[i * active + j] = t.im[i * n + j] - (i === j ? muI : 0);
    }
    const { Q } = qr(V.complexValue(br, bi, [active, active]));
    [t, z] = step(t, z, Q, n, active);
  }
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) { re[i] = t.data[i * n + i]; im[i] = t.im[i * n + i]; }
  const out = { values: V.complexValue(re, im, [n]), vectors: vectors(t, z, n) };
  // Backward-error certificate in scaled units, avoiding products of the
  // original matrix norm with epsilon and retaining both complex planes.
  let scale = 0;
  for (let i = 0; i < a.data.length; i++) scale = Math.max(scale, Math.abs(a.data[i]), Math.abs(a.im?.[i] || 0));
  if (scale > 0) {
    let norm = 0, error = 0;
    for (let i = 0; i < n; i++) {
      let row = 0;
      for (let k = 0; k < n; k++) row += Math.hypot(a.data[i * n + k] / scale, (a.im?.[i * n + k] || 0) / scale);
      norm = Math.max(norm, row);
      for (let j = 0; j < n; j++) {
        const v = out.vectors, vr = v.data[i * n + j], vi = v.im[i * n + j];
        let rr = -(re[j] / scale * vr - im[j] / scale * vi);
        let ri = -(re[j] / scale * vi + im[j] / scale * vr);
        for (let k = 0; k < n; k++) {
          const ar = a.data[i * n + k] / scale, ai = (a.im?.[i * n + k] || 0) / scale;
          rr += ar * v.data[k * n + j] - ai * v.im[k * n + j];
          ri += ar * v.im[k * n + j] + ai * v.data[k * n + j];
        }
        error = Math.max(error, Math.hypot(rr, ri));
      }
    }
    if (!Number.isFinite(error) || error > 128 * EPS * n * norm) {
      throw new Error('eigen: right-eigenvector residual did not converge');
    }
  }
  return out;
}

module.exports = { eigen };
