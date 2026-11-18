'use strict';

// §07 Linear algebra. Unitary Householder QR returns a thin Q even when
// the input is tall. Schur iteration uses the same factorization on squares.
const V = require('./value.ts');

/** Apply I - beta v v† to the active rows and columns. */
function reflect(re: Float64Array, im: Float64Array, vr: Float64Array, vi: Float64Array,
  rows: number, columns: number, k: number, beta: number): void {
  for (let j = k; j < columns; j++) {
    let dr = 0, di = 0;
    for (let i = k; i < rows; i++) {
      const a = i * columns + j;
      dr += vr[i] * re[a] + vi[i] * im[a];
      di += vr[i] * im[a] - vi[i] * re[a];
    }
    dr *= beta; di *= beta;
    for (let i = k; i < rows; i++) {
      const a = i * columns + j;
      re[a] -= vr[i] * dr - vi[i] * di;
      im[a] -= vr[i] * di + vi[i] * dr;
    }
  }
}

function qr(input: any): any {
  V.requireMatrix(input, 'qr');
  const A = V._logicalDense(input), [m, n] = A.shape;
  if (m < n) throw new Error('qr: requires m >= n (got ' + m + 'x' + n + ')');
  const rr = new Float64Array(A.data), ri = new Float64Array(A.im || m * n);
  const vr = new Float64Array(m), vi = new Float64Array(m);
  const leadR = new Float64Array(n), leadI = new Float64Array(n), betas = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let scale = 0;
    for (let i = k; i < m; i++) scale = Math.max(scale, Math.abs(rr[i * n + k]), Math.abs(ri[i * n + k]));
    if (scale === 0) continue;
    let norm = 0;
    for (let i = k; i < m; i++) {
      vr[i] = rr[i * n + k] / scale; vi[i] = ri[i * n + k] / scale;
      norm = Math.hypot(norm, vr[i], vi[i]);
    }
    // Choose the phase of the leading entry to avoid cancellation in v.
    const first = Math.hypot(vr[k], vi[k]);
    const phaseR = first === 0 ? 1 : vr[k] / first, phaseI = first === 0 ? 0 : vi[k] / first;
    vr[k] += phaseR * norm; vi[k] += phaseI * norm;
    let square = 0;
    for (let i = k; i < m; i++) square += vr[i] * vr[i] + vi[i] * vi[i];
    const beta = 2 / square;
    reflect(rr, ri, vr, vi, m, n, k, beta);
    betas[k] = beta;
    leadR[k] = vr[k]; leadI[k] = vi[k];
    // The eliminated part of each column stores its reflector.
    for (let i = k + 1; i < m; i++) { rr[i * n + k] = vr[i]; ri[i * n + k] = vi[i]; }
  }
  const qr = new Float64Array(m * n), qi = new Float64Array(m * n);
  for (let i = 0; i < n; i++) qr[i * n + i] = 1;
  // Apply the saved reflectors in reverse to n identity columns.
  for (let k = n - 1; k >= 0; k--) {
    if (betas[k] === 0) continue;
    vr[k] = leadR[k]; vi[k] = leadI[k];
    for (let i = k + 1; i < m; i++) {
      vr[i] = rr[i * n + k]; vi[i] = ri[i * n + k];
      rr[i * n + k] = 0; ri[i * n + k] = 0;
    }
    reflect(qr, qi, vr, vi, m, n, k, betas[k]);
  }
  return { Q: V.complexValue(qr, qi, [m, n]),
    R: V.complexValue(m === n ? rr : rr.slice(0, n * n), m === n ? ri : ri.slice(0, n * n), [n, n]) };
}

module.exports = { qr };
