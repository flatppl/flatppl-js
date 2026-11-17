'use strict';

// Unitary Householder QR for the square complex blocks used by Schur iteration.
// H=I-2*v*vᴴ/(vᴴv) is Hermitian and unitary. Choosing the phase of the first
// entry avoids cancellation in v; R's diagonal may be complex (§09 needs no
// positive-diagonal convention). Each reflector scales its column first.
const V = require('./value.ts');

function qr(input: any): any {
  const A = V._logicalDense(input), n = A.shape[0];
  if (A.shape.length !== 2 || A.shape[1] !== n) throw new Error('Complex QR requires a square matrix');
  const rr = new Float64Array(A.data), ri = new Float64Array(A.im || n * n);
  const qr = new Float64Array(n * n), qi = new Float64Array(n * n);
  for (let i = 0; i < n; i++) qr[i * n + i] = 1;
  for (let k = 0; k < n; k++) {
    let scale = 0;
    for (let i = k; i < n; i++) scale = Math.max(scale, Math.abs(rr[i * n + k]), Math.abs(ri[i * n + k]));
    if (scale === 0) continue;
    const vr = new Float64Array(n - k), vi = new Float64Array(n - k);
    let norm = 0;
    for (let i = k; i < n; i++) {
      vr[i - k] = rr[i * n + k] / scale; vi[i - k] = ri[i * n + k] / scale;
      norm = Math.hypot(norm, vr[i - k], vi[i - k]);
    }
    const first = Math.hypot(vr[0], vi[0]);
    const phaseR = first === 0 ? 1 : vr[0] / first, phaseI = first === 0 ? 0 : vi[0] / first;
    vr[0] += phaseR * norm; vi[0] += phaseI * norm;
    let square = 0;
    for (let i = 0; i < vr.length; i++) square += vr[i] * vr[i] + vi[i] * vi[i];
    const beta = 2 / square;
    // Left apply H to R, then right apply H to the accumulated Q.
    for (let j = k; j < n; j++) {
      let dr = 0, di = 0;
      for (let i = k; i < n; i++) {
        const a = i * n + j, v = i - k;
        dr += vr[v] * rr[a] + vi[v] * ri[a];
        di += vr[v] * ri[a] - vi[v] * rr[a];
      }
      dr *= beta; di *= beta;
      for (let i = k; i < n; i++) {
        const a = i * n + j, v = i - k;
        rr[a] -= vr[v] * dr - vi[v] * di;
        ri[a] -= vr[v] * di + vi[v] * dr;
      }
    }
    for (let i = 0; i < n; i++) {
      let dr = 0, di = 0;
      for (let j = k; j < n; j++) {
        const a = i * n + j, v = j - k;
        dr += qr[a] * vr[v] - qi[a] * vi[v];
        di += qr[a] * vi[v] + qi[a] * vr[v];
      }
      dr *= beta; di *= beta;
      for (let j = k; j < n; j++) {
        const a = i * n + j, v = j - k;
        qr[a] -= dr * vr[v] + di * vi[v];
        qi[a] -= di * vr[v] - dr * vi[v];
      }
    }
    for (let i = k + 1; i < n; i++) { rr[i * n + k] = 0; ri[i * n + k] = 0; }
  }
  return { Q: V.complexValue(qr, qi, [n, n]), R: V.complexValue(rr, ri, [n, n]) };
}

module.exports = { qr };
