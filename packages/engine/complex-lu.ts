'use strict';

// §09 LU with complex magnitude pivoting. Packed row swaps include already
// computed L multipliers, so the permutation convention remains P*A=L*U.
// A zero remaining column is skipped, allowing singular factorizations too.
const V = require('./value.ts');

function lu(A: any): any {
  A = V._logicalDense(A);
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('lu: A must be a square matrix');
  const re = new Float64Array(A.data), im = new Float64Array(A.im);
  const order = Array.from({ length: n }, (_, i) => i);
  for (let k = 0; k < n; k++) {
    let pivot = k, largest = 0;
    for (let i = k; i < n; i++) {
      const size = Math.hypot(re[i * n + k], im[i * n + k]);
      if (size > largest) { largest = size; pivot = i; }
    }
    if (largest === 0) continue;
    if (pivot !== k) {
      for (let j = 0; j < n; j++) {
        const a = k * n + j, b = pivot * n + j;
        [re[a], re[b]] = [re[b], re[a]];
        [im[a], im[b]] = [im[b], im[a]];
      }
      [order[k], order[pivot]] = [order[pivot], order[k]];
    }
    const p = k * n + k, scale = Math.max(Math.abs(re[p]), Math.abs(im[p]));
    const pr = re[p] / scale, pi = im[p] / scale, denominator = pr * pr + pi * pi;
    for (let i = k + 1; i < n; i++) {
      const index = i * n + k, ar = re[index] / scale, ai = im[index] / scale;
      // Pivot magnitude bounds the numerator by sqrt(2), so this scaled
      // complex division cannot overflow merely by squaring the pivot.
      const fr = (ar * pr + ai * pi) / denominator;
      const fi = (ai * pr - ar * pi) / denominator;
      re[index] = fr; im[index] = fi;
      for (let j = k + 1; j < n; j++) {
        const from = k * n + j, to = i * n + j;
        re[to] -= fr * re[from] - fi * im[from];
        im[to] -= fr * im[from] + fi * re[from];
      }
    }
  }
  const lr = new Float64Array(n * n), li = new Float64Array(n * n);
  const ur = new Float64Array(n * n), ui = new Float64Array(n * n), perm = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    lr[i * n + i] = 1; perm[i * n + order[i]] = 1;
    for (let j = 0; j < n; j++) {
      const index = i * n + j;
      if (i > j) { lr[index] = re[index]; li[index] = im[index]; }
      else { ur[index] = re[index]; ui[index] = im[index]; }
    }
  }
  return { P: { shape: [n, n], data: perm }, L: V.complexValue(lr, li, [n, n]),
    U: V.complexValue(ur, ui, [n, n]) };
}

module.exports = { lu };
