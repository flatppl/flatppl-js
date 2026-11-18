'use strict';

// §09 LU with complex magnitude pivoting. Packed row swaps include already
// computed L multipliers, so the permutation convention remains P*A=L*U.
// A zero remaining column is skipped, allowing singular factorizations too.
const V = require('./value.ts');

function factor(A: any) {
  A = V._logicalDense(A);
  const n = A.shape[0];
  if (A.shape.length !== 2 || n !== A.shape[1]) throw new Error('lu: A must be a square matrix');
  const re = new Float64Array(A.data), im = new Float64Array(A.im || n * n);
  const order = Array.from({ length: n }, (_, i) => i);
  let sign = 1;
  for (let k = 0; k < n; k++) {
    let pivot = k, largest = 0;
    for (let i = k; i < n; i++) {
      const size = Math.hypot(re[i * n + k], im[i * n + k]);
      if (size > largest) { largest = size; pivot = i; }
    }
    if (largest === 0) { sign = 0; continue; }
    if (pivot !== k) {
      for (let j = 0; j < n; j++) {
        const a = k * n + j, b = pivot * n + j;
        [re[a], re[b]] = [re[b], re[a]];
        [im[a], im[b]] = [im[b], im[a]];
      }
      [order[k], order[pivot]] = [order[pivot], order[k]];
      sign = -sign;
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
  return { n, re, im, order, sign };
}

function lu(A: any): any {
  const { n, re, im, order } = factor(A);
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

/** §07 Linear algebra. Row swaps contribute their parity to the determinant. */
function determinant(A: any): any {
  const { n, re, im, sign } = factor(A);
  if (sign === 0) return V.complexValue([0], [0], []);
  let real = sign, imag = 0;
  for (let i = 0; i < n; i++) {
    const k = i * (n + 1);
    const next = real * re[k] - imag * im[k];
    imag = real * im[k] + imag * re[k];
    real = next;
  }
  return V.complexValue([real], [imag], []);
}

/** §07 Linear algebra. Sum log magnitudes before forming the determinant. */
function logAbsDeterminant(A: any): number {
  const { n, re, im, sign } = factor(A);
  if (sign === 0) return -Infinity;
  let result = 0;
  for (let i = 0; i < n; i++) {
    const k = i * (n + 1), scale = Math.max(Math.abs(re[k]), Math.abs(im[k]));
    result += Math.log(scale) + Math.log(Math.hypot(re[k] / scale, im[k] / scale));
  }
  return result;
}

/** §07 Linear algebra. Solve L U X = P B with the packed factors. */
function solveFactors({ n, re, im, order, sign }: ReturnType<typeof factor>, b: any): any {
  if (sign === 0) throw new Error('linsolve: matrix is singular');
  if (b.shape[0] !== n || (b.shape.length !== 1 && b.shape.length !== 2)) {
    throw new Error('linsolve: right operand must be a vector or matrix with ' + n + ' rows');
  }
  const columns = b.shape[1] ?? 1;
  const xr = new Float64Array(n * columns), xi = new Float64Array(n * columns);
  for (let i = 0; i < n; i++) {
    const source = order[i] * columns;
    xr.set(b.data.subarray(source, source + columns), i * columns);
    if (b.im) xi.set(b.im.subarray(source, source + columns), i * columns);
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) subtractFactorRow(i, j);
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) subtractFactorRow(i, j);
    const k = i * (n + 1), scale = Math.max(Math.abs(re[k]), Math.abs(im[k]));
    const dr = re[k] / scale, di = im[k] / scale;
    const den = dr * dr + di * di;
    for (let c = 0; c < columns; c++) {
      const dest = i * columns + c, a = xr[dest] / scale, b = xi[dest] / scale;
      xr[dest] = (a * dr + b * di) / den;
      xi[dest] = (b * dr - a * di) / den;
    }
  }
  return V.complexValue(xr, xi, b.shape);

  function subtractFactorRow(i: number, j: number): void {
    const k = i * n + j;
    for (let c = 0; c < columns; c++) {
      const dest = i * columns + c, source = j * columns + c;
      xr[dest] -= re[k] * xr[source] - im[k] * xi[source];
      xi[dest] -= re[k] * xi[source] + im[k] * xr[source];
    }
  }
}

function solve(A: any, b: any): any {
  return solveFactors(factor(A), V._logicalDense(b));
}

function inverse(A: any): any {
  const factors = factor(A), { n } = factors;
  if (factors.sign === 0) throw new Error('inv: matrix is singular');
  const identity = new Float64Array(n * n);
  for (let i = 0; i < n; i++) identity[i * (n + 1)] = 1;
  return solveFactors(factors, { shape: [n, n], data: identity });
}

module.exports = { lu, determinant, logAbsDeterminant, solve, inverse };
