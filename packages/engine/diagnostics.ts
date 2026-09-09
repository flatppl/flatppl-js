'use strict';

function mean(a: Float64Array) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
function variance(a: Float64Array) { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; } return s / (a.length - 1); }

// Split each chain in half, then standard between/within R-hat (Gelman et al.).
function splitRHat(chains: Float64Array[]): number {
  const halves: Float64Array[] = [];
  for (const c of chains) {
    const h = c.length >> 1;
    halves.push(c.subarray(0, h));
    halves.push(c.subarray(h, 2 * h));
  }
  const m = halves.length, n = halves[0].length;
  const means = halves.map(mean);
  const grand = means.reduce((p, x) => p + x, 0) / m;
  let B = 0; for (const mu of means) B += (mu - grand) ** 2; B = (n / (m - 1)) * B;
  let W = 0; for (const h of halves) W += variance(h); W /= m;
  const varHat = ((n - 1) / n) * W + B / n;
  return Math.sqrt(varHat / W);
}

// In-place radix-2 DFT. Used only for the zero-padded autocorrelation below.
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const r = re[i], v = im[i];
      re[i] = re[j]; im[i] = im[j]; re[j] = r; im[j] = v;
    }
  }
  for (let width = 2; width <= n; width *= 2) {
    const half = width / 2, angle = -2 * Math.PI / width;
    const wr = Math.cos(angle), wi = Math.sin(angle);
    for (let start = 0; start < n; start += width) {
      let ur = 1, ui = 0;
      for (let j = start; j < start + half; j++) {
        const k = j + half;
        const tr = ur * re[k] - ui * im[k], ti = ur * im[k] + ui * re[k];
        re[k] = re[j] - tr; im[k] = im[j] - ti;
        re[j] += tr; im[j] += ti;
        const next = ur * wr - ui * wi;
        ui = ur * wi + ui * wr; ur = next;
      }
    }
  }
}

// Wiener-Khinchin: the inverse DFT of |DFT(x)|² gives lag sums. Pad to
// at least 2*n-1 so these are linear, not circular, autocovariances. A real
// power spectrum has the same real forward/inverse transform up to 1/size.
// Scale each centered chain before transforming to avoid squaring overflow.
function meanAutocovariances(chains: Float64Array[], means: number[], n: number): Float64Array {
  let size = 1;
  while (size < 2 * n - 1) size *= 2;
  const re = new Float64Array(size), im = new Float64Array(size);
  const acov = new Float64Array(n);
  for (let c = 0; c < chains.length; c++) {
    const chain = chains[c], mu = means[c];
    let scale = 0;
    for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(chain[i] - mu));
    if (scale === 0) continue;
    re.fill(0); im.fill(0);
    for (let i = 0; i < n; i++) re[i] = (chain[i] - mu) / scale;
    fft(re, im);
    for (let i = 0; i < size; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; }
    fft(re, im);
    for (let t = 0; t < n; t++) acov[t] += (re[t] / size / n) * scale * scale;
  }
  for (let t = 0; t < n; t++) acov[t] /= chains.length;
  return acov;
}

// Bulk ESS via the multi-chain autocorrelation estimator (Vehtari et al. 2021),
// truncated at the first negative pair of autocorrelations (Geyer).
function essBulk(chains: Float64Array[]): number {
  const m = chains.length, n = chains[0].length;
  const chainMeans = chains.map(mean);
  const grand = chainMeans.reduce((p, x) => p + x, 0) / m;
  let B = 0; for (const mu of chainMeans) B += (mu - grand) ** 2; B = (n / (m - 1)) * B;
  let W = 0; for (const c of chains) W += variance(c); W /= m;
  const varPlus = ((n - 1) / n) * W + B / n;
  if (varPlus <= 0) return m * n;

  // mean autocorrelation across chains at lag t
  function rhoAt(t: number): number {
    let acov = 0;
    for (let ci = 0; ci < m; ci++) {
      const c = chains[ci], mu = chainMeans[ci];
      let s = 0; for (let i = 0; i < n - t; i++) s += (c[i] - mu) * (c[i + t] - mu);
      acov += s / n;
    }
    acov /= m;
    return 1 - (W - acov) / varPlus;
  }

  let sumRho = 0, t = 1;
  const directLimit = n >= 1024 ? 65 : n - 1;
  let stopped = false;
  // Keep the early-exit path free of FFT dispatch inside each lag pair.
  while (t < directLimit) {
    const pair = rhoAt(t) + rhoAt(t + 1);
    if (pair < 0) { stopped = true; break; }
    sumRho += pair;
    t += 2;
  }
  if (!stopped && t < n - 1) {
    // Only pay O(m*n*log n) setup after 64 direct lags, avoiding the
    // O(m*n²) tail for long, sticky chains.
    // Keep direct rounding in the subnormal-variance regime: FFT scaling
    // changes where products underflow, even when both variances are finite.
    const acov = W >= 2 ** -1022 && varPlus >= 2 ** -1022
      && Number.isFinite(varPlus) && Number.isFinite(W)
      ? meanAutocovariances(chains, chainMeans, n) : undefined;
    while (t < n - 1) {
      let pair = acov
        ? (1 - (W - acov[t]) / varPlus) + (1 - (W - acov[t + 1]) / varPlus)
        : rhoAt(t) + rhoAt(t + 1);
      // Use direct sums at the truncation boundary so transform roundoff near
      // zero does not choose whether to include the next lag pair.
      if (acov && pair < 1e-10) pair = rhoAt(t) + rhoAt(t + 1);
      if (pair < 0) break;
      sumRho += pair;
      t += 2;
    }
  }
  const tau = 1 + 2 * sumRho;
  return (m * n) / Math.max(tau, 1 / Math.log10(m * n));
}

module.exports = { splitRHat, essBulk };
