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
  while (t < n - 1) {
    const pair = rhoAt(t) + rhoAt(t + 1);
    if (pair < 0) break;
    sumRho += pair;
    t += 2;
  }
  const tau = 1 + 2 * sumRho;
  return (m * n) / Math.max(tau, 1 / Math.log10(m * n));
}

module.exports = { splitRHat, essBulk };
