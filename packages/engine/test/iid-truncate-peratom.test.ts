'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./density/regression-baseline.test.ts');

// Regression: sampling `iid(truncate(Dist, S), k)` with PER-ATOM parameters
// (a param drawn from an upstream column, not a literal). The iid-truncate leaf
// path used to request count = N*k from the worker's truncateSampleN while
// passing length-N per-atom param columns, so positions ≥ N read undefined
// params → ~80% of draws came back NaN (only the first N/k of the N*k slots
// were finite). Fixed by giving truncateSampleN a `repeat` axis (like sampleN),
// so per-atom params are indexed by atom and shared across the atom's k iid
// replicates.
//
// Oracle: scipy.stats.truncnorm(a=(0-mu)/1, b=inf, loc=mu, scale=1).mean().
//   mu = 0.5 → 1.00916 ; std 0.69726.

function mean(a: number[]) { let s = 0; for (const v of a) s += v; return s / a.length; }
function std(a: number[]) { const m = mean(a); let s = 0; for (const v of a) s += (v - m) * (v - m); return Math.sqrt(s / a.length); }

test('iid(truncate(Normal(mu,1),[0,inf)), k) with per-atom mu: every draw finite', async () => {
  const src = `
mu ~ Normal(0.5, 1.0)
z ~ iid(truncate(Normal(mu, 1.0), interval(0.0, inf)), 5)
out = lawof(z)
`;
  const { ctx } = ctxFor(src, 4000);
  const m = await ctx.getMeasure('out');
  const s = Array.from(m.samples) as number[];
  const finite = s.filter((v) => Number.isFinite(v));
  // Regression guard: the bug left ~80% NaN. A few genuine tail atoms (very
  // negative mu → tiny truncated mass → rejection budget spent) may still NaN,
  // but the overwhelming majority must be finite now.
  assert.ok(finite.length > 0.97 * s.length, `finite ${finite.length}/${s.length} (was ~20% before the fix)`);
  assert.ok(finite.every((v) => v >= 0), 'all finite draws respect the truncation support [0, inf)');
});

test('per-atom iid-truncate matches the scipy truncnorm mean (oracle)', async () => {
  // Tight prior pins mu ≈ 0.5 so the per-atom (refArray) path is scored against
  // the closed-form truncnorm(loc=0.5) moments.
  const src = `
mu ~ Normal(0.5, 0.0005)
z ~ iid(truncate(Normal(mu, 1.0), interval(0.0, inf)), 5)
out = lawof(z)
`;
  const { ctx } = ctxFor(src, 8000);
  const s = Array.from((await ctx.getMeasure('out')).samples) as number[];
  assert.ok(s.every((v) => Number.isFinite(v)), 'all finite');
  assert.ok(Math.abs(mean(s) - 1.00916) < 0.03, `predictive mean ${mean(s).toFixed(4)} vs oracle 1.00916`);
  assert.ok(Math.abs(std(s) - 0.69726) < 0.03, `predictive std ${std(s).toFixed(4)} vs oracle 0.69726`);
});

test('per-atom iid-truncate conditions each atom’s replicates on its own param (atom-major)', async () => {
  // mu ~ Normal(10, 2): truncation at 0 is negligible, so each atom's k
  // replicates are ~Normal(mu_atom, 1). If the params are indexed atom-major,
  // the per-atom replicate-mean tracks mu_atom (theoretical corr = 2/sqrt(2^2 +
  // 1/5) ≈ 0.976). A broken (scrambled / dropped) param mapping destroys it.
  const N = 4000, K = 5;
  const src = `
mu ~ Normal(10.0, 2.0)
z ~ iid(truncate(Normal(mu, 1.0), interval(0.0, inf)), ${K})
out = lawof(z)
`;
  const { ctx } = ctxFor(src, N);
  const mus = Array.from((await ctx.getMeasure('mu')).samples) as number[];
  const zs = Array.from((await ctx.getMeasure('out')).samples) as number[];
  assert.ok(zs.every((v) => Number.isFinite(v)), 'all finite');
  const repMean = [];
  for (let a = 0; a < N; a++) { let s = 0; for (let r = 0; r < K; r++) s += zs[a * K + r]; repMean.push(s / K); }
  const mm = mean(mus), rm = mean(repMean);
  let cov = 0, vm = 0, vr = 0;
  for (let a = 0; a < N; a++) { cov += (mus[a] - mm) * (repMean[a] - rm); vm += (mus[a] - mm) ** 2; vr += (repMean[a] - rm) ** 2; }
  const corr = cov / Math.sqrt(vm * vr);
  assert.ok(corr > 0.9, `atom-major conditioning corr(mu, replicate-mean) = ${corr.toFixed(4)} (expect ~0.976)`);
});
