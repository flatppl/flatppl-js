'use strict';
// packages/engine/test/slice-kernel.test.ts
// Unit test for the slice kernel driven directly against closed-form targets
// (a mock ModelView exposing logPosterior + initFromPrior). Oracle: analytic
// Gaussian moments; known bimodal mode split; the eval-budget cap.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeSliceKernel } = require('../slice-kernel.ts');

// Deterministic LCG prng.
function lcg(seed: number) { let s = seed >>> 0; return () => { s = (1103515245 * s + 12345) & 0x7fffffff; return s / 0x7fffffff; }; }

function mockMv(dim: number, logPosterior: (y: Float64Array) => number, std: number[]) {
  // initFromPrior seeds the slice widths from the pool's per-coord std, so the
  // pool must carry real spread: ±std alternating gives mean 0, std exactly std[d].
  return { dim, logPosterior, initFromPrior: (n: number) => Array.from({ length: n }, (_v, p) => { const a = new Float64Array(dim); for (let d = 0; d < dim; d++) a[d] = (p % 2 === 0 ? std[d] : -std[d]); return a; }) };
}
function runChain(mv: any, dim: number, nsweep: number, seed: number, burn: number) {
  const k = makeSliceKernel();
  const prng = lcg(seed);
  const A = k.init(1, dim, {}, mv);
  const ens = [new Float64Array(dim)];
  const logp = new Float64Array([mv.logPosterior(ens[0])]);
  const out: Float64Array[] = [];
  for (let it = 0; it < nsweep; it++) { k.step(ens, logp, mv, prng, A, 'sample'); if (it >= burn) out.push(Float64Array.from(ens[0])); }
  return out;
}
function stats(xs: number[]) { let m = 0; for (const x of xs) m += x; m /= xs.length; let v = 0; for (const x of xs) v += (x - m) ** 2; return { m, sd: Math.sqrt(v / xs.length) }; }

test('slice recovers a 1-D standard normal', () => {
  const mv = mockMv(1, (y) => -0.5 * y[0] * y[0], [1]);
  const st = stats(runChain(mv, 1, 20000, 1, 2000).map((s) => s[0]));
  assert.ok(Math.abs(st.m) < 0.1, `mean ${st.m.toFixed(3)} ~ 0`);
  assert.ok(Math.abs(st.sd - 1) < 0.1, `sd ${st.sd.toFixed(3)} ~ 1`);
});

test('slice recovers a tight off-origin N(5, 0.3) (stepping-out reaches it)', () => {
  const mv = mockMv(1, (y) => -0.5 * ((y[0] - 5) / 0.3) ** 2, [1]);
  const st = stats(runChain(mv, 1, 20000, 2, 2000).map((s) => s[0]));
  assert.ok(Math.abs(st.m - 5) < 0.1, `mean ${st.m.toFixed(3)} ~ 5`);
  assert.ok(Math.abs(st.sd - 0.3) < 0.05, `sd ${st.sd.toFixed(3)} ~ 0.3`);
});

test('slice crosses the valley of a bimodal target (both modes)', () => {
  const mv = mockMv(1, (y) => { const a = -0.5 * ((y[0] + 4) / 0.7) ** 2, b = -0.5 * ((y[0] - 4) / 0.7) ** 2; return Math.log(0.5 * Math.exp(a) + 0.5 * Math.exp(b)); }, [5]);
  const xs = runChain(mv, 1, 40000, 3, 4000).map((s) => s[0]);
  const negFrac = xs.filter((x) => x < 0).length / xs.length;
  assert.ok(negFrac > 0.35 && negFrac < 0.65, `mode balance ${negFrac.toFixed(3)} (both modes visited)`);
});

test('slice step reports a finite bounded eval budget on a flat density', () => {
  const mv = mockMv(1, () => 0, [1]);
  const k = makeSliceKernel({ m: 20 });
  const A = k.init(1, 1, {}, mv);
  const ens = [new Float64Array(1)]; const logp = new Float64Array([0]);
  const r = k.step(ens, logp, mv, lcg(4), A, 'sample');
  assert.ok(r.proposals > 0 && Number.isFinite(r.proposals), 'evals finite');
  assert.ok(r.proposals < 200, `evals ${r.proposals} bounded by the cap, no runaway`);
});
