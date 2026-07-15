'use strict';

// =====================================================================
// broadcast-truncate-kernel-density.test.ts — Buffy #73(4)
// =====================================================================
//
// A per-row `normalize(truncate(...))` KERNEL BROADCAST used to drop the
// truncate normalizer, scoring the UN-truncated density. Concretely
// `energy = p -> normalize(truncate(Normal(p,1.0), interval(-1.0,2.0)))`
// broadcast as `E ~ energy.(w)` scored Σ logpdf(Normal(w_i,1), E_i) instead
// of Σ logpdf(Truncated(Normal(w_i,1),-1,2), E_i) — it omitted the per-row
// −log Z_i where Z_i = ∫_{-1}^{2} Normal(w_i,1).
//
// Root cause: walkKernelBroadcastMeasureKernel scores each broadcast cell
// by calling walkAcc on the RAW body, bypassing the main-thread pre-pass
// (mat-density's resolveTruncateNormalizers) that every other scoring path
// gets. A bare normalize(truncate(...)) reaches walkNormalize's "legacy
// bare normalize: Z treated as 1" branch, which drops the normalizer.
//
// Oracle: Distributions.jl (verified out-of-band):
//   per-row Truncated(Normal(w_i,1), -1, 2), w=[0,0.5,1,-0.3,0.8],
//   E=[0.1,0.4,1.2,-0.2,0.9] → -3.6359665981852225

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const TOL = 1e-9;

async function scoreOf(src: string): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) {
    throw new Error('scoreOf: __score__ produced no data (measure shape unexpected)');
  }
  return s[0];
}

const W_73 = [0.0, 0.5, 1.0, -0.3, 0.8];
const E_73 = [0.1, 0.4, 1.2, -0.2, 0.9];
const ORACLE_73_4 = -3.6359665981852225;

const MODEL_73_4 = `
w = [${W_73.join(', ')}]
energy = p -> normalize(truncate(Normal(p, 1.0), interval(-1.0, 2.0)))
E ~ energy.(w)
`;

test('#73(4) per-row normalize(truncate) kernel broadcast scores (vectorized == hand-unrolled == oracle)', async () => {
  // Vectorized broadcast form.
  const vectorized = await scoreOf(MODEL_73_4 + `\n__score__ = logdensityof(E, [${E_73.join(', ')}])\n`);
  assert.ok(Number.isFinite(vectorized),
    `vectorized broadcast score must be finite; got ${vectorized}`);

  // Hand-unrolled: one normalize(truncate(...)) per row, each scored at its datum.
  let unrolled = MODEL_73_4;
  const parts: string[] = [];
  for (let j = 0; j < W_73.length; j++) {
    unrolled += `tk_${j} = normalize(truncate(Normal(${W_73[j]}, 1.0), interval(-1.0, 2.0)))\n`;
    parts.push(`logdensityof(tk_${j}, ${E_73[j]})`);
  }
  unrolled += `__score__ = ${parts.join(' + ')}\n`;
  const unrolledScore = await scoreOf(unrolled);

  assert.ok(Math.abs(vectorized - unrolledScore) <= TOL,
    `vectorized ${vectorized} == hand-unrolled ${unrolledScore} (Δ ${Math.abs(vectorized - unrolledScore)})`);
  assert.ok(Math.abs(vectorized - ORACLE_73_4) <= 1e-9,
    `vectorized ${vectorized} == Distributions.jl truncated-Normal oracle ${ORACLE_73_4} (Δ ${Math.abs(vectorized - ORACLE_73_4)})`);
});
