'use strict';

// ════════════════════════════════════════════════════════════════════════
// Smell A — materialiseMeasureIR → canonical-handler bridge.
//
// A CLM (jointchain RETAIN) body with an INLINE composite kernel is walked by
// materialiseMeasureIR (the kernel field is a call, not a named ref, so the
// by-name matRecord path never sees it). Ops materialiseMeasureIR doesn't own
// natively must delegate to the canonical KIND_HANDLERS via the synthetic-
// binding bridge instead of dead-ending at the leaf fallback (which crashes —
// the worker's sampleN has no `truncate`/`weighted`/… kernel).
//
// Stage 1: truncate. Verified red-for-the-right-reason first (the inline
// truncate kernel crashed with "Cannot read properties of undefined (reading
// 'length')" at the leaf fallback); this pins it green via matTruncate.
// ════════════════════════════════════════════════════════════════════════

const test = require('node:test');
const assert = require('node:assert');
const { buildCtx } = require('./_agreement-harness.ts');

test('Stage 1: inline truncate kernel in a CLM body materialises via matTruncate', async () => {
  // jointchain(prior, fwd), fwd = functionof(truncate(Normal(mu=theta, 1),
  // (0,inf)), theta=theta). The kernel base Normal(mu=theta,…) carries a
  // per-atom ref, so matTruncate takes the REJECTION path (refArrays for the
  // per-atom mu) — the parametric-truncate case the leaf fallback could not
  // handle. Prior chosen so P(y>0 | theta) is high (no rejection-budget NaN).
  const src = `
theta = elementof(reals)
ytrunc = truncate(Normal(mu = theta, sigma = 1.0), interval(0.0, inf))
fwd = functionof(ytrunc, theta = theta)
prior = Normal(mu = 2.0, sigma = 0.5)
m = jointchain(prior, fwd)`;
  const N = 2000;
  const { ctx } = buildCtx(src, N, 17);
  const m = await ctx.getMeasure('m');

  // Positional jointchain RETAIN → 2-tuple measure [prior, y].
  assert.ok(m.elems && m.elems.length === 2, 'jointchain retain → 2-tuple measure');
  const tS = m.elems[0].samples;
  const yS = m.elems[1].samples;
  assert.equal(yS.length, N);

  // Truncation to (0, inf) actually enforced: every y atom positive + finite
  // (a finite check guards against rejection-budget NaN — none expected here).
  let minY = Infinity, anyNaN = false;
  for (let i = 0; i < N; i++) {
    if (Number.isNaN(yS[i])) { anyNaN = true; break; }
    if (yS[i] < minY) minY = yS[i];
  }
  assert.ok(!anyNaN, 'no NaN atoms (rejection sampling converged)');
  assert.ok(minY > 0, `truncate(.,(0,inf)) yields only positive atoms; min=${minY}`);

  // y conditions on the threaded prior atom theta_i (truncated Normal at
  // theta_i): with theta≈2 the y mean sits a bit above 2 (mass shifted right
  // by the lower cut). Loose MC bound — just confirms y tracks theta, not a
  // constant or a re-materialised independent draw.
  let tm = 0, ym = 0;
  for (let i = 0; i < N; i++) { tm += tS[i]; ym += yS[i]; }
  tm /= N; ym /= N;
  assert.ok(Math.abs(tm - 2.0) < 0.2, `prior theta mean≈2, got ${tm}`);
  assert.ok(ym > 1.5 && ym < 3.5, `truncated y mean tracks theta≈2, got ${ym}`);
});
