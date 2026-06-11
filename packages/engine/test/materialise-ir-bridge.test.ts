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

test('Stage 2: inline weighted kernel in a CLM body materialises via matWeighted', async () => {
  // fwd = functionof(weighted(2.0, Normal(mu=theta, 1)), theta=theta). Before
  // the bridge the leaf fallback crashed (no `weighted` sampler kernel). Now
  // routes to matWeighted: a constant weight is a log-mass shift log(2) on the
  // base; the atoms are the (unchanged) base Normal draws. Per the composite-
  // measure invariant the mass lives at the OUTER measure (sub-fields are pure
  // SoA), so the log-2 shift surfaces on m.logTotalmass.
  const src = `
theta = elementof(reals)
yw = weighted(2.0, Normal(mu = theta, sigma = 1.0))
fwd = functionof(yw, theta = theta)
prior = Normal(mu = 0.0, sigma = 1.0)
m = jointchain(prior, fwd)`;
  const N = 1500;
  const { ctx } = buildCtx(src, N, 23);
  const m = await ctx.getMeasure('m');
  assert.ok(m.elems && m.elems.length === 2, 'jointchain retain → 2-tuple measure');
  assert.ok(Math.abs(m.logTotalmass - Math.log(2)) < 1e-9,
    `weighted(2, ·) shifts log-total-mass by log 2; got ${m.logTotalmass}`);
  const yS = m.elems[1].samples;
  let anyNaN = false;
  for (let i = 0; i < N; i++) if (Number.isNaN(yS[i])) { anyNaN = true; break; }
  assert.ok(!anyNaN, 'no NaN atoms');
});

test('Stage 3: inline normalize(truncate) kernel in a CLM body materialises via matNormalize', async () => {
  // fwd = functionof(normalize(truncate(Normal(mu=theta,1),(0,inf))), theta=theta).
  // normalize wraps truncate (itself bridged): the base is composite, so the
  // bridge's register() recurses (normalize→truncate→leaf). matNormalize
  // renormalises → outer log-total-mass 0; truncation still enforced (y>0).
  const src = `
theta = elementof(reals)
yn = normalize(truncate(Normal(mu = theta, sigma = 1.0), interval(0.0, inf)))
fwd = functionof(yn, theta = theta)
prior = Normal(mu = 2.0, sigma = 0.5)
m = jointchain(prior, fwd)`;
  const N = 1500;
  const { ctx } = buildCtx(src, N, 29);
  const m = await ctx.getMeasure('m');
  assert.ok(m.elems && m.elems.length === 2, 'jointchain retain → 2-tuple measure');
  assert.ok(Math.abs(m.logTotalmass) < 1e-9,
    `normalize(·) → log-total-mass 0; got ${m.logTotalmass}`);
  const yS = m.elems[1].samples;
  let minY = Infinity, anyNaN = false;
  for (let i = 0; i < N; i++) {
    if (Number.isNaN(yS[i])) { anyNaN = true; break; }
    if (yS[i] < minY) minY = yS[i];
  }
  assert.ok(!anyNaN, 'no NaN atoms');
  assert.ok(minY > 0, `truncation still enforced under normalize; min=${minY}`);
});

test('Stage 4: inline select IR materialises via matSelect (one select sampler)', async () => {
  // An inline `select` with two constant-weighted, well-separated Normal
  // branches — the viewer kernel-plot / CLM-body mixture shape. It used to
  // sample via the IR-direct `materialiseSelectIR` (now deleted); it routes
  // through the bridge → matSelect, the ONE select sampler (engine-concepts
  // §12). matSelect synthesizes a Bernoulli(p₀) selector from the CONSTANT
  // branch weights (no external selectorRef) — the capability folded out of
  // materialiseSelectIR — then eval-all-branches-then-gather. Before the fold
  // matSelect REFUSED a no-selectorRef select ("needs a materialisable
  // selector"), so this is red-for-the-right-reason on the new synth path.
  const materialiser = require('../materialiser.ts');
  const N = 4000;
  // A trivial module just to get a fully-wired ctx (worker, rootKey, maps).
  const { ctx } = buildCtx('x = elementof(reals)', N, 41);
  const branch = (w: number, mu: number) => ({ kind: 'call', op: 'weighted', args: [
    { kind: 'lit', value: w, numType: 'real' },
    { kind: 'call', op: 'Normal', kwargs: {
      mu: { kind: 'lit', value: mu, numType: 'real' },
      sigma: { kind: 'lit', value: 1.0, numType: 'real' } } },
  ] });
  // 0.75·Normal(-5,1) + 0.25·Normal(+5,1). K=2 synth → Bernoulli(0.75); the
  // sel?0:1 gather sends sel=1 (p=0.75) to branch 0 (mu=-5).
  const selIR = { kind: 'call', op: 'select',
    branches: [branch(0.75, -5.0), branch(0.25, 5.0)] };
  const m = await materialiser.materialiseMeasureIR(selIR, ctx);
  const s = m.samples;
  assert.equal(s.length, N);
  let neg = 0, nearHi = 0;
  for (let i = 0; i < N; i++) {
    if (s[i] < 0) neg++;
    if (s[i] > 3) nearHi++;
  }
  // ~75% of atoms come from the mu=-5 branch (negative), ~25% from mu=+5.
  assert.ok(Math.abs(neg / N - 0.75) < 0.05, `mixing ratio ≈ 0.75; got ${neg / N}`);
  assert.ok(nearHi > 0.15 * N && nearHi < 0.35 * N,
    `~25% near +5 (both modes present); got ${nearHi / N}`);
});

test('Stage 5: pushfwd IR (lowered MvNormal) materialises via matPushfwd through the bridge', async () => {
  // A fixed-D MvNormal lowers (lift, §22) to pushfwd(affine-bijection,
  // iid(Normal, D)). By-name it samples via matPushfwd's bijection-registry
  // affine fast-path. Through materialiseMeasureIR (the IR-direct path: viewer
  // concrete measure / a CLM kernel body / the collapse), expandMeasure('X')
  // yields a `pushfwd` node — fn = self-ref to the synthetic bijection binding
  // (matPushfwd reads its registryName='affine' + paramIRs from ctx.bindings),
  // base = the expanded iid(Normal, 2). Before the bridge's pushfwd arm this
  // dead-ended at the leaf fallback (the worker has no `pushfwd` kernel); now it
  // routes to matPushfwd. Assert a valid MvNormal sample: [N,2], mean ≈ [1,2],
  // and the off-diagonal covariance 0.5 reproduced (the affine L·z + b).
  const der = require('../derivations.ts');
  const mat = require('../materialiser.ts');
  const src = 'X = MvNormal(mu = [1.0, 2.0], cov = rowstack([[2.0, 0.5], [0.5, 1.0]]))';
  const { built } = buildCtx(src, 1, 7);
  const ir = der.expandMeasure('X',
    { derivations: built.derivations, bindings: built.bindings });
  assert.ok(ir && ir.op === 'pushfwd',
    `expandMeasure('X') → pushfwd (§22 lowering); got ${ir && (ir.op || ir.kind)}`);
  const N = 5000;
  const { ctx } = buildCtx(src, N, 7);
  const m = await mat.materialiseMeasureIR(ir, ctx);
  assert.ok(m.value && Array.isArray(m.value.shape) && m.value.shape.length === 2
    && m.value.shape[1] === 2, 'MvNormal pushfwd → [N,2] vector-atom measure');
  const dd = m.value.data;
  let m0 = 0, m1 = 0;
  for (let i = 0; i < N; i++) { m0 += dd[i * 2]; m1 += dd[i * 2 + 1]; }
  m0 /= N; m1 /= N;
  assert.ok(Math.abs(m0 - 1.0) < 0.1, `MvNormal mean[0] ≈ 1; got ${m0}`);
  assert.ok(Math.abs(m1 - 2.0) < 0.1, `MvNormal mean[1] ≈ 2; got ${m1}`);
  let c = 0;
  for (let i = 0; i < N; i++) c += (dd[i * 2] - m0) * (dd[i * 2 + 1] - m1);
  c /= N;
  assert.ok(Math.abs(c - 0.5) < 0.2, `cov(x0,x1) ≈ 0.5; got ${c}`);
});

test('Stage 6: iid of a composite conditioning on a per-atom draw — repeat axis (§22.4)', async () => {
  // y = iid(mixture, 4), each mixture component conditioning on a per-atom draw
  // `loc`. Spec §06 / §22.4: the 4 inner draws of atom i are iid from the
  // mixture *at loc_i* — they SHARE atom i's loc. materialiseMeasureIR's nested-
  // iid branch used to recurse at count·k WITHOUT the repeat axis (loc resolved
  // at the parent N, not tiled), decoupling the k draws from loc_i (or
  // mismatching shapes). Now it routes to matIid, whose composite fallback tiles
  // loc ×k and redraws the mixture subtree at N·k. Observable: within each
  // k-block the draws share loc_i (spread ~0.3), while loc varies across atoms
  // (~2) — so the mean within-block std is far below the overall std.
  const src = `
loc ~ Normal(mu = 0.0, sigma = 2.0)
a = Normal(mu = loc, sigma = 0.3)
b = Normal(mu = loc, sigma = 0.3)
mix = superpose(weighted(0.5, a), weighted(0.5, b))
y = iid(mix, 4)`;
  const der = require('../derivations.ts');
  const mat = require('../materialiser.ts');
  const { built } = buildCtx(src, 1, 13);
  const ir = der.expandMeasure('y',
    { derivations: built.derivations, bindings: built.bindings });
  assert.ok(ir && ir.op === 'iid', `expandMeasure('y') → iid; got ${ir && (ir.op || ir.kind)}`);
  const N = 3000, k = 4;
  const { ctx } = buildCtx(src, N, 13);
  const m = await mat.materialiseMeasureIR(ir, ctx);
  assert.ok(m.value && m.value.shape.length === 2 && m.value.shape[1] === k,
    `iid composite → [N,${k}]; got shape ${m.value && JSON.stringify(m.value.shape)}`);
  const d = m.value.data;
  const all: number[] = [];
  let sumWithin = 0;
  for (let i = 0; i < N; i++) {
    let bm = 0;
    for (let j = 0; j < k; j++) { bm += d[i * k + j]; all.push(d[i * k + j]); }
    bm /= k;
    let bv = 0;
    for (let j = 0; j < k; j++) { const e = d[i * k + j] - bm; bv += e * e; }
    sumWithin += Math.sqrt(bv / k);
  }
  const meanWithin = sumWithin / N;
  let om = 0; for (const v of all) om += v; om /= all.length;
  let ov = 0; for (const v of all) ov += (v - om) * (v - om);
  const overall = Math.sqrt(ov / all.length);
  assert.ok(meanWithin < 0.8, `within-block std small (shared loc_i); got ${meanWithin}`);
  assert.ok(overall > 1.5, `overall std tracks loc spread ~2; got ${overall}`);
  assert.ok(meanWithin < 0.5 * overall,
    `repeat axis: within-block std << overall (${meanWithin.toFixed(3)} vs ${overall.toFixed(3)})`);
});
