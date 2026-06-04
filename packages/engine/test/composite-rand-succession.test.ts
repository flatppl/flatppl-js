'use strict';

// =====================================================================
// Composite-`rand` succession (engine-concepts §11 / §17.4).
// =====================================================================
//
// The STATE half of a composite `rand` — `tuple_get(<rand>, 1)`, a
// successor rngstate for chaining a second `rand` — is rewritten by lift's
// rewriteCompositeRandSucc to the value-domain `rand_succ` op (split lane 1
// of the parent key; the draw, matRandSample, takes lane 0). This makes the
// successor a synchronously-evaluable VALUE so a chained composite `rand`
// resolves through fixedValues / resolveIRToValue instead of hitting the
// per-draw walker (which can't sample the forward composite and would
// throw).
//
// Properties pinned (NOT circular value-pinning — RNG seeds are observable
// so we assert structural + statistical properties, not specific draws):
//   1. CLASSIFICATION: the state-half binding IR becomes rand_succ; the
//      draw halves are randsample; leaf state-halves are NOT rewritten.
//   2. CHAINING: the second composite draw materialises (was unreachable).
//   3. INDEPENDENCE: the two chained draws are domain-separated streams.
//   4. DETERMINISM: a composite draw is fixed — independent of session N.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { makeMatCtx } = require('./_materialise-helpers.ts');

const FIXTURE = path.join(__dirname, 'fixtures', 'polyeval-iid-broadcast-chain.flatppl');

// polyeval(C, x) = 2.3 + 1.5x + 0.7x² with x ~ Normal(0,1): E ≈ 3.0.
const EXPECTED_MEAN = 3.0;

function meanOf(data: any): number {
  let s = 0;
  for (let i = 0; i < data.length; i++) s += data[i];
  return s / data.length;
}

function fractionDiffer(a: any, b: any): number {
  const n = Math.min(a.length, b.length);
  let diff = 0;
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) diff++;
  return diff / n;
}

test('composite-rand succession: state half rewrites to rand_succ; draws are randsample', () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { built } = makeMatCtx(src, { sampleCount: 64 });

  // The non-discard state half `s2` is rewritten to the value-domain
  // successor op (NOT left as tuple_get → walk → throw).
  const s2 = built.bindings.get('s2');
  assert.ok(s2 && s2.ir, 's2 binding exists');
  assert.equal(s2.ir.op, 'rand_succ',
    's2 (composite state half) must be rewritten to rand_succ');
  assert.equal(s2.ir.args.length, 1, 'rand_succ takes the state arg only');
  // It is classified as an ordinary evaluable (not a randsample / walk).
  assert.equal(built.derivations.s2.kind, 'evaluate',
    'rand_succ binding classifies as evaluate');

  // Both composite draws route to randsample off Y_dist.
  assert.equal(built.derivations.Y1.kind, 'randsample');
  assert.equal(built.derivations.Y1.from, 'Y_dist');
  assert.equal(built.derivations.Y1.count, 200);
  assert.equal(built.derivations.Y2.kind, 'randsample');
  assert.equal(built.derivations.Y2.from, 'Y_dist');
  assert.equal(built.derivations.Y2.count, 200);
});

test('composite-rand succession: the chained second draw materialises', async () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { ctx } = makeMatCtx(src, { sampleCount: 64 });

  const m1 = await ctx.getMeasure('Y1');
  const m2 = await ctx.getMeasure('Y2');   // was unreachable (walk threw on the successor)
  for (const [name, m] of [['Y1', m1], ['Y2', m2]] as any) {
    assert.ok(m && m.value, `${name} materialises with a .value`);
    assert.deepEqual(m.value.shape, [200, 10], `${name} shape [count, variate]`);
    assert.equal(m.value.data.length, 2000);
    for (let i = 0; i < m.value.data.length; i++) {
      assert.ok(Number.isFinite(m.value.data[i]), `${name}[${i}] finite`);
    }
    assert.ok(Math.abs(meanOf(m.value.data) - EXPECTED_MEAN) < 0.2,
      `${name} mean ≈ ${EXPECTED_MEAN}`);
  }
});

test('composite-rand succession: chained draws are independent (domain-separated lanes)', async () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { ctx } = makeMatCtx(src, { sampleCount: 64 });

  const d1 = (await ctx.getMeasure('Y1')).value.data;
  const d2 = (await ctx.getMeasure('Y2')).value.data;
  // Y1 draws off split lane 0 of rstate; Y2 off split lane 0 of the
  // successor (split lane 1 of rstate). Independent streams → essentially
  // every element differs (a coincidental match would be a ~2^-32 event).
  const frac = fractionDiffer(d1, d2);
  assert.ok(frac >= 0.9,
    `Y1 and Y2 should be independent: only ${(frac * 100).toFixed(1)}% of elements differ`);
  // And each stream's own draws are distinct realisations (not one tiled).
  const distinct = new Set(Array.from(d2.slice(0, 50)).map((x: any) => x.toFixed(8)));
  assert.equal(distinct.size, 50, 'Y2 draws are distinct, not repeated');
});

test('composite-rand succession: a composite draw is fixed — independent of session N', async () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const a = makeMatCtx(src, { sampleCount: 16, rootSeed: 1 });
  const b = makeMatCtx(src, { sampleCount: 512, rootSeed: 999 });

  const a2 = (await a.ctx.getMeasure('Y2')).value.data;
  const b2 = (await b.ctx.getMeasure('Y2')).value.data;
  assert.equal(a2.length, b2.length);
  // Byte-identical despite different session N AND rootSeed — the chained
  // draw is seeded only off rstate (the in-language rnginit), through the
  // rand_succ successor, not the session key.
  let maxDiff = 0;
  for (let i = 0; i < a2.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a2[i] - b2[i]));
  assert.equal(maxDiff, 0, 'Y2 is deterministic given rstate, independent of session');
});

test('composite-rand succession: a LEAF state half is NOT rewritten (stays threaded)', () => {
  // The gate must fire ONLY for composite inners. A leaf rand's state half
  // stays a tuple_get so it keeps threading the counter-advanced successor
  // via sampleLeafN (preserving builtin_sample ≡ rand+iid). Routing a leaf
  // successor through the split-key rand_succ would silently change leaf
  // chaining.
  const src = `
rstate = rnginit([1, 2, 3, 4])
A, As = rand(rstate, iid(Normal(0, 1), [3, 3]))
`;
  const { built } = makeMatCtx(src, { sampleCount: 16 });
  const as = built.bindings.get('As');
  assert.ok(as && as.ir, 'As binding exists');
  assert.notEqual(as.ir.op, 'rand_succ',
    'a LEAF state half must NOT be rewritten to rand_succ');
  assert.equal(as.ir.op, 'tuple_get',
    'leaf state half stays a tuple_get (threaded successor via sampleLeafN)');
});

test('composite-rand succession: a BARE composite rand fails loud (deferred)', async () => {
  // A bare un-destructured `r = rand(state, iid(<composite>, n))` has no
  // tuple_get projection to reclassify, so its 2-tuple (async display-array
  // draw + sync rngstate successor) has no value representation yet
  // (deferred — see TODO). It FAILS LOUD at the walker (never silently
  // wrong): the inner measure is a hoisted self-ref the bare evaluate-path
  // can't resolve, so it surfaces at the measure-ref guard — which carries
  // the destructure guidance pointing at the supported form.
  const src = `
polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
C = [2.3, 1.5, 0.7]
X ~ iid(Normal(0, 1), 10)
Y = polyeval.([C], X)
Y_dist = lawof(Y)
rstate = rnginit([1, 2, 3, 4])
r = rand(rstate, iid(Y_dist, 200))
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 16 });
  await assert.rejects(
    async () => { await ctx.getMeasure('r'); },
    (e: any) => /sampler\.walk:/.test(e.message) && /destructure it/.test(e.message),
    'bare composite rand should fail loud with the destructure guidance');
});
