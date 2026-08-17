'use strict';

// wave-j1 fix round 1: a NAMED-BINDING weighted/logweighted measure sampled
// via `rand` is a second live route to the same defect walkWeightedRefuse
// (sampler.ts) closes for the INLINE form. `w = weighted(...); s, _ =
// rand(state, w)` classifies as a `randsample` derivation (materialiser.ts)
// — a completely different code path from `rand(state, weighted(...))`
// written inline, which classifies `evaluate` and routes through
// sampler.walk. `matRandSample` used to resolve `w`'s materialised measure
// and rebuild an unweighted array measure from its `.samples`/`.value.data`,
// discarding `logWeights` entirely — the identical silent-wrong-answer
// defect, one binding shape away from the one sampler.ts fixed.
//
// Covers: bare named weighted/logweighted, the function-weight form (where
// the derivation carries an explicit `isLog`), `iid(w, n)` over a named
// weighted binding (same `d.from` field per classifyRandTuple, lift.ts),
// and an alias chain (`w2 = w`) — alias-resolution collapses the ref before
// classification, so `d.from` lands on the underlying `weighted` binding
// either way.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { makeMatCtx } = require('./_materialise-helpers.ts');

const RNGSEED_SRC = `
rngseed = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
rstate = rnginit(rngseed)
`;

test('randsample: named weighted binding refuses to sample rather than dropping the weight', async () => {
  const src = RNGSEED_SRC + `
w = weighted(2.0, Normal(0, 1))
s, rstate2 = rand(rstate, w)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  assert.equal(built.derivations.s.kind, 'randsample',
    'named-binding rand is a randsample derivation, not evaluate');
  await assert.rejects(
    () => ctx.getMeasure('s'),
    /'weighted\/logweighted' cannot be sampled \(spec §07 sec:random\)/,
  );
});

test('randsample: named logweighted binding refuses to sample rather than dropping the weight', async () => {
  const src = RNGSEED_SRC + `
w = logweighted(0.5, Normal(0, 1))
s, rstate2 = rand(rstate, w)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  assert.equal(built.derivations.s.kind, 'randsample');
  await assert.rejects(
    () => ctx.getMeasure('s'),
    /'weighted\/logweighted' cannot be sampled \(spec §07 sec:random\)/,
  );
});

test('randsample: function-weighted named binding names the exact op (weighted)', async () => {
  const src = RNGSEED_SRC + `
w = weighted(fn(abs(_) + 0.1), Normal(0, 1))
s, rstate2 = rand(rstate, w)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  assert.equal(built.derivations.s.kind, 'randsample');
  // A function-weight derivation carries an explicit isLog boolean, so the
  // message can name the exact surface op rather than the generic pair.
  await assert.rejects(
    () => ctx.getMeasure('s'),
    /'weighted' cannot be sampled \(spec §07 sec:random\)/,
  );
});

test('randsample: function-logweighted named binding names the exact op (logweighted)', async () => {
  const src = RNGSEED_SRC + `
w = logweighted(fn(-abs(_)), Normal(0, 1))
s, rstate2 = rand(rstate, w)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  assert.equal(built.derivations.s.kind, 'randsample');
  await assert.rejects(
    () => ctx.getMeasure('s'),
    /'logweighted' cannot be sampled \(spec §07 sec:random\)/,
  );
});

test('randsample: iid(w, n) over a named weighted binding also refuses', async () => {
  const src = RNGSEED_SRC + `
w = weighted(2.0, Normal(0, 1))
s, rstate2 = rand(rstate, iid(w, 5))
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  assert.equal(built.derivations.s.kind, 'randsample');
  assert.equal(built.derivations.s.from, 'w');
  await assert.rejects(
    () => ctx.getMeasure('s'),
    /'weighted\/logweighted' cannot be sampled \(spec §07 sec:random\)/,
  );
});

test('randsample: an alias chain to a weighted binding still refuses', async () => {
  const src = RNGSEED_SRC + `
w = weighted(2.0, Normal(0, 1))
w2 = w
s, rstate2 = rand(rstate, w2)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  // Alias-resolution (lift.ts) collapses the ref before classification, so
  // the randsample derivation's \`from\` lands on the underlying weighted
  // binding regardless of the alias hop.
  assert.equal(built.derivations.s.from, 'w');
  await assert.rejects(
    () => ctx.getMeasure('s'),
    /'weighted\/logweighted' cannot be sampled \(spec §07 sec:random\)/,
  );
});

test('randsample: a non-weighted composite still samples (no regression)', async () => {
  // Sanity check pinning that this fix is scoped to weighted/logweighted —
  // an ordinary composite named binding (a deterministic transform of a
  // stochastic draw) still routes through matRandSample and materialises.
  const src = RNGSEED_SRC + `
x ~ Normal(0, 1)
y = 2.0 * x
law = lawof(y)
s, rstate2 = rand(rstate, law)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 8 });
  assert.equal(built.derivations.s.kind, 'randsample');
  const m = await ctx.getMeasure('s');
  assert.equal(typeof m.value.data[0], 'number');
  assert.ok(Number.isFinite(m.value.data[0]));
});
