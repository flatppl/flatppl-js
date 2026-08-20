'use strict';

// Moment conformance for the Beta sampler, against closed-form oracles.
//
// Oracles are the closed forms for Beta(a, b), never engine output:
//
//   mean = a / (a+b)
//   var  = ab / ((a+b)² (a+b+1))
//   μ4   = 3ab (2(a+b)² + ab(a+b-6)) / ((a+b)⁴ (a+b+1)(a+b+2)(a+b+3))
//
// μ4 gives the standard error of the sample variance,
// SE = sqrt((μ4 - var²)/N), so each envelope is stated in units of its
// own analytic SE rather than as a bare magic number.
//
// A known-bad region is skipped rather than shipped red — see
// "upstream defect" below.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const rng = require('../rng.ts');

const N = 200_000;
const TOL_SIGMA = 5;

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}

function betaIR(alpha: number, beta: number) {
  return {
    kind: 'call',
    op:   'Beta',
    kwargs: {
      alpha: { kind: 'lit', value: alpha, loc: synthLoc() },
      beta:  { kind: 'lit', value: beta,  loc: synthLoc() },
    },
    loc: synthLoc(),
  };
}

function oracle(a: number, b: number, n: number) {
  const s = a + b;
  const variance = (a * b) / (s * s * (s + 1));
  const mu4 = (3 * a * b * ((2 * s * s) + (a * b * (s - 6)))) /
              (s ** 4 * (s + 1) * (s + 2) * (s + 3));
  return {
    mean:    a / s,
    variance,
    seMean:  Math.sqrt(variance / n),
    seVar:   Math.sqrt((mu4 - (variance * variance)) / n),
  };
}

function draws(alpha: number, beta: number, seed: number) {
  const s = sampler.makeSampler(rng.seedFromBytes([seed, 7, 0]), betaIR(alpha, beta), {});
  const xs = new Float64Array(N);
  for (let i = 0; i < N; i++) xs[i] = s.draw();
  return xs;
}

function sampleMoments(xs: Float64Array) {
  let sum = 0;
  for (let i = 0; i < xs.length; i++) sum += xs[i];
  const mean = sum / xs.length;
  let c2 = 0;
  for (let i = 0; i < xs.length; i++) { const d = xs[i] - mean; c2 += d * d; }
  return { mean, variance: c2 / xs.length };
}

function assertMoments(xs: Float64Array, alpha: number, beta: number, label: string) {
  const o = oracle(alpha, beta, xs.length);
  const m = sampleMoments(xs);
  const zMean = (m.mean - o.mean) / o.seMean;
  const zVar = (m.variance - o.variance) / o.seVar;
  assert.ok(Math.abs(zMean) < TOL_SIGMA,
    `Beta(${alpha},${beta}) ${label}: mean ${m.mean} vs oracle ${o.mean} — ${zMean.toFixed(1)}σ`);
  assert.ok(Math.abs(zVar) < TOL_SIGMA,
    `Beta(${alpha},${beta}) ${label}: var ${m.variance} vs oracle ${o.variance} — ${zVar.toFixed(1)}σ`);
}

function checkMoments(alpha: number, beta: number, seed: number) {
  assertMoments(draws(alpha, beta, seed), alpha, beta, `seed ${seed}`);
}

// Params supplied per draw rather than baked into the factory. A separate
// branch of the Beta wiring, with its own dispatch on each call.
function parametricDraws(alpha: number, beta: number, seed: number) {
  const s = sampler.makeParametricSampler(rng.seedFromBytes([seed, 7, 0]), betaIR(alpha, beta));
  const xs = new Float64Array(N);
  for (let i = 0; i < N; i++) xs[i] = s.drawWith({});
  return xs;
}

// Every draw must land strictly inside the open unit interval — the
// support, and a guard on the rejection loops returning an endpoint.
test('Beta draws lie in (0, 1)', () => {
  for (const [a, b] of [[0.5, 0.5], [1, 1], [2, 3], [5, 2]]) {
    const xs = draws(a, b, 11);
    for (let i = 0; i < xs.length; i++) {
      assert.ok(xs[i] > 0 && xs[i] < 1, `Beta(${a},${b}) drew ${xs[i]}`);
    }
  }
});

// @stdlib/random-base-beta dispatches on the parameter region
// (lib/beta.js), so cover every branch that is currently correct:
//   alpha < 1, beta < 1                    → sample3
//   alpha > 1, beta > 1, alpha !== beta    → sample2 (Cheng BB)
//   otherwise                              → two standard gamma variates
// The alpha === beta && alpha > 1.5 branch (sample1) is broken upstream
// and is covered by the skipped test below.
for (const [alpha, beta] of [
  [0.5, 0.5],   // sample3
  [1, 1],       // two-gamma
  [1, 3],       // two-gamma
  [2, 3],       // sample2
  [5, 2],       // sample2, asymmetric the other way — pins alpha vs beta
  [2, 2.0000001], // sample2, just off the symmetric branch
  [10, 10],     // sample1, but its bias is far below TOL_SIGMA at this N
] as [number, number][]) {
  test(`Beta(${alpha}, ${beta}): mean and variance within ${TOL_SIGMA}σ of closed form`, () => {
    for (const seed of [1, 2, 3]) checkMoments(alpha, beta, seed);
  });
}

// The symmetric alpha === beta > 1.5 region, which @stdlib draws with a
// variance biased low (beta.js:44 → sample1.js:63 — its second squeeze is
// not an upper bound on the acceptance probability, so legitimate draws
// are rejected and the near-extreme mass is depleted). The engine works
// around it by drawing that ONE region from two standard gammas; these
// rows are what pins the workaround.
//
// alpha = 1.6 and alpha = 2 are the cases the defect moves far enough to
// see at this N and tolerance: unfixed they land at −143σ and −12σ. The
// same branch biases alpha = 2.5 by −0.47 % and alpha = 3 by −0.13 %,
// both inside this envelope, so those two would pass either way and are
// listed only for coverage of the branch.
for (const alpha of [1.6, 2, 2.5, 3]) {
  test(`Beta(${alpha}, ${alpha}): mean and variance within ${TOL_SIGMA}σ of closed form`, () => {
    for (const seed of [1, 2, 3]) checkMoments(alpha, alpha, seed);
  });
}

// The per-draw-params path dispatches on every call, so it needs its own
// coverage on both sides of the branch.
test(`per-draw params: mean and variance within ${TOL_SIGMA}σ of closed form`, () => {
  for (const [a, b] of [[2, 2], [1.6, 1.6], [2, 3], [0.5, 0.5]] as [number, number][]) {
    assertMoments(parametricDraws(a, b, 5), a, b, 'per-draw params');
  }
});

// One per-draw-params closure must serve both sides of the branch, interleaved
// — the workaround builds its Gamma sampler on first need, so a closure that
// has already served unaffected params must still route correctly when an
// affected pair arrives, and vice versa.
test('per-draw params: one closure alternating across the branch stays correct', () => {
  const ir = {
    kind: 'call',
    op:   'Beta',
    kwargs: {
      alpha: { kind: 'ref', ns: 'self', name: 'a', loc: synthLoc() },
      beta:  { kind: 'ref', ns: 'self', name: 'b', loc: synthLoc() },
    },
    loc: synthLoc(),
  };
  const s = sampler.makeParametricSampler(rng.seedFromBytes([6, 7, 0]), ir);
  const pairs: [number, number][] = [[2, 3], [2, 2], [0.5, 0.5], [1.6, 1.6]];
  const out: number[][] = pairs.map(() => []);
  for (let i = 0; i < N; i++) {
    const j = i % pairs.length;
    out[j].push(s.drawWith({ a: pairs[j][0], b: pairs[j][1] }));
  }
  for (let j = 0; j < pairs.length; j++) {
    const [a, b] = pairs[j];
    assertMoments(Float64Array.from(out[j]), a, b, 'alternating closure');
  }
});

// =====================================================================
// randFn.factory edge cases (review minors on #161's workaround)
// =====================================================================

const BETA_RAND_FN = sampler._internal.REGISTRY.Beta.randFn;

// Minor 1 — the unaffected static path (alpha !== beta, or alpha <= 1.5)
// must delegate the caller's own arguments to @stdlib unchanged, so
// @stdlib's own validator sees what was actually passed, not a `+`-coerced
// number.
test('Beta.randFn: the unaffected static path delegates args unchanged to @stdlib', () => {
  // @stdlib's own validator rejects a non-number even though `+` would
  // coerce it — an array or a numeric string must throw, matching what
  // `randBeta.factory` does when called directly.
  assert.throws(() => BETA_RAND_FN.factory([2], [3], { prng: Math.random }),
    /positive number/);
  assert.throws(() => BETA_RAND_FN.factory('2', '3', { prng: Math.random }),
    /positive number/);
  // A negative/NaN static param still throws either way (unchanged).
  assert.throws(() => BETA_RAND_FN.factory(-1, -1, { prng: Math.random }),
    /positive number/);
  // A genuine number pair keeps drawing (the common, unaffected case).
  assert.doesNotThrow(() => BETA_RAND_FN.factory(2, 3, { prng: Math.random })());
});

// Minor 1, closed the rest of the way — the AFFECTED path (alpha === beta
// > 1.5) never calls `randBeta.factory` at all (it draws via two Gammas
// instead), so it got none of @stdlib's validation for free even after
// the fix above. `_validateBetaParams` reproduces that validation on the
// raw args before the Gamma route. Literal repro calls from the #161
// review (`wave-beta-review.md`, minor 1 follow-up): both used to draw
// silently; both must now throw the same way the unaffected path does.
test('Beta.randFn: the affected (two-gamma) static path also rejects a non-number', () => {
  assert.throws(() => BETA_RAND_FN.factory([2], [2], { prng: Math.random }),
    /First argument must be a positive number/);
  assert.throws(() => BETA_RAND_FN.factory('2', '2', { prng: Math.random }),
    /First argument must be a positive number/);
  // alpha valid, beta not — exercises the second-argument branch, which
  // the first-argument-only repro calls above never reach.
  assert.throws(() => BETA_RAND_FN.factory(2, [2], { prng: Math.random }),
    /Second argument must be a positive number/);
  // A genuine affected pair still draws (the fix must not touch this).
  assert.doesNotThrow(() => BETA_RAND_FN.factory(2, 2, { prng: Math.random })());
});

// Minor 2 — the parametric form (a single options-object argument) must be
// recognised by arity and type, not by requiring a `prng` key specifically.
// `{ seed }` with no `prng` is still the parametric form: @stdlib's own
// `randBeta.factory` accepts it, and so must this wrapper.
test('Beta.randFn: parametric-form detection accepts {seed} as well as {prng}', () => {
  const bySeed = BETA_RAND_FN.factory({ seed: 17 });
  assert.equal(typeof bySeed, 'function');
  const draw = bySeed(2, 3);
  assert.ok(draw > 0 && draw < 1, `draw ${draw} should lie in (0, 1)`);

  // Regression: the existing {prng}-keyed parametric form still works.
  const byPrng = BETA_RAND_FN.factory({ prng: Math.random });
  assert.equal(typeof byPrng, 'function');
  const draw2 = byPrng(2, 3);
  assert.ok(draw2 > 0 && draw2 < 1, `draw ${draw2} should lie in (0, 1)`);
});

// A/B proof that both fixes leave the draw stream unmoved for a clean,
// unaffected pair on every path — same method as the #161 review
// (`wave-beta-review.md` §2): same seed, elementwise comparison.
// makeSampler routes through the fixed delegation line (minor 1);
// makeParametricSampler routes through the fixed detection (minor 2, via
// the {prng} object it always passes) — neither line's fix changes the
// value seen by @stdlib for a normal numeric call, so the streams must
// be bit-identical to the pre-fix code path on origin/main.
test('A/B: Beta(2,3) draws are bit-identical to the pre-fix implementation', () => {
  // Captured once from origin/main (a8dba03) via makeSampler(seed=917),
  // before the two randFn.factory edits in this change — see
  // wave-jsempty-report.md for the reproduction command.
  const PRE_FIX_FIRST5 = [
    0.26847073287101375, 0.6989095105826877, 0.5390169467192227,
    0.32271517194735966, 0.35944345579589404,
  ];
  const first5 = Array.from(draws(2, 3, 917)).slice(0, 5);
  assert.deepEqual(first5, PRE_FIX_FIRST5);
});
