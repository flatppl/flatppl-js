'use strict';

// =====================================================================
// nonneg-weights.test.ts — spec §06 requires non-negative weights
// =====================================================================
//
// §06 `weighted`: "produces the measure ν(A) = ∫_A f(x) dM(x), with
// dν = f · dM, where f is a NON-NEGATIVE weight (a constant or a function
// of the variate x of M) and M the base measure."
//
// §06 `ksuperpose`: "`weights` is a distinguished input, not a member of
// the family, and never expands. It MUST BE NON-NEGATIVE but need not be
// normalized: the result has total mass Σᵢ wᵢ·totalmass(κ(θᵢ)) … and when
// every weight is ZERO it is the zero measure (density 0, log-density −∞,
// sampling undefined)."
//
// So zero is legal and negative is not, PER WEIGHT — a non-negative total
// is no defence. `superpose(weighted(-0.3, M₁), weighted(1.2, M₂))` is a
// signed set function, not a measure: it takes negative values on sets
// where M₁ dominates, so it has no density with respect to anything and
// `normalize` divides by a mass that never was one.
//
// WHAT THIS FIXED. `[-0.3, 1.2]` used to score
// -4.543469795850773 — byte-identical to the `[0.0, 1.2]` control below,
// because the density walker mapped every weight that was not strictly
// positive to log 0 = −∞. The negative component was silently DELETED and
// the engine answered with the density of a different measure. Normalized,
// the same model returned that number shifted by −log(0.9), a division by
// a signed total. Both ran green on mh and emcee too.
//
// ORACLE — closed form Σᵢ wᵢ·f(x; μᵢ, σᵢ), summed in the NON-log domain
// (so it shares no logsumexp step with the engine's §06 lowering) and
// computed in Julia before any engine output was read:
//
//   lpdf(x,m,s) = -log(s) - 0.5*log(2pi) - 0.5*((x-m)/s)^2
//   w = [0.3, 1.2]; mus = [-1.0, 2.0]; sig = [1.0, 0.5]
//   f(x) = log(sum(w[i]*exp(lpdf(x,mus[i],sig[i])) for i in 1:2))
//   f(x) - log(sum(w))    # normalized
//
//   | x    | normalized        | unnormalized       |
//   |------|-------------------|--------------------|
//   |  0.5 |  -3.411415107516122 | -3.0059499994079575 |
//   | -1.0 |  -2.528376323798943 | -2.1229112156907783 |
//   |  2.0 |  -0.447547242639128 | -0.0420821345309640 |
//
// These pin the VALID mixture across this change: enforcing the sign must
// not move a single legal number.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const TOL = 1e-9;

async function score(src: string): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) throw new Error('score: __score__ produced no data');
  return s[0];
}

function diagnosticsOf(src: string): any[] {
  return require('..').processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error');
}

const FAMILY = 'means = [-1.0, 2.0]\nsigmas = [1.0, 0.5]\n';

// =====================================================================
// The valid mixture is unchanged — the regression floor for this change
// =====================================================================

test('§06: a valid two-component mixture scores the closed form at three '
  + 'points, normalized and not', async () => {
  const oracle: [number, number, number][] = [
    [0.5, -3.411415107516122, -3.0059499994079575],
    [-1.0, -2.528376323798943, -2.1229112156907783],
    [2.0, -0.447547242639128, -0.0420821345309640],
  ];
  for (const [x, norm, unnorm] of oracle) {
    const gotN = await score('w = [0.3, 1.2]\n' + FAMILY
      + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'
      + `__score__ = logdensityof(mix, ${x})\n`);
    assert.ok(Math.abs(gotN - norm) <= TOL,
      `normalized x = ${x}: got ${gotN}, oracle ${norm}`);
    const gotU = await score('w = [0.3, 1.2]\n' + FAMILY
      + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'
      + `__score__ = logdensityof(mix, ${x})\n`);
    assert.ok(Math.abs(gotU - unnorm) <= TOL,
      `unnormalized x = ${x}: got ${gotU}, oracle ${unnorm}`);
  }
});

test('§06: a ZERO weight stays legal — it drops the component out, it is not '
  + 'a sign violation', async () => {
  const got = await score('w = [0.0, 1.2]\n' + FAMILY
    + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  // log(1.2) + lpdf(0.5; 2.0, 0.5), by hand — no mixture involved.
  assert.ok(Math.abs(got - (-4.543469795850773)) <= TOL,
    `got ${got}, closed form -4.543469795850773`);
  assert.deepEqual(diagnosticsOf('w = [0.0, 1.2]\n' + FAMILY
    + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'), []);
});

test('§06: ALL-zero weights stay legal — the zero measure, log-density −∞',
  async () => {
    const got = await score('w = [0.0, 0.0]\n' + FAMILY
      + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'
      + '__score__ = logdensityof(mix, 0.5)\n');
    assert.equal(got, -Infinity);
  });

// =====================================================================
// Static per-weight refusal — ksuperpose
// =====================================================================

test('§06: ONE negative weight in a ksuperpose vector is refused, even with a '
  + 'positive total', () => {
  const ds = diagnosticsOf('w = [-0.3, 1.2]\n' + FAMILY
    + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n');
  const hit = ds.find((d: any) => /non-negative/.test(d.message));
  assert.ok(hit, `want a sign refusal, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
  // It must name WHICH weight, or the reader has to guess in a long vector.
  assert.match(hit.message, /weight #1/);
  assert.match(hit.message, /-0\.3/);
  assert.match(hit.message, /§06/);
  assert.ok(hit.loc, 'the refusal is located');
});

test('the offending index is the real one, not always the first', () => {
  const ds = diagnosticsOf('w = [0.3, 1.2, -0.5]\n'
    + 'means3 = [-1.0, 2.0, 4.0]\n'
    + 'mix = ksuperpose(Normal, w)(mu = means3, sigma = 1.0)\n');
  const hit = ds.find((d: any) => /non-negative/.test(d.message));
  assert.ok(hit, `want a sign refusal, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
  assert.match(hit.message, /weight #3/);
});

test('an INLINE weight-vector literal is checked too', () => {
  const ds = diagnosticsOf(FAMILY
    + 'mix = ksuperpose(Normal, [-0.3, 1.2])(mu = means, sigma = sigmas)\n');
  assert.ok(ds.some((d: any) => /weight #1/.test(d.message) && /non-negative/.test(d.message)),
    `got ${JSON.stringify(ds.map((d: any) => d.message))}`);
});

test('a NEGATIVE TOTAL is refused at the offending weight, not deferred to the '
  + 'normalizer — the diagnosis is the weight, not the mass', () => {
  const ds = diagnosticsOf('w = [-1.3, 0.2]\n' + FAMILY
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n');
  assert.ok(ds.some((d: any) => /weight #1/.test(d.message) && /non-negative/.test(d.message)),
    `got ${JSON.stringify(ds.map((d: any) => d.message))}`);
});

// =====================================================================
// Static refusal — the hand-written superpose(weighted(…)) spelling
// =====================================================================

test('§06: a negative constant weight on `weighted` is refused with a located '
  + 'message, not left as an undiagnosed "no derivation"', () => {
  const ds = diagnosticsOf(FAMILY
    + 'mix = superpose(weighted(-0.3, Normal(means[1], sigmas[1])), '
    + 'weighted(1.2, Normal(means[2], sigmas[2])))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  const hit = ds.find((d: any) => /non-negative/.test(d.message));
  assert.ok(hit, `want a sign refusal, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
  assert.match(hit.message, /weighted/);
  assert.match(hit.message, /-0\.3/);
  assert.ok(hit.loc, 'the refusal is located');
});

test('a bare `weighted` with a negative weight is refused', () => {
  const ds = diagnosticsOf('mix = weighted(-0.3, Normal(0.0, 1.0))\n');
  assert.ok(ds.some((d: any) => /non-negative/.test(d.message)),
    `got ${JSON.stringify(ds.map((d: any) => d.message))}`);
});

test('`logweighted` is NOT sign-checked — its weight is exp(logweight), so a '
  + 'negative log-weight is an ordinary weight below one', async () => {
  assert.deepEqual(diagnosticsOf('mix = logweighted(-0.3, Normal(0.0, 1.0))\n'), []);
  const got = await score('mix = logweighted(-0.3, Normal(0.0, 1.0))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  const want = -0.3 + (-0.5 * Math.log(2 * Math.PI) - 0.125);
  assert.ok(Math.abs(got - want) <= TOL, `got ${got}, closed form ${want}`);
});

test('a positive constant weight raises nothing — the check does not fire on '
  + 'the legal spelling', () => {
  assert.deepEqual(diagnosticsOf('mix = weighted(0.3, Normal(0.0, 1.0))\n'), []);
  assert.deepEqual(diagnosticsOf('mix = weighted(0.0, Normal(0.0, 1.0))\n'), []);
});

// =====================================================================
// Runtime refusal — a weight no static pass can read
// =====================================================================

test('§06: a negative weight that only appears at RUNTIME is refused while '
  + 'scoring, not silently zeroed', async () => {
  // `-0.3` reaches the walker through arithmetic, so no static pass folds
  // it. This is the case that produced the wrong NUMBER rather than a bad
  // message: the walker used to answer -4.543469795850773, the density of
  // the mixture with that component deleted.
  const src = 'a = 0.3\nw = [0.0 - a, 1.2]\n' + FAMILY
    + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'
    + '__score__ = logdensityof(mix, 0.5)\n';
  await assert.rejects(() => score(src), (e: any) => {
    assert.match(e.message, /non-negative/);
    assert.match(e.message, /-0\.3/);
    assert.match(e.message, /§06/);
    return true;
  });
});

test('a MINUS over a non-literal does not fold, so the static pass defers and '
  + 'the walker refuses', async () => {
  // `[-a, 1.2]` is a unary minus whose operand is a name: the static pass
  // cannot know the value, so it must NOT refuse (a negated name could hold a
  // negative number and be a perfectly legal positive weight). The refusal
  // therefore has to come from the walker, at the value.
  const src = 'a = 0.3\nw = [-a, 1.2]\n' + FAMILY
    + 'mix = ksuperpose(Normal, w)(mu = means, sigma = sigmas)\n'
    + '__score__ = logdensityof(mix, 0.5)\n';
  assert.deepEqual(diagnosticsOf(src), [],
    'the static pass must not fold a minus over a name');
  await assert.rejects(() => score(src), /non-negative/);

  // The control: negating a name that holds a NEGATIVE number is a POSITIVE
  // weight, and must score. This is why the static pass cannot just refuse
  // any leading minus. -(-0.3) = 0.3, so this is the valid oracle row.
  const ok = await score('a = 0.0 - 0.3\nw = [-a, 1.2]\n' + FAMILY
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(ok - (-3.411415107516122)) <= TOL,
    `got ${ok}, oracle -3.411415107516122`);
});

test('§06: a NaN weight gets its own message and is NOT reported as negative '
  + 'total mass', async () => {
  // The old path let NaN through the weight, so `-log Z` came out NaN and
  // the normalize guard announced "NEGATIVE total mass" — a cause it could
  // not know, sending the reader to look for a minus sign that is not there.
  const src = 'w = [0.0 / 0.0, 1.2]\n' + FAMILY
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n';
  await assert.rejects(() => score(src), (e: any) => {
    assert.match(e.message, /not a number|NaN/);
    assert.doesNotMatch(e.message, /NEGATIVE/);
    return true;
  });
});

// =====================================================================
// The `normalize` shift guard names only what it can know
// =====================================================================

// Driven through a HAND-BUILT IR rather than a model, because after the fix no
// model reaches the NaN arm: every route that used to put a NaN into `-log Z`
// went through a NaN or negative WEIGHT, and those now refuse at the weight
// (the two tests above). The arm stays as a guard for the rewrite sites that
// bake a `-log Z` literal from a mass computation of their own
// (`mat-density.ts` resolveTruncateNormalizers / resolveProductNormalizers),
// and this is the only way left to exercise it.
//
// `fromNormalize` is what marks a `logweighted` node as carrying `-log Z`
// (density.ts walkLogWeighted), so these three nodes are exactly what a
// resolved `normalize` hands the walker for Z = NaN, Z = 0 and Z = ∞.
test('§06 normalize: the shift guard reports NOT A NUMBER, ZERO and INFINITE '
  + 'as three distinct causes', () => {
  const density = require('../density.ts');
  const N01 = {
    kind: 'call', op: 'Normal',
    kwargs: { mu: { kind: 'lit', value: 0.0 }, sigma: { kind: 'lit', value: 1.0 } },
  };
  const shifted = (v: number) => ({
    kind: 'call', op: 'logweighted', fromNormalize: true,
    args: [{ kind: 'lit', value: v }, N01],
  });
  const cases: [number, RegExp][] = [
    [NaN, /NOT A NUMBER/],
    [Infinity, /ZERO total mass/],      // -log Z = +∞ ⇔ Z = 0
    [-Infinity, /INFINITE total mass/], // -log Z = -∞ ⇔ Z = ∞
  ];
  for (const [v, want] of cases) {
    assert.throws(() => density.logDensityN(shifted(v), 0.5, {}, 1, {}),
      (e: any) => {
        assert.match(e.message, want);
        assert.match(e.message, /spec §06/);
        // The MCMC scorer keys off this to refuse instead of rejecting a
        // proposal; an untagged throw would be swallowed to −∞.
        assert.equal(e.undefinedNormalize, true);
        return true;
      }, `shift ${v} should report ${want}`);
  }
  // The NaN arm must not name a cause it cannot know: the mass came out NaN
  // OR negative, and the guard sees only the NaN shift either way.
  assert.throws(() => density.logDensityN(shifted(NaN), 0.5, {}, 1, {}),
    (e: any) => {
      assert.doesNotMatch(e.message, /NEGATIVE/);
      return true;
    });
  // Vacuity guard: a FINITE shift is an ordinary weight and must pass through,
  // or the three throws above would prove nothing about the guard.
  const finite = density.logDensityN(shifted(-Math.log(2)), 0.5, {}, 1, {})[0];
  const want = -Math.log(2) + (-0.5 * Math.log(2 * Math.PI) - 0.125);
  assert.ok(Math.abs(finite - want) <= TOL, `got ${finite}, closed form ${want}`);
});

// =====================================================================
// The MCMC backends refuse too, rather than running a signed chain
// =====================================================================

test('a runtime negative weight in the LIKELIHOOD reaches the MCMC ModelView as '
  + 'a refusal, not as a rejected proposal', async () => {
  // mh/emcee swallow every density throw to −∞ (a proposal outside support),
  // which would turn this model into a constant chain with no diagnostic.
  // The refusal is tagged so the tractability probe re-raises it, exactly as
  // the §06-undefined-normalize refusal already is.
  const MV = require('../model-view.ts');
  const src = 'a = 0.3\nw = [0.0 - a, 1.2]\n'
    + 'mu ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'mix = ksuperpose(Normal, w)(mu = [mu, 2.0], sigma = [1.0, 0.5])\n'
    + 'y ~ mix\n'
    + 'prior = lawof(record(mu = mu))\n'
    + 'fk = kernelof(record(y = y), mu = mu)\n'
    + 'L = likelihoodof(fk, record(y = 0.5))\n'
    + 'posterior = bayesupdate(L, prior)\n';
  const { ctx } = ctxFor(src, 64);
  const dv = ctx.lookupDerivation
    ? ctx.lookupDerivation('posterior') : ctx.derivations.posterior;
  await assert.rejects(() => MV.buildModelViewFromCtx(ctx, dv), /non-negative/);
});
