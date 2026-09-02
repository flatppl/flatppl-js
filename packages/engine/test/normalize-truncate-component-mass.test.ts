'use strict';

// =====================================================================
// normalize-truncate-component-mass.test.ts — the DENSITY route
// =====================================================================
//
// A mixture with a `truncate` component has a θ-DEPENDENT total mass. §06
// "Support restriction" makes `truncate(M, S)` "ν(A) = M(A ∩ S). Does not
// normalize automatically", so the component's mass is
// `Z_t = 2Φ(1) − 1 = 0.6826894921370859`, and §06 `superpose`'s
// "ν(A) = M₁(A) + M₂(A) + …" makes the superposition's `Z(θ) = θ·Z_t + 1`.
// §06 "Density of composed measures" then fixes the shift: "`normalize` (from
// M / Z): logdensityof(normalize(M), x) = logdensityof(M, x) − log Z".
//
// THE DEFECT. `normalize-mass.totalMassExpr` had no `truncate` arm, so this
// mixture had no expression at all and the density route fell to the
// θ-CONSTANT materialised Z bake. The error therefore GREW with θ rather than
// sitting at a fixed offset: −10.253382 / −7.504509 / −5.741149 at
// θ = 0.2 / 0.5 / 0.9 against the closed form below, i.e. 0.13, 0.96 and 1.89
// nats out. The companion sampling witnesses are in
// normalize-pooled-divisor.test.ts.
//
// THE SECOND DEFECT, and it is why the arm alone was not enough.
// `mat-density.asScalarFactor` reads `kwargs` only, so it declined the
// POSITIONAL leaf spelling `Normal(0.0, 1.0)` — two spellings of one measure
// would have taken different mass routes and disagreed. The arm resolves
// parameters with `sampler-registry.resolveParamsN`'s precedence instead
// (kwargs by name, then by alias, then positional by declared index), so BOTH
// spellings are scored here and must give the same number.
//
// THE ORACLE, mpmath at 40 digits, independent of the engine:
//   density(y; θ) = (θ·1[|y| ≤ 1]·N(y; 0, 1) + N(y; 10, 1)) / (θ·Z_t + 1)
// summed in logs over y = [0.1, −0.2, 0.3, 9.5, 10.5]. Two of the five points
// sit OUTSIDE the truncation set, where §06 `truncate`'s density is −∞ and
// only the second component contributes — so the truncation genuinely bites
// rather than cancelling.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { totalMassExpr } = require('../normalize-mass.ts');

const YDATA = [0.1, -0.2, 0.3, 9.5, 10.5];

// `weighted(1.0, ·)` lowers to `logweighted(0, ·)`, so this also drives
// totalMassExpr's log-space arm alongside the truncate one.
const MODEL = (theta: number, trunc: string, other: string) => `flatppl_compat = "0.1"
theta ~ Beta(2.0, 2.0)
mixdir = normalize(superpose(weighted(theta, truncate(${trunc}, interval(-1.0, 1.0))), weighted(1.0, ${other})))
y ~ iid(mixdir, 5)
forward_kernel = kernelof(record(y = y), theta = theta)
L = likelihoodof(forward_kernel, record(y = [${YDATA.join(', ')}]))
__score__ = logdensityof(L, record(theta = ${theta.toFixed(6)}))
`;

// The two spellings of the same two leaves: positional, then keyword.
const SPELLINGS: Array<[string, string, string]> = [
  ['positional', 'Normal(0.0, 1.0)', 'Normal(10.0, 1.0)'],
  ['keyword', 'Normal(mu = 0.0, sigma = 1.0)', 'Normal(mu = 10.0, sigma = 1.0)'],
];

// [θ, exact, the value if the truncate component's mass were taken as 1]
const ORACLE: Array<[number, number, number]> = [
  [0.2, -10.38293995517070597, -10.654614187295437964],
  [0.5, -8.4624974706708800857, -9.021459748244021547],
  [0.9, -7.6256546896974614216, -8.4400436438588164925],
];

async function scoreOf(src: string): Promise<number> {
  const { proc, ctx } = ctxFor(src, 1);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const m = await ctx.getMeasure('__score__');
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) throw new Error('scoreOf: __score__ produced no data');
  return s[0];
}

for (const [label, trunc, other] of SPELLINGS) {
  for (const [theta, exact, mass1] of ORACLE) {
    test(`a truncate component's mass reaches the normalize shift (${label}, θ=${theta})`,
      async () => {
        const got = await scoreOf(MODEL(theta, trunc, other));
        assert.ok(Math.abs(got - exact) < 1e-12,
          `θ=${theta}: engine ${got} vs closed form ${exact} (Δ ${Math.abs(got - exact)})`);
        // The mass-1 hypothesis — the component treated as a probability
        // measure — is 0.27 to 0.81 nats away, so it is excluded rather than
        // merely un-asserted.
        assert.ok(Math.abs(got - mass1) > 0.2,
          `θ=${theta}: engine ${got} is the mass-1 value ${mass1}`);
      });
  }
}

test('the θ-constant bake is excluded: the shift tracks θ across all three points',
  async () => {
    // A baked Z is one number, so the three scores would differ only by the
    // numerator. The exact gaps are 1.9204424844998259 and 0.8368427809734187
    // nats; a baked Z gave 2.7488721956 and 1.7633599947 (measured before the
    // fix), so the SPACING discriminates even without the absolute values.
    const s: number[] = [];
    for (const [theta] of ORACLE) s.push(await scoreOf(MODEL(theta, 'Normal(0.0, 1.0)', 'Normal(10.0, 1.0)')));
    assert.ok(Math.abs((s[1] - s[0]) - 1.9204424844998259) < 1e-9,
      `gap θ=0.2→0.5 is ${s[1] - s[0]}, closed form 1.9204424844998259`);
    assert.ok(Math.abs((s[2] - s[1]) - 0.8368427809734187) < 1e-9,
      `gap θ=0.5→0.9 is ${s[2] - s[1]}, closed form 0.8368427809734187`);
  });

test('the two leaf spellings score bit-for-bit alike', async () => {
  // The reason the arm resolves parameters through the registry rather than
  // reading kwargs: a positional leaf that declined the arm would keep the
  // θ-constant bake while its keyword twin was exact.
  for (const [theta] of ORACLE) {
    const pos = await scoreOf(MODEL(theta, SPELLINGS[0][1], SPELLINGS[0][2]));
    const kw = await scoreOf(MODEL(theta, SPELLINGS[1][1], SPELLINGS[1][2]));
    assert.equal(pos, kw, `θ=${theta}: positional ${pos} vs keyword ${kw}`);
  }
});

test('a truncated Uniform component is expressed too, not left to the bake',
  async () => {
    // `Uniform` declares ONE parameter, `support`, and takes its bounds from an
    // `interval` inside it, while the CDF row reads lo/hi — so it needs its own
    // mapping in the arm or it would be the single hole in the accepted set.
    // Z_u = mass(truncate(Uniform([0,4]), [−1,1])) = (1 − 0)/4 = 0.25, so
    // Z(θ) = 0.25θ + 1. At y = 0.5 the Uniform density is 1/4 and the
    // Normal(10,1) contribution is negligible but not zero, so the score is
    // computed in full below rather than approximated.
    const theta = 0.8;
    const src = 'flatppl_compat = "0.1"\n'
      + 'theta ~ Beta(2.0, 2.0)\n'
      + 'mixdir = normalize(superpose('
      + 'weighted(theta, truncate(Uniform(interval(0.0, 4.0)), interval(-1.0, 1.0))), '
      + 'weighted(1.0, Normal(mu = 10.0, sigma = 1.0))))\n'
      + 'y ~ iid(mixdir, 1)\n'
      + 'forward_kernel = kernelof(record(y = y), theta = theta)\n'
      + 'L = likelihoodof(forward_kernel, record(y = [0.5]))\n'
      + '__score__ = logdensityof(L, record(theta = ' + theta.toFixed(6) + '))\n';
    // (0.8·0.25 + N(0.5; 10, 1)) / (0.8·0.25 + 1), in logs — mpmath, 40 digits.
    const exact = -1.7917594692280550;
    const mass1 = -2.1972245773362194;   // if Z were 0.8 + 1
    const got = await scoreOf(src);
    assert.ok(Math.abs(got - exact) < 1e-12,
      `engine ${got} vs closed form ${exact} (Δ ${Math.abs(got - exact)})`);
    assert.ok(Math.abs(got - mass1) > 0.1, `engine ${got} is the mass-1 value ${mass1}`);
  });

// =====================================================================
// THE DECLINES, driven directly
// =====================================================================
//
// Each of these keeps the caller's existing route rather than emitting a
// literal. Most are unreachable through a whole model — `matTruncate` needs the
// same CDF the arm does and throws first, a discrete component fails §06's
// variate-space check statically, an unbounded `Uniform` is refused by the
// sampler — so the arm is driven on its own IR. A decline that only ever showed
// up as somebody else's error message would not be pinned at all.

const _lit = (v: number) => ({ kind: 'lit', value: v });
const _interval = (lo: number, hi: number) =>
  ({ kind: 'call', op: 'interval', args: [_lit(lo), _lit(hi)] });
const _trunc = (base: any, set: any) => ({ kind: 'call', op: 'truncate', args: [base, set] });
// The mixture the arm actually sits inside: `weighted(θ, truncate(...))`. Its
// mass is θ·M(S), so a declining component makes the WHOLE expression null,
// which is what the callers branch on.
const _weighted = (m: any) =>
  ({ kind: 'call', op: 'weighted', args: [{ kind: 'ref', ns: 'self', name: 'theta' }, m] });

test('the arm expresses a truncated Normal and declines every unsupported base',
  async () => {
    const Zt = 0.6826894921370859;   // 2Φ(1) − 1
    const ok = totalMassExpr(_trunc(
      { kind: 'call', op: 'Normal', kwargs: { mu: _lit(0), sigma: _lit(1) } },
      _interval(-1, 1)));
    assert.deepEqual(ok, _lit(Zt), 'a constant-parameter Normal must give F(1) − F(−1)');

    const decline = (label: string, m: any) =>
      assert.equal(totalMassExpr(_weighted(m)), null, label);

    // No CDF row: eleven registered continuous kernels have none.
    decline('VonMises has no CDF row', _trunc(
      { kind: 'call', op: 'VonMises', kwargs: { mu: _lit(0), kappa: _lit(1) } },
      _interval(-1, 1)));
    // A DISCRETE base's restricted mass is Σ pmf over S; the continuous CDF
    // difference drops the lower endpoint, and §06 defines no discrete-truncate
    // normalizer.
    decline('a discrete base must not take the CDF difference', _trunc(
      { kind: 'call', op: 'Poisson', kwargs: { rate: _lit(3) } }, _interval(0, 5)));
    // §04 makes a surplus keyword a static error, raised by whichever call site
    // recognises the distribution. This builder's contract is an expression or
    // null, so it must decline rather than throw that error from here.
    decline('a surplus keyword declines instead of throwing', _trunc(
      { kind: 'call', op: 'Normal', kwargs: { mu: _lit(0), sigma: _lit(1), zz: _lit(9) } },
      _interval(-1, 1)));
    // Not a distribution at all: a reference measure's mass over S is a volume,
    // not a CDF difference.
    decline('a Lebesgue base is not a scalar leaf', _trunc(
      { kind: 'call', op: 'Lebesgue', args: [_interval(-1, 1)] }, _interval(-0.5, 0.5)));
    decline('a non-call base has no kernel', _trunc(
      { kind: 'ref', ns: 'self', name: 'base' }, _interval(-1, 1)));
    // A parameter that moves with a latent has no value at rewrite time.
    decline('a latent-parameterised base declines', _trunc(
      { kind: 'call', op: 'Normal', kwargs: { mu: { kind: 'ref', ns: 'self', name: 'p' }, sigma: _lit(1) } },
      _interval(-1, 1)));
    // A missing parameter resolves to nothing, so the leaf is not constant.
    decline('a leaf missing a parameter declines', _trunc(
      { kind: 'call', op: 'Normal', kwargs: { mu: _lit(0) } }, _interval(-1, 1)));
    // §06 leaves normalize undefined at Z = 0. Φ(101) − Φ(100) underflows to
    // exactly 0, so the arm must decline rather than emit a literal that makes
    // the shift +∞ from inside an algebraic sum.
    decline('a zero-mass truncation declines', _trunc(
      { kind: 'call', op: 'Normal', kwargs: { mu: _lit(0), sigma: _lit(1) } },
      _interval(100, 101)));
    // Not a 1-D interval: an N-D box mass is a Lebesgue volume, and a named set
    // is resolved by another route.
    decline('a cartprod set is not a CDF difference', _trunc(
      { kind: 'call', op: 'Normal', kwargs: { mu: _lit(0), sigma: _lit(1) } },
      { kind: 'call', op: 'cartprod', args: [_interval(-1, 1), _interval(-1, 1)] }));
    decline('a named set declines', _trunc(
      { kind: 'call', op: 'Normal', kwargs: { mu: _lit(0), sigma: _lit(1) } },
      { kind: 'const', name: 'reals' }));
    // A bound that is not a constant leaves `parseTruncationBox` with nothing.
    decline('a latent bound declines', _trunc(
      { kind: 'call', op: 'Normal', kwargs: { mu: _lit(0), sigma: _lit(1) } },
      { kind: 'call', op: 'interval', args: [{ kind: 'ref', ns: 'self', name: 'a' }, _lit(1)] }));
    // `Uniform` carries its bounds inside `support`; anything but an `interval`
    // there, or an unbounded one, has no CDF difference here.
    decline('a Uniform whose support is a named set declines', _trunc(
      { kind: 'call', op: 'Uniform', kwargs: { support: { kind: 'const', name: 'unitinterval' } } },
      _interval(0, 0.5)));
    decline('a Uniform with an unbounded support declines', _trunc(
      { kind: 'call', op: 'Uniform', args: [{ kind: 'call', op: 'interval', args: [_lit(0), { kind: 'const', name: 'inf' }] }] },
      _interval(0, 0.5)));
  });

test('an infinite truncation bound takes the CDF limit, not an evaluation at ±∞',
  async () => {
    // §06's own example is `normalize(truncate(Normal(mu = 0, sigma = 1),
    // interval(0, inf)))`, so the half-open spelling has to work: F(∞) takes the
    // limit 1 and F(0) = 0.5, giving exactly 0.5. Feeding ±∞ to the CDF function
    // instead would give NaN and the arm would decline.
    const N01 = { kind: 'call', op: 'Normal', kwargs: { mu: _lit(0), sigma: _lit(1) } };
    const inf = { kind: 'const', name: 'inf' };
    assert.deepEqual(
      totalMassExpr(_trunc(N01, { kind: 'call', op: 'interval', args: [_lit(0), inf] })),
      _lit(0.5), 'mass over [0, ∞) is 1 − Φ(0) = 0.5');
    // A NEGATIVE infinite bound declines, and the reason is not this arm:
    // `-inf` lowers to `neg(const inf)`, and `mat-density.numericBoundValue`
    // resolves a `lit` or a `const` but not a `neg` of one, so
    // `parseTruncationBox` returns null. The pre-existing
    // `resolveTruncateNormalizers` route defers on the same spelling, so the two
    // agree — pinned here because a future bound resolver that folds `neg`
    // would silently start emitting a literal from this arm too.
    assert.equal(
      totalMassExpr(_trunc(N01, {
        kind: 'call',
        op: 'interval',
        args: [{ kind: 'call', op: 'neg', args: [inf] }, inf],
      })),
      null, 'a neg(inf) bound is not resolvable by parseTruncationBox');
  });

test('the half-open spelling reaches the normalize shift end to end', async () => {
    // The unit test above pins the literal; this one pins that it is the number
    // the density route divides by. §06 truncate makes the component's mass 0.5,
    // so Z(θ) = 0.5θ + 1, and at y = 0.5 the component's unnormalized density
    // is N(0.5; 0, 1) since 0.5 is inside [0, ∞).
    const theta = 0.6;
    const src = 'flatppl_compat = "0.1"\n'
      + 'theta ~ Beta(2.0, 2.0)\n'
      + 'mixdir = normalize(superpose('
      + 'weighted(theta, truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, inf))), '
      + 'weighted(1.0, Normal(mu = 10.0, sigma = 1.0))))\n'
      + 'y ~ iid(mixdir, 1)\n'
      + 'forward_kernel = kernelof(record(y = y), theta = theta)\n'
      + 'L = likelihoodof(forward_kernel, record(y = [0.5]))\n'
      + '__score__ = logdensityof(L, record(theta = ' + theta.toFixed(6) + '))\n';
    // log((0.6·N(0.5;0,1) + N(0.5;10,1)) / (0.6·0.5 + 1)) — mpmath, 40 digits.
    const exact = -1.8171284214381545;
    const mass1 = -2.0247677862163990;   // if Z were 0.6 + 1
    const got = await scoreOf(src);
    assert.ok(Math.abs(got - exact) < 1e-12,
      `engine ${got} vs closed form ${exact} (Δ ${Math.abs(got - exact)})`);
    assert.ok(Math.abs(got - mass1) > 0.15, `engine ${got} is the mass-1 value ${mass1}`);
  });
