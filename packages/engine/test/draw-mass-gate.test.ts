'use strict';

// =====================================================================
// draw-mass-gate.test.ts — `draw` requires a probability measure (§04)
// =====================================================================
//
// Pins `typeinfer.checkDrawMass` and the sum-to-one proof that feeds it
// (`superposeIsProvablyNormalized`).
//
// The normative rule, `flatppl-design` `docs/04-design.md`, "Reification":
//
//   "`x ~ m` (equivalent to `x = draw(m)`) introduces a stochastic node
//    `x` by drawing a variate from a normalized measure (i.e. a
//    probability measure) `m`."
//
// So a draw from a measure whose mass class is not `normalized` is a
// static error, and the engine refuses instead of normalizing quietly —
// implicit normalization would make a model's meaning depend on a step
// the user never wrote. The rule is **reject unless proven normalized, or
// not yet inferred**: `unknown` is rejected without anything having been
// proven about it, while `deferred` (a gap in mass inference) passes.
//
// The rejection set mirrors flatppl-rust's `draw_mass_gate` controls, and
// the two sum-to-one readings mirror `literal_weights_sum_to_one` and
// `complement_pair`. The engines are not each other's oracle; the shared
// anchor is §04 above and the §06 measure algebra each proof cites.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

const H = 'flatppl_compat = "0.1"\n';
const N0 = 'Normal(mu = 0.0, sigma = 1.0)';
const N1 = 'Normal(mu = 1.0, sigma = 1.0)';

function errorsOf(src: string): string[] {
  return (processSource(H + src + '\n').diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

function gateErrors(src: string): string[] {
  return errorsOf(src).filter((m) => m.startsWith('draw requires'));
}

function massOf(src: string, name: string): any {
  const b = processSource(H + src + '\n').loweredModule.bindings.get(name);
  return b && b.inferredType && b.inferredType.kind === 'measure'
    ? b.inferredType.mass : undefined;
}

// A superpose of `n` equal literal weights over distinct components.
function equalWeights(weight: string, n: number): string {
  const parts = [];
  for (let i = 0; i < n; i++) {
    parts.push(`weighted(${weight}, Normal(mu = ${i}.0, sigma = 1.0))`);
  }
  return 'm = superpose(' + parts.join(', ') + ')\nx ~ m';
}

// ---------------------------------------------------------------------
// Accepted: proven normalized, or not yet inferred
// ---------------------------------------------------------------------

test('§04: a draw from a distribution is accepted', () => {
  assert.deepEqual(gateErrors(`x ~ ${N0}`), []);
  assert.deepEqual(gateErrors(`x = draw(${N0})`), []);
});

test('§04: normalize(...) is the escape the diagnostic names', () => {
  assert.deepEqual(gateErrors(`m = normalize(weighted(2.0, ${N0}))\nx ~ m`), []);
  assert.equal(massOf(`m = normalize(weighted(2.0, ${N0}))`, 'm'), 'normalized');
});

test('§04: a shape with no mass rule yet (deferred) is accepted', () => {
  // `deferred` is "not yet inferred", which must not be reported as an
  // error on a well-formed model. See `DEFERRED_BASE` for why this shape
  // is the rule-less one.
  const src = DEFERRED_BASE + 'm = pr\nx ~ m';
  assert.deepEqual(errorsOf(src), []);
  assert.equal(massOf(src, 'm'), 'deferred');
});

test('§04: a broadcast draw over a distribution head is accepted', () => {
  const src = 'means = [1.0, 2.0, 3.0]\ny ~ Normal.(means, 1.0)';
  assert.deepEqual(gateErrors(src), []);
});

// ---------------------------------------------------------------------
// A reweighting of a base with NO class: `deferred` rides out only when
// the weight is provably the identity scale
// ---------------------------------------------------------------------
//
// Total mass is w · mass(base), so proving it one needs mass(base) = 1/w,
// which no rule establishes for a base that has no class. A weight other
// than one therefore settles "not provably normalized" whatever the base
// turns out to be. This shipped the other way — the class rode a
// `deferred` base as if the scale did not matter — and the gate passed
// every spelling below.

// `pr` is a measure with NO mass class. A measure-`relabel` over a NAMED
// base survives lowering as a `relabel` node, and no rule in this pass
// classifies `relabel`, so its class is `deferred` — "not yet inferred".
// (The base has to be a separate binding: `relabel(joint(…), […])` written
// inline folds into a keyword `joint`, which the product rule then classifies
// `normalized`.)
//
// These tests used `kchain`/`jointchain` for this until those gained mass
// rules of their own (§06 dependent composition), at which point the same
// sources classified `normalized` and stopped exercising the no-class path.
// If `relabel` ever gains a mass rule, re-point this at whatever shape is
// then genuinely rule-less — do NOT relax the assertions below, since the
// behaviour under test is the weight rule's treatment of an unclassified
// base, not anything about the base's own op.
const DEFERRED_BASE = 'j = joint(Normal(mu = 0.0, sigma = 1.0), '
  + 'Beta(alpha = 1.0, beta = 1.0))\n'
  + 'pr = relabel(j, ["a", "b"])\n';

const RESCALED_DEFERRED: [string, string][] = [
  // Four DISTINCT cases: each weight op above and below its identity scale
  // (1 for `weighted`, 0 for `logweighted`), since the rule turns on "provably
  // the identity" rather than on the direction of the rescaling.
  ['weighted(2.0, <no class>)', 'm = weighted(2.0, pr)'],
  ['weighted(0.5, <no class>)', 'm = weighted(0.5, pr)'],
  ['logweighted(2.0, <no class>)', 'm = logweighted(2.0, pr)'],
  ['logweighted(-1.0, <no class>)', 'm = logweighted(-1.0, pr)'],
];

for (const [label, body] of RESCALED_DEFERRED) {
  test(`§04 refuses a draw from ${label} — a rescaled base with no class`, () => {
    const src = DEFERRED_BASE + body + '\nx ~ m';
    assert.equal(massOf(src, 'm'), 'unknown');
    const errs = gateErrors(src);
    assert.equal(errs.length, 1, `expected one gate error, got ${errs.length}`);
    assert.match(errs[0], /total mass is %unknown/);
  });
}

test('§06: the identity scale DOES ride a base with no class', () => {
  // `weighted(1, M)` and `logweighted(0, M)` are both dν = dM, so they
  // leave the base's class alone — including "not yet inferred", which the
  // gate passes. This is the control that makes the four refusals above
  // mean "the scale settled it" rather than "this base is refused".
  for (const body of [
    'm = weighted(1.0, pr)',
    'm = logweighted(0.0, pr)',
    'm = pr',
  ]) {
    const src = DEFERRED_BASE + body + '\nx ~ m';
    assert.equal(massOf(src, 'm'), 'deferred', body);
    assert.deepEqual(gateErrors(src), [], body);
  }
});

test('§06: a fixed scalar BINDING as the weight is not provably one', () => {
  // `isFixedScalarWeight` accepts a ref to a fixed scalar binding, but this
  // pass does not evaluate it, so `w` is not proven to be the identity
  // scale even when it happens to be 1.0 in the source.
  for (const w of ['2.0', '1.0']) {
    const src = DEFERRED_BASE + `w = ${w}\nm = weighted(w, pr)\nx ~ m`;
    assert.equal(massOf(src, 'm'), 'unknown', w);
    assert.equal(gateErrors(src).length, 1, w);
  }
});

test('§06: a rescaled base WITH a class keeps its existing rule', () => {
  // The change above touches only the no-class fallthrough: a classified
  // base still demotes to finite (or stays locally finite) exactly as
  // before, for any fixed scalar weight including one.
  const N = `Normal(mu = 0.0, sigma = 1.0)`;
  assert.equal(massOf(`m = weighted(2.0, ${N})`, 'm'), 'finite');
  assert.equal(massOf(`m = weighted(1.0, ${N})`, 'm'), 'finite');
  assert.equal(massOf('m = weighted(2.0, Lebesgue(support = reals))', 'm'), 'locallyfinite');
});

test('§04: a broadcast over a locally-finite cell measure is refused', () => {
  // The cell classifies through the reified head's body, so a broadcast
  // of an unbounded reference measure is not a probability measure. The
  // product of locally-finite cells has no class in the lattice, so this
  // lands on `unknown` rather than `locallyfinite`.
  const src = 'ss = [1.0, 2.0]\ns = elementof(posreals)\n'
    + 'K = functionof(Lebesgue(support = interval(0.0, s)), s = s)\nm = K.(ss)\ny ~ m';
  assert.equal(massOf(src, 'm'), 'unknown');
  assert.equal(gateErrors(src).length, 1);
});

// ---------------------------------------------------------------------
// Sum-to-one reading (a): literal weights, DECIMAL-exact
// ---------------------------------------------------------------------

test('§06: a literal mixture whose decimals sum to one is normalized', () => {
  const src = `m = superpose(weighted(0.3, ${N0}), weighted(0.7, ${N1}))\nx ~ m`;
  assert.equal(massOf(src, 'm'), 'normalized');
  assert.deepEqual(gateErrors(src), []);
});

test('§06: ten 0.1 weights are accepted, where an f64 fold would refuse', () => {
  // The engine must not inherit f64 addition's verdict: summing ten
  // 0.1 doubles gives 0.9999999999999999, but the model says 1.
  let f64 = 0;
  for (let i = 0; i < 10; i++) f64 += 0.1;
  assert.notEqual(f64, 1, 'premise: the f64 fold of ten 0.1 is not 1');
  const src = equalWeights('0.1', 10);
  assert.equal(massOf(src, 'm'), 'normalized');
  assert.deepEqual(gateErrors(src), []);
});

test('§06: three 0.3333333333333333 weights are refused, where an f64 fold would accept', () => {
  // The opposite direction of the same disagreement: two roundings put
  // the f64 fold exactly on 1.0, while the declared decimals sum to
  // 0.9999999999999999. The written model is not a mixture.
  const w = 0.3333333333333333;
  assert.equal(w + w + w, 1, 'premise: the f64 fold of three 0.3333333333333333 IS 1');
  const src = equalWeights('0.3333333333333333', 3);
  assert.equal(massOf(src, 'm'), 'finite');
  assert.equal(gateErrors(src).length, 1);
});

test('§06: integer literal weights sum exactly (1 + 0 is a mixture, 1 + 1 is not)', () => {
  assert.equal(massOf(`m = superpose(weighted(1, ${N0}), weighted(0, ${N1}))`, 'm'),
    'normalized');
  assert.equal(massOf(`m = superpose(weighted(1, ${N0}), weighted(1, ${N1}))`, 'm'),
    'finite');
});

test('§06: a negative weight is refused AT THE WEIGHT — a signed combination is '
  + 'not a mixture', () => {
  // The declared decimals do sum to one; the component with weight -0.5 is
  // not a measure, so the sum is not a mixture.
  //
  // This used to arrive as a draw-mass refusal: the sum-to-one proof declined
  // the negative weight, `m` fell to mass `finite`, and the gate reported that
  // `x ~ m` needs a probability measure. §06 refuses the WEIGHT itself now
  // ("f is a non-negative weight"), which is both earlier and the actual
  // diagnosis — the reader was previously told the mixture was unnormalized,
  // not that a weight was negative. `weighted` then has no measure type, so
  // there is no mass class left to classify and no second refusal to make.
  const src = `m = superpose(weighted(-0.5, ${N0}), weighted(1.5, ${N1}))\nx ~ m`;
  const sign = errorsOf(src).filter((mm) => /non-negative weight/.test(mm));
  assert.ok(sign.length >= 1, `want a sign refusal, got ${JSON.stringify(errorsOf(src))}`);
  assert.match(sign[0], /-0\.5/);
  assert.equal(massOf(src, 'm'), undefined);
});

test('§06: an unnormalized component defeats the literal proof', () => {
  const src = `m = superpose(weighted(0.3, Lebesgue(support = interval(0.0, 1.0))), `
    + `weighted(0.7, ${N1}))\nx ~ m`;
  assert.equal(massOf(src, 'm'), 'finite');
  assert.equal(gateErrors(src).length, 1);
});

test('§06: the keyword spelling of weighted is declined, not proven', () => {
  // §04 admits `weighted(weight = w, base = m)`; the proof reads the
  // positional spelling only, so this stays unproven rather than wrong.
  const src = `m = superpose(weighted(weight = 0.3, base = ${N0}), `
    + `weighted(weight = 0.7, base = ${N1}))`;
  assert.notEqual(massOf(src, 'm'), 'normalized');
});

test('§06: logweighted is not proven — its weight is exp(logweight)', () => {
  const src = `m = superpose(logweighted(0.3, ${N0}), logweighted(0.7, ${N1}))`;
  assert.notEqual(massOf(src, 'm'), 'normalized');
});

// ---------------------------------------------------------------------
// Sum-to-one reading (b): the complement pair {e, 1 - e}
// ---------------------------------------------------------------------

test('§06: the complement pair over one binding with unit-interval support is normalized', () => {
  const src = 'psi ~ Beta(1.5, 1.5)\n'
    + `m = superpose(weighted(psi, ${N0}), weighted(1 - psi, Dirac(0.0)))\nx ~ m`;
  assert.equal(massOf(src, 'm'), 'normalized');
  assert.deepEqual(gateErrors(src), []);
});

test('§06: the complement pair is proven in either argument order', () => {
  const src = 'psi ~ Beta(1.5, 1.5)\n'
    + `m = superpose(weighted(1 - psi, Dirac(0.0)), weighted(psi, ${N0}))`;
  assert.equal(massOf(src, 'm'), 'normalized');
});

test('§06: a complement weight outside [0, 1] is not proven', () => {
  // `1 - e` sums to one whatever `e` is, but an unconstrained `e` makes
  // a component a signed measure rather than part of a mixture.
  const src = 'e = elementof(reals)\n'
    + `m = superpose(weighted(e, ${N0}), weighted(1 - e, Dirac(0.0)))\nx ~ m`;
  assert.equal(massOf(src, 'm'), 'unknown');
  assert.equal(gateErrors(src).length, 1);
});

test('§06 UNSOUNDNESS CONTROL: two inline duplicate draws are two coordinates', () => {
  // Structural equality is NOT value identity. Each `draw` is a fresh
  // stochastic coordinate (§04), so these two syntactically identical
  // subtrees are independent: `w1 + (1 - w2) = 1` holds only on a
  // probability-zero event. Proving this pair normalized would lower it
  // as a law with no normalizer — a silently wrong number.
  const U = 'draw(Uniform(support = interval(0.0, 1.0)))';
  const src = `m = superpose(weighted(${U}, ${N0}), weighted(1 - ${U}, Dirac(0.0)))\nx ~ m`;
  assert.notEqual(massOf(src, 'm'), 'normalized');
  assert.equal(gateErrors(src).length, 1);
});

test('§06 UNSOUNDNESS CONTROL: the same duplicate spelling via `~` bindings', () => {
  // The legitimate spelling, for contrast with the case above: the draw
  // sits in `psi`'s own binding, both compared subtrees are the ref
  // `psi`, and one binding is one coordinate.
  const src = 'psi ~ Uniform(support = interval(0.0, 1.0))\n'
    + `m = superpose(weighted(psi, ${N0}), weighted(1 - psi, Dirac(0.0)))\nx ~ m`;
  assert.equal(massOf(src, 'm'), 'normalized');
  assert.deepEqual(gateErrors(src), []);
});

test('§06: two DIFFERENT bindings holding equal draws are not a complement pair', () => {
  const src = 'psi ~ Beta(1.5, 1.5)\nphi ~ Beta(1.5, 1.5)\n'
    + `m = superpose(weighted(psi, ${N0}), weighted(1 - phi, Dirac(0.0)))\nx ~ m`;
  assert.notEqual(massOf(src, 'm'), 'normalized');
  assert.equal(gateErrors(src).length, 1);
});

test('§06: structural equality of a complement weight covers keyword arguments', () => {
  // `f(z = eta)` on both sides: the compared subtrees must agree on the
  // keyword NAMES as well as their values. A user call's value set is not
  // resolved to the unit interval, so neither spelling is proven — what
  // is pinned here is that the two do not differ in the comparison.
  const head = 'z = elementof(reals)\nf = functionof(invlogit(z), z = z)\n'
    + 'eta = elementof(reals)\nnu = elementof(reals)\n';
  const same = head + `m = superpose(weighted(f(z = eta), ${N0}), `
    + 'weighted(1 - f(z = eta), Dirac(0.0)))';
  const differs = head + `m = superpose(weighted(f(z = eta), ${N0}), `
    + 'weighted(1 - f(z = nu), Dirac(0.0)))';
  assert.notEqual(massOf(same, 'm'), 'normalized');
  assert.notEqual(massOf(differs, 'm'), 'normalized');
});

test('§06: a complement of a derived expression needs no draw inside it', () => {
  // Both compared subtrees are `invlogit(eta)` — no opaque source is
  // written inside either, so the spelling does determine the value, and
  // invlogit's value set is the unit interval.
  const src = 'eta = elementof(reals)\n'
    + `m = superpose(weighted(invlogit(eta), ${N0}), `
    + 'weighted(1 - invlogit(eta), Dirac(0.0)))';
  assert.equal(massOf(src, 'm'), 'normalized');
});

// ---------------------------------------------------------------------
// Ten refusal controls, one per rejected mass class / shape
// ---------------------------------------------------------------------

const REFUSALS: [string, string, string][] = [
  ['a reference measure with infinite mass',
    'm = Lebesgue(support = reals)\nx ~ m', 'locallyfinite'],
  ['a constant reweighting above one',
    `m = weighted(2.0, ${N0})\nx ~ m`, 'finite'],
  ['a constant reweighting of exactly one — still not a proof of normalization',
    `m = weighted(1.0, ${N0})\nx ~ m`, 'finite'],
  ['a stochastic reweighting',
    `psi ~ Beta(1.5, 1.5)\nm = weighted(psi, ${N0})\nx ~ m`, 'unknown'],
  ['a truncation, which is a sub-measure',
    `m = truncate(${N0}, interval(0.0, 1.0))\nx ~ m`, 'finite'],
  ['a mixture whose literal weights sum to 1.1',
    `m = superpose(weighted(0.3, ${N0}), weighted(0.8, ${N1}))\nx ~ m`, 'finite'],
  ['a superposition with no weights at all',
    `m = superpose(${N0}, ${N1})\nx ~ m`, 'finite'],
  ['three 0.3333333333333333 weights (an f64 fold would accept)',
    equalWeights('0.3333333333333333', 3), 'finite'],
  ['a complement pair whose weight is unconstrained',
    `e = elementof(reals)\nm = superpose(weighted(e, ${N0}), `
      + 'weighted(1 - e, Dirac(0.0)))\nx ~ m', 'unknown'],
  ['a bayesupdate posterior, whose mass is the evidence integral',
    'mu = elementof(reals)\n'
      + 'K = kernelof(Normal(mu = mu, sigma = 1.0), mu = mu)\n'
      + 'L = likelihoodof(K, 0.5)\n'
      + `prior = lawof(draw(${N0}))\nm = bayesupdate(L, prior)\nx ~ m`, 'unknown'],
];

for (const [label, src, expectedMass] of REFUSALS) {
  test(`§04 refuses a draw from ${label}`, () => {
    assert.equal(massOf(src, 'm'), expectedMass);
    const errs = gateErrors(src);
    assert.equal(errs.length, 1, `expected one gate error, got ${errs.length}`);
    assert.match(errs[0], new RegExp('total mass is %' + expectedMass));
    // The diagnostic must name the escape, not merely refuse.
    assert.match(errs[0], /normalize\(\.\.\.\)/);
  });
}

test('§04: the gate fires on `x = draw(m)` exactly as on `x ~ m`', () => {
  const tilde = gateErrors(`m = weighted(2.0, ${N0})\nx ~ m`);
  const call = gateErrors(`m = weighted(2.0, ${N0})\nx = draw(m)`);
  assert.equal(tilde.length, 1);
  assert.deepEqual(call, tilde);
});

test('§04: the gate reads each binding\'s own class, not a shared type slot', () => {
  // Unification shares one type object between `N` and `weighted(2.0, N)`,
  // so a gate reading `meta.type.mass` would refuse this correct draw.
  const src = `N = ${N0}\nm2 = weighted(2.0, N)\nx ~ N`;
  assert.deepEqual(gateErrors(src), []);
  assert.equal(gateErrors(`N = ${N0}\nm = weighted(2.0, N)\nx ~ m`).length, 1);
});

// ---------------------------------------------------------------------
// Corpus acceptance: the zero-inflated binomial
// ---------------------------------------------------------------------

test('zero-inflated-binomial: the psi / 1 - psi mixture types clean through the gate', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'zero-inflated-binomial.flatppl'), 'utf8');
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), []);
  // The mixture's weights are `psi` and `1 - psi` over one Beta-supported
  // binding, so the complement reading proves it a probability measure.
  const zib = r.loweredModule.bindings.get('ZeroInflatedBinomial');
  assert.equal(zib.inferredType.mass, 'normalized');
  // `y ~ iid(ZeroInflatedBinomial, N)`: iid is a homomorphism on the class, so the
  // drawn measure is normalized too and the gate passes it.
  const y = r.loweredModule.bindings.get('y');
  assert.equal(y.rhs.op, 'draw');
  assert.equal(y.rhs.args[0].meta.type.mass, 'normalized');
});

test('zero-inflated-binomial: the mixture density matches an independent oracle, and the weights move it', async () => {
  // Oracle (scipy): Σᵢ log(psi·binom.pmf(yᵢ; 20, p) + (1 − psi)·[yᵢ = 0]) at
  // p = 0.4, psi = 0.7 over y = [7,0,5,8,0,6,4,0,9,3]. The COMPARISON is
  // the point: with the two `weighted` wrappers removed the same model
  // scores the unweighted sum Σᵢ log(binom.pmf(yᵢ) + [yᵢ = 0]), so a
  // density path that dropped the weights would land on the second value.
  const fs = require('node:fs');
  const path = require('node:path');
  const { ctxFor } = require('./_ctx-factory.ts');
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'zero-inflated-binomial.flatppl'), 'utf8');
  const unweighted = src
    .replace('weighted(psi, Binomial(K, p))', 'Binomial(K, p)')
    .replace('weighted(1 - psi, Dirac(0))', 'Dirac(0)');

  const scoreOf = async (model: string, binding: string) => {
    const { ctx } = ctxFor(
      model + `\n__score__ = logdensityof(${binding}, record(p = 0.4, psi = 0.7))\n`, 1);
    const mm = await ctx.getMeasure('__score__');
    return mm.value ? mm.value.data[0] : mm.samples[0];
  };
  const score = (model: string) => scoreOf(model, 'L');

  const weighted = await score(src);
  assert.ok(Math.abs(weighted - (-23.881454058598102)) < 1e-12,
    `weighted mixture logdensity ${weighted}, oracle -23.881454058598102`);
  const bare = await score(unweighted);
  assert.ok(Math.abs(bare - (-17.77295727547568)) < 1e-12,
    `unweighted superposition logdensity ${bare}, oracle -17.77295727547568`);
  assert.ok(Math.abs(weighted - bare) > 6.1,
    'the two spellings must not score the same — the weights are carried');
});

test('zero-inflated-binomial: the posterior carries the Beta(1.5, 1.5) priors', async () => {
  // `L` is prior-independent, so the likelihood score above cannot tell this
  // fixture apart from the stale local copy it replaced (which carried
  // Beta(1, 1) priors — logpdf exactly 0 everywhere, so ITS posterior equalled
  // its likelihood). `posterior = bayesupdate(L, prior)` is what pins them.
  //
  // Oracle (scipy, independent of this engine):
  //   L                              = -23.881454058598102  (pinned above)
  //   beta.logpdf(0.4, 1.5, 1.5)
  //     + beta.logpdf(0.7, 1.5, 1.5) =   0.37554125970846486
  //   posterior = L + that           = -23.505912798889636
  const fs = require('node:fs');
  const path = require('node:path');
  const { ctxFor } = require('./_ctx-factory.ts');
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'zero-inflated-binomial.flatppl'), 'utf8');
  const { ctx } = ctxFor(
    src + '\n__score__ = logdensityof(posterior, record(p = 0.4, psi = 0.7))\n', 1);
  const mm = await ctx.getMeasure('__score__');
  const got = mm.value ? mm.value.data[0] : mm.samples[0];
  assert.ok(Math.abs(got - (-23.505912798889636)) < 1e-12,
    `posterior logdensity ${got}, oracle -23.505912798889636`);
  // The prior contribution is the whole point: a posterior that silently
  // dropped it would land on `L` itself.
  assert.ok(Math.abs(got - (-23.881454058598102)) > 0.37,
    'the posterior must differ from the likelihood — the priors are carried');
});
