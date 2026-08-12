'use strict';

// =====================================================================
// lawof-mass-gate.test.ts — `lawof` requires a probability measure (§04)
// =====================================================================
//
// Pins `typeinfer.checkLawofMass` and the `lawof` arm of the mass pass.
// The gate shares `unprovableNormalization` with the draw gate, so the two
// accept and reject exactly the same classes — see draw-mass-gate.test.ts
// for the class-by-class rationale.
//
// What grounds it. `flatppl-design` `docs/04-design.md`, "Reification to
// measures":
//
//   "`lawof(x)` reifies the ancestor sub-DAG of `x` as the **probability
//    measure** that is the total law of x — the probability measure that
//    `x`, considered as a random variable, is distributed according to."
//
// A VALUE argument therefore needs no gate: the law of a variate is a
// probability measure by construction, and stays `normalized` below.
//
// The gated case is a MEASURE argument, which this engine accepts as the
// identity (`inferLawof`: `lawof(measure) = measure`, §04's "Identity law"
// read in the other direction). AUTHORITY NOTE: §04 as merged does not
// define `lawof` on a measure at all — that extension, and the explicit
// requirement that its argument be `%normalized`, is flatppl-design#73,
// which is still OPEN. So unlike the draw gate this rests on proposed
// normative text. The identity it gates already ships here, so the question
// is not whether to anticipate #73 but whether the shipped path stays sound
// while #73 is pending. If #73 lands differently, these expectations move
// with `inferLawof`.
//
// The bug that motivated it is the last test here: `draw(truncate(N, S))`
// was correctly refused while `draw(lawof(truncate(N, S)))` passed, because
// the `lawof` mass arm claimed `normalized` unconditionally. An identity
// cannot change a total mass, so that claim let any unnormalized measure
// launder its class and walk past the draw gate.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

const H = 'flatppl_compat = "0.1"\n';
const N0 = 'Normal(mu = 0.0, sigma = 1.0)';
const UNIT = 'interval(0.0, 1.0)';

function errorsOf(src: string): string[] {
  return (processSource(H + src + '\n').diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

function lawofErrors(src: string): string[] {
  return errorsOf(src).filter((m) => m.startsWith('lawof requires'));
}

function massOf(src: string, name: string): any {
  const b = processSource(H + src + '\n').loweredModule.bindings.get(name);
  return b && b.inferredType && b.inferredType.kind === 'measure'
    ? b.inferredType.mass : undefined;
}

// ---------------------------------------------------------------------
// Accepted: a value argument, a proven-normalized measure, or no class yet
// ---------------------------------------------------------------------

test('§04: lawof of a VALUE is a probability measure, and is never gated', () => {
  // The law of a variate is normalized by construction — the §04 sentence
  // quoted above. This is the overwhelmingly common spelling in the corpus
  // (`prior = lawof(record(...))`), so it must stay untouched by the gate.
  assert.deepEqual(lawofErrors(`x ~ ${N0}\nm = lawof(x)`), []);
  assert.equal(massOf(`x ~ ${N0}\nm = lawof(x)`, 'm'), 'normalized');

  const rec = `x ~ ${N0}\ny ~ Beta(alpha = 1.0, beta = 1.0)\n`
    + 'm = lawof(record(a = x, b = y))';
  assert.deepEqual(lawofErrors(rec), []);
  assert.equal(massOf(rec, 'm'), 'normalized');
});

test('§04: lawof of a normalized measure is accepted (identity law)', () => {
  assert.deepEqual(lawofErrors(`m = lawof(${N0})`), []);
  assert.equal(massOf(`m = lawof(${N0})`, 'm'), 'normalized');
  // A superposition proven to sum to one is normalized before this is
  // consulted, so it passes too.
  const mix = `m = lawof(superpose(weighted(0.3, ${N0}), `
    + 'weighted(0.7, Normal(mu = 1.0, sigma = 1.0))))';
  assert.deepEqual(lawofErrors(mix), []);
  assert.equal(massOf(mix, 'm'), 'normalized');
});

test('§04: a shape with no mass rule yet (deferred) is accepted', () => {
  // `deferred` is §11's "not yet inferred", not a verdict — rejecting it
  // would turn every gap in mass inference into an error on a well-formed
  // model. A measure-`relabel` over a named base has no rule in this pass.
  const src = 'j = joint(Normal(mu = 0.0, sigma = 1.0), '
    + 'Beta(alpha = 1.0, beta = 1.0))\n'
    + 'r = relabel(j, ["a", "b"])\n'
    + 'm = lawof(r)';
  assert.deepEqual(errorsOf(src), []);
  assert.equal(massOf(src, 'm'), 'deferred');
});

test('§04: normalize(...) is the escape the diagnostic names', () => {
  const src = `b = normalize(truncate(${N0}, ${UNIT}))\nm = lawof(b)`;
  assert.deepEqual(errorsOf(src), []);
  assert.equal(massOf(src, 'm'), 'normalized');
  // Also on a measure that has no finite total mass to begin with.
  const leb = `b = normalize(Lebesgue(support = ${UNIT}))\nm = lawof(b)`;
  assert.deepEqual(errorsOf(leb), []);
  assert.equal(massOf(leb, 'm'), 'normalized');
});

// ---------------------------------------------------------------------
// Refused: every class that is not a proof of normalization
// ---------------------------------------------------------------------

const REFUSED: [string, string, string][] = [
  ['%finite (truncate)', `b = truncate(${N0}, ${UNIT})\nm = lawof(b)`, 'finite'],
  ['%finite (weighted)', `b = weighted(2.0, ${N0})\nm = lawof(b)`, 'finite'],
  ['%locallyfinite (Lebesgue)', 'b = Lebesgue(support = reals)\nm = lawof(b)',
    'locallyfinite'],
];

for (const [label, src, cls] of REFUSED) {
  test(`§04 refuses lawof of a ${label} measure`, () => {
    assert.equal(massOf(src, 'm'), cls);
    const errs = lawofErrors(src);
    assert.equal(errs.length, 1, `expected one gate error, got ${errs.length}`);
    assert.match(errs[0], new RegExp(`total mass is %${cls}`));
    // The diagnostic must name the escape and disclaim any implicit fixing.
    assert.match(errs[0], /normalize\(\.\.\.\)/);
    assert.match(errs[0], /lawof never normalizes its argument/);
  });
}

test('§04 refuses lawof of an %unknown measure (bayesupdate evidence)', () => {
  // `unknown` is rejected without anything having been PROVEN about it: the
  // gate asks whether normalization was established, not whether
  // non-normalization was.
  const src = 'mu = elementof(reals)\n'
    + 'K = kernelof(Normal(mu = mu, sigma = 1.0), mu = mu)\n'
    + 'L = likelihoodof(K, 0.5)\n'
    + `pr = lawof(draw(${N0}))\n`
    + 'b = bayesupdate(L, pr)\n'
    + 'm = lawof(b)';
  assert.equal(massOf(src, 'm'), 'unknown');
  assert.equal(lawofErrors(src).length, 1);
  assert.match(lawofErrors(src)[0], /total mass is %unknown/);
});

test('§04: a KERNEL argument is not this gate\'s business', () => {
  // §04 discusses `lawof` lifting pointwise on a non-nullary kernel, but
  // `inferLawof` accepts only a value or a measure, so a kernel argument is
  // already an ARG-TYPE error one level up. Recorded so that a later change
  // admitting kernels has to decide the mass question deliberately rather
  // than inheriting silence.
  const src = 'mu = elementof(reals)\n'
    + 'K = kernelof(Normal(mu = mu, sigma = 1.0), mu = mu)\n'
    + 'm = lawof(K)';
  assert.deepEqual(lawofErrors(src), []);
  assert.equal(errorsOf(src).length, 1);
  assert.match(errorsOf(src)[0], /lawof expects a value-typed argument/);
});

// ---------------------------------------------------------------------
// The regression: `lawof` must not launder a class past the draw gate
// ---------------------------------------------------------------------

test('§04: lawof cannot launder an unnormalized measure past the draw gate', () => {
  const base = `b = truncate(${N0}, ${UNIT})\n`;
  // Directly drawing from the restriction was always refused.
  assert.equal(errorsOf(base + 'x ~ b').length, 1);
  // Routing it through `lawof` used to pass with NO diagnostic at all: the
  // mass arm claimed `normalized`, so the draw gate saw a probability
  // measure. The identity cannot change the mass, so `m` is `finite` and
  // BOTH gates now fire — `lawof` on its own argument, `draw` on `m`.
  const laundered = base + 'm = lawof(b)\nx ~ m';
  assert.equal(massOf(laundered, 'm'), 'finite');
  const errs = errorsOf(laundered);
  assert.equal(errs.length, 2, `expected both gates to fire, got ${errs.length}`);
  assert.ok(errs.some((e) => e.startsWith('lawof requires')));
  assert.ok(errs.some((e) => e.startsWith('draw requires')));
  // Normalizing once fixes both.
  assert.deepEqual(errorsOf(`b = normalize(truncate(${N0}, ${UNIT}))\n`
    + 'm = lawof(b)\nx ~ m'), []);
});
