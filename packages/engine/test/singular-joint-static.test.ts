'use strict';

// ════════════════════════════════════════════════════════════════════════
// Singular joints: the STATIC gate (spec §06 "Singular joints")
// ════════════════════════════════════════════════════════════════════════
//
// §06 "Singular joints", quoted verbatim from flatppl-design 9e35262:
//
//   "When one component's variate is determined by the others given the shared
//   ancestors (the same draw referenced twice, a deterministic transform of
//   another component), the joint law has no density w.r.t. the product
//   reference measure. Sampling is well-defined; a density query is a static
//   error where statically detectable, and is otherwise refused by the engine."
//
// Three properties, one per sentence-clause, and each is tested here:
//   1. a density query over a detectably singular joint is a STATIC error
//      (`processSource` diagnostics, so an editor marks it with nothing run);
//   2. sampling stays legal — no diagnostic on a model that only samples;
//   3. the runtime refusal survives as the backstop (`clm._refuseIfSingular`).
//
// (1) and (3) are asserted TOGETHER for every singular shape. That pairing is
// the point: the static diagnostic is what makes the error visible, and the
// runtime throw is what makes it prevent a number. Dropping either would leave
// §06 half-implemented, and a test that checked only the diagnostic would pass
// against an engine that then happily scored the singular joint anyway.
//
// The absolutely-continuous neighbours are pinned just as hard. A false positive
// here refuses a model the engine answers exactly (§06 "Equivalent record law"'s
// correlated case, and §06 "Joint composition"'s fresh-coordinate constructor),
// which is a worse failure than a missed diagnostic — a miss still refuses at
// density time.

const test = require('node:test');
const assert = require('node:assert');
const { processSource } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

function errorsOf(src: string): string[] {
  const { diagnostics } = processSource(src);
  return (diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => String(d.message));
}

function singularErrorsOf(src: string): string[] {
  return errorsOf(src).filter((m) => m.startsWith('singular joint:'));
}

// ── (1) + (3): the statically detectable singular shapes ───────────────────
//
// Each entry names the §06 class it instantiates. `ancestor` is the shared draw
// the diagnostic must name — naming it is what makes the message actionable
// rather than just a verdict.

const SINGULAR = [
  {
    label: 'the same draw reified twice',
    ancestor: 'y',
    components: ['a', 'b'],
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`,
  },
  {
    label: 'the equivalent record law of the same draw twice',
    ancestor: 'y',
    components: ['a', 'b'],
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(a = y, b = y))
ld = logdensityof(R, record(a = 0.5, b = 0.9))
`,
  },
  {
    label: 'a deterministic transform of another component',
    ancestor: 'y',
    components: ['a', 'b'],
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
w = exp(y)
T = joint(a = lawof(y), b = lawof(w))
ld = logdensityof(T, record(a = 0.5, b = 0.9))
`,
  },
  {
    label: 'a NAMED reified law used as two components',
    ancestor: 'y',
    components: ['a', 'b'],
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
Ly = lawof(y)
S = joint(a = Ly, b = Ly)
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`,
  },
  {
    // §06 "Equivalent record law": "the positional form is the corresponding
    // `cat` law". Same measure, so the same verdict — a spelling must not
    // decide whether the engine catches this.
    label: 'the positional (cat-law) spelling',
    ancestor: 'y',
    components: ['#1', '#2'],
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(lawof(y), lawof(y))
ld = logdensityof(S, [0.5, 0.9])
`,
  },
  {
    // A transform CHAIN, not a single op: the noise-root walk must be
    // transitive, or an intermediate binding hides the shared draw.
    label: 'a two-hop deterministic chain to the shared draw',
    ancestor: 'y',
    components: ['a', 'b'],
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
u = 2.0 * y
v = exp(u)
T = joint(a = lawof(y), b = lawof(v))
ld = logdensityof(T, record(a = 0.5, b = 0.9))
`,
  },
];

for (const c of SINGULAR) {
  test(`STATIC: ${c.label} is a static error on the density query`, () => {
    const errs = singularErrorsOf(c.src);
    assert.equal(errs.length, 1, 'expected exactly one singular-joint error, got: '
      + JSON.stringify(errs));
    assert.match(errs[0], new RegExp("components '" + c.components[0]
      + "' and '" + c.components[1] + "'"));
    assert.match(errs[0], new RegExp("the same draw '" + c.ancestor + "'"));
    assert.match(errs[0], /static error \(spec §06 "Singular joints"\)/);
  });

  test(`BACKSTOP: ${c.label} still refuses at density time`, async () => {
    const { ctx } = ctxFor(c.src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'),
      /no density w\.r\.t\. the product reference measure|joint law is singular/);
  });
}

// ── (2) sampling stays legal ───────────────────────────────────────────────

test('§06 "Sampling is well-defined": a singular joint that is only SAMPLED gets '
  + 'no diagnostic', () => {
  // The joint binding is identical to the first SINGULAR case; only the
  // `logdensityof` query is gone. A gate that fired on the joint itself would
  // make an legal model un-writable.
  assert.deepEqual(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
`), []);
});

test('§06 "Sampling is well-defined": drawing FROM the singular joint is not a '
  + 'density query either', () => {
  assert.deepEqual(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
r ~ S
`), []);
});

// ── the absolutely-continuous neighbours must stay clean ───────────────────

const LEGAL = [
  {
    // §06 "Equivalent record law": the components share ANCESTOR z but each
    // carries its own noise, so the law is correlated and has a density. This
    // is the case F1's recogniser answers in closed form.
    label: 'components sharing an ancestor but carrying their own noise',
    src: `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
S = joint(a = lawof(a), b = lawof(b))
ld = logdensityof(S, record(a = 0.5, b = 0.7))
`,
  },
  {
    // §06 "Joint composition": "A component contributes a fresh coordinate".
    // `q` is a CONSTRUCTOR whose parameter reaches a draw, so the two
    // coordinates are conditionally independent given z — the compound law,
    // not the singular diagonal. This is the wrong refusal js #143 fixed at
    // runtime; the static gate must not reintroduce it.
    label: 'a named constructor measure used as two components',
    src: `
z ~ Normal(mu = 0.5, sigma = 2.0)
q = Normal(mu = z, sigma = 0.6)
Q = joint(a = q, b = q)
ld = logdensityof(Q, record(a = 0.5, b = 0.9))
`,
  },
  {
    label: 'an ancestor-free constructor used as two components (§04 Identity law)',
    src: `
m = Normal(mu = 0.0, sigma = 1.0)
Q = joint(a = m, b = m)
ld = logdensityof(Q, record(a = 0.5, b = 0.9))
`,
  },
  {
    label: 'a record law over deterministic transforms of DISTINCT draws',
    src: `
a ~ Normal(mu = 0.0, sigma = 1.0)
b ~ Exponential(rate = 1.0)
prior = lawof(record(x = exp(a), y = sqrt(b)))
lp = logdensityof(prior, record(x = 2.0, y = 1.2))
`,
  },
  {
    label: 'a written-out-twice constructor joint over a shared latent',
    src: `
z ~ Normal(mu = 0.5, sigma = 2.0)
Q = joint(a = Normal(mu = z, sigma = 0.6), b = Normal(mu = z, sigma = 0.8))
ld = logdensityof(Q, record(a = 0.5, b = 0.9))
`,
  },
  {
    // One field is no joint at all: there is no second component for the
    // variate to be determined by, and the single-field pushforward density
    // is scored exactly (#260 (d)).
    label: 'a single-field record law (no second component to be singular against)',
    src: `
sigma2 ~ Exponential(rate = 1.0)
prior = lawof(record(sigma = sqrt(sigma2)))
lp = logdensityof(prior, record(sigma = 1.5))
`,
  },
];

for (const c of LEGAL) {
  test(`NO FALSE POSITIVE: ${c.label}`, () => {
    assert.deepEqual(singularErrorsOf(c.src), []);
  });
}

// ── located, and reported once per query ──────────────────────────────────

test('the diagnostic is located on the measure argument of the query', () => {
  const { diagnostics } = processSource(`
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`);
  const d = (diagnostics || []).find((x: any) =>
    String(x.message).startsWith('singular joint:'));
  assert.ok(d && d.loc, 'the diagnostic must carry a location');
  // The `logdensityof` line — NOT the joint's own line (1-based, and the
  // leading newline is not a line of its own). The location matters: the joint
  // is legal, so marking its binding would point the user at the wrong line.
  assert.equal(d.loc.start.line, 3);
});

test('a three-component joint on one draw reports ONE error, not one per pair', () => {
  assert.equal(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y), c = lawof(y))
ld = logdensityof(S, record(a = 0.5, b = 0.9, c = 0.1))
`).length, 1);
});

test('two separate density queries over the same singular joint each report', () => {
  assert.equal(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
ld1 = logdensityof(S, record(a = 0.5, b = 0.9))
ld2 = logdensityof(S, record(a = 1.5, b = 1.9))
`).length, 2);
});

// ── detection is per-PAIR, not per-joint ─────────────────────────────────
//
// A joint may mix a singular pair with an independent third component. The
// overlap test must still fire — an all-or-nothing check keyed on the whole
// component set would miss it.

test('a singular pair beside an independent third component still errors, and '
  + 'names the offending pair', () => {
  const errs = singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), c = lawof(t), b = lawof(y))
ld = logdensityof(S, record(a = 0.5, c = 0.2, b = 0.9))
`);
  assert.equal(errs.length, 1);
  assert.match(errs[0], /components 'a' and 'b'/);
  assert.match(errs[0], /the same draw 'y'/);
});

// ── the fail-silent contract ──────────────────────────────────────────────

test('an unclassifiable measure argument yields no diagnostic rather than a guess',
  () => {
    // `kchain` is not a joint, so `_componentRoots` declines it. A pass that
    // guessed here would refuse a legal model; declining leaves the runtime
    // backstop to judge it. Asserting the SILENCE is what documents the
    // direction the pass fails in.
    assert.deepEqual(singularErrorsOf(`
z ~ Normal(mu = 0.0, sigma = 1.0)
K = kernelof(Normal(mu = z, sigma = 1.0), z = z)
C = kchain(lawof(z), K)
ld = logdensityof(C, 0.5)
`), []);
  });
