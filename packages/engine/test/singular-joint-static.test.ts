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
// The criterion is "determined by the others GIVEN THE SHARED ANCESTORS", not
// "shares an ancestor". `singular-joint.ts` fires only when two components are
// deterministic functions of the same single CONTINUOUS SCALAR draw, which is the
// |S| = 2 case of Hall's condition on the component→noise-root graph. The
// NOT-SINGULAR table below is the half of this file that pins that distinction,
// and it is the more important half: a false positive rejects a model the engine
// answers exactly, and the user cannot work around a compile error.
//
// Two properties are asserted TOGETHER for the pair-rule shapes: the STATIC
// diagnostic, and the runtime refusal that accompanies it. Neither alone is §06:
// the diagnostic makes the error visible, and only the refusal makes the query
// reach no number. A test that checked the diagnostic alone would pass against an
// engine that then scored the joint anyway — which is exactly the state the
// INHERITED shapes are in (see that section, and the WILL-FLIP marker on it).

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

// Filtered to this pass's own messages: some fixtures below deliberately carry an
// unrelated type error (the broadcast-arity ones), and pinning those is another
// test's job.
function singularErrorsOf(src: string): string[] {
  return errorsOf(src).filter((m) => m.startsWith('singular joint:'));
}

// ════════════════════════════════════════════════════════════════════
// SINGULAR by the pair rule — §06's two named classes
// ════════════════════════════════════════════════════════════════════
//
// Every one of these is two deterministic functions of ONE continuous scalar
// draw, so the pair's support is a curve in R² — 2-D Lebesgue-null, hence no
// density w.r.t. the product reference measure.

const SINGULAR_PAIR = [
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
    // `cat` law". Same measure, so the same verdict — a spelling must not decide
    // whether the engine catches this.
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
    // A transform CHAIN, not a single op: the noise-root walk must be transitive,
    // or an intermediate binding hides the shared draw.
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

for (const c of SINGULAR_PAIR) {
  test(`STATIC: ${c.label} is a static error on the density query`, () => {
    const errs = singularErrorsOf(c.src);
    assert.equal(errs.length, 1, 'expected exactly one singular-joint error, got: '
      + JSON.stringify(errs));
    assert.match(errs[0], new RegExp("components '" + c.components[0]
      + "' and '" + c.components[1] + "'"));
    assert.match(errs[0], new RegExp("the single draw '" + c.ancestor + "'"));
    assert.match(errs[0], /static error \(spec §06 "Singular joints"\)/);
  });

  test(`BACKSTOP: ${c.label} still refuses at density time`, async () => {
    const { ctx } = ctxFor(c.src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'),
      /no density w\.r\.t\. the product reference measure|joint law is singular/);
  });
}

// ════════════════════════════════════════════════════════════════════
// SINGULAR by inheritance — and the runtime does NOT refuse these
// ════════════════════════════════════════════════════════════════════
//
// Singularity propagates outward: `iid(M, size)` is the product measure M^⊗N (§06
// `iid`, which "never shares nodes between copies", so the copies are independent
// and the product is null exactly when M is), and a component that is itself a
// singular joint drags the outer joint down with it.
//
// These three are the reason the "a miss still refuses at density time" claim was
// wrong. At 61c29f0 each scored a finite number, oracled below against
// independent closed forms, for a law that has none. The static diagnostic now
// makes them VISIBLE, but a diagnostic is not a gate: the runtime still returns
// the number. The WILL-FLIP tests pin that honestly, so fixing the underlying
// density bug turns a red test rather than drifting silently.

const L = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;

const SINGULAR_INHERITED = [
  {
    label: 'iid over a singular joint',
    reason: /it is an iid product over a singular joint/,
    // Four independent standard normals at 0.1, 0.2, 0.3, 0.4 — i.e. the engine
    // scores iid(S, 2) as if S's two coordinates were independent. The true law
    // concentrates on {(a₁,a₁,a₂,a₂)} ⊂ R⁴ and has no density at all.
    wrongValue: L(0.1) + L(0.2) + L(0.3) + L(0.4),
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
I = iid(S, 2)
ld = logdensityof(I, table(a = [0.1, 0.3], b = [0.2, 0.4]))
`,
  },
  {
    label: 'a nested singular joint component',
    reason: /component 'inner' is itself a singular joint/,
    // Three independent standard normals at 0.1, 0.2, 0.3. True support is
    // {(u, u, t)} ⊂ R³ — 2-dimensional, R³-Lebesgue-null.
    wrongValue: L(0.1) + L(0.2) + L(0.3),
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
N = joint(inner = S, c = lawof(t))
ld = logdensityof(N, record(inner = record(a = 0.1, b = 0.2), c = 0.3))
`,
  },
  {
    label: 'a nested singular record law',
    reason: /component 'inner' is itself a singular joint/,
    wrongValue: L(0.1) + L(0.2) + L(0.3),
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(inner = record(a = y, b = y), c = t))
ld = logdensityof(R, record(inner = record(a = 0.1, b = 0.2), c = 0.3))
`,
  },
];

for (const c of SINGULAR_INHERITED) {
  test(`STATIC: ${c.label} is a static error`, () => {
    const errs = singularErrorsOf(c.src);
    assert.equal(errs.length, 1, 'expected exactly one singular-joint error, got: '
      + JSON.stringify(errs));
    assert.match(errs[0], c.reason);
  });

  test(`[WILL-FLIP] ${c.label} still SCORES a wrong number at runtime — the `
    + 'static diagnostic is not a gate', async () => {
    // Pinning a value known to be WRONG, deliberately and only here: the law has
    // no density, so any finite answer is wrong. It is pinned because the
    // alternative — asserting nothing — is how the hole stayed invisible. When
    // the runtime learns to refuse these, this test goes red and should be
    // rewritten to `assert.rejects`, exactly like the pair-rule BACKSTOP tests.
    const { ctx } = ctxFor(c.src, 1);
    const m: any = await ctx.getMeasure('ld');
    assert.ok(Math.abs(m.samples[0] - c.wrongValue) < 1e-12,
      `expected the known-wrong value ${c.wrongValue}, got ${m.samples[0]}`);
  });
}

// ════════════════════════════════════════════════════════════════════
// NOT SINGULAR — the predicate must stay silent
// ════════════════════════════════════════════════════════════════════
//
// Grouped by WHY each is absolutely continuous, because an earlier revision of
// this pass fired on the whole "shares a noise root" class and the table that was
// supposed to rule out false positives had a structural blind spot: every case in
// it terminated each component at a DISTINCT draw binding, so no case had unequal
// root sets, a strict-superset relation, or a multi-coordinate draw.

const NOT_SINGULAR = [
  // ── each component carries its own noise ──────────────────────────────
  {
    label: 'components sharing an ancestor, each with its own draw (§06 correlated case)',
    src: `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
S = joint(a = lawof(a), b = lawof(b))
ld = logdensityof(S, record(a = 0.5, b = 0.7))
`,
  },
  {
    // The same law as the previous case, written as an explicit deterministic
    // transform instead of two `~`-draws. (u, v) is Gaussian with covariance
    // [[2,1],[1,2]], determinant 3 — full rank, absolutely continuous. The
    // verdict must not depend on the spelling, which is what fired before:
    // root sets are {y,n1} and {y,n2}, which overlap but are not equal.
    label: 'shared ancestor plus per-component explicit noise (cov det 3, full rank)',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
n1 ~ Normal(mu = 0.0, sigma = 1.0)
n2 ~ Normal(mu = 0.0, sigma = 1.0)
u = y + n1
v = y + n2
S = joint(a = lawof(u), b = lawof(v))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`,
  },
  {
    label: 'the record spelling of the same full-rank shape',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
n1 ~ Normal(mu = 0.0, sigma = 1.0)
n2 ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(a = y + n1, b = y + n2))
ld = logdensityof(R, record(a = 0.5, b = 0.9))
`,
  },
  {
    // Root sets {y} and {y,n}: a STRICT SUBSET, not equality. `b` is not
    // determined by `a` — it carries `n` on top. Covariance [[1,1],[1,2]],
    // determinant 1, full rank.
    label: 'the draw itself beside a noisy transform of it (strict-subset roots)',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
n ~ Normal(mu = 0.0, sigma = 1.0)
w = y + n
S = joint(a = lawof(y), b = lawof(w))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`,
  },
  // ── the shared root is one draw, but not a continuous scalar ──────────
  {
    // Root identity is per-draw-BINDING, so it cannot see that the two
    // components read different COORDINATES of `v`. Under identity covariance
    // these are two independent standard normals — full rank. Excluded by the
    // scalar requirement.
    label: 'two coordinates of one multivariate draw',
    src: `
v ~ MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]])
a = v[1]
b = v[2]
S = joint(a = lawof(a), b = lawof(b))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`,
  },
  {
    // The deepest of the exclusions. `(k, k)` for a Bernoulli draw is NOT
    // singular: its reference measure is counting ⊗ counting, and the diagonal
    // of {0,1}² is not null w.r.t. counting measure — the law has a perfectly
    // good pmf (p at (1,1), 1-p at (0,0)). Lebesgue-nullity is a
    // continuous-support argument and does not transfer to a discrete draw.
    label: 'a duplicated BERNOULLI draw (counting reference measure, has a pmf)',
    src: `
k ~ Bernoulli(p = 0.3)
S = joint(a = lawof(k), b = lawof(k))
ld = logdensityof(S, record(a = 1.0, b = 1.0))
`,
  },
  {
    label: 'a duplicated POISSON draw (integer support, has a pmf)',
    src: `
c ~ Poisson(rate = 2.0)
S = joint(a = lawof(c), b = lawof(c))
ld = logdensityof(S, record(a = 1.0, b = 1.0))
`,
  },
  // ── no shared root at all ─────────────────────────────────────────────
  {
    // §06 "Joint composition": "A component contributes a fresh coordinate".
    // `q` is a CONSTRUCTOR whose parameter reaches a draw, so the two
    // coordinates are conditionally independent given z — the compound law, not
    // the singular diagonal. This is the wrong refusal js #143 fixed at runtime;
    // the static gate must not reintroduce it.
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
    // One field is no joint at all: there is no second component for the variate
    // to be determined by, and the single-field pushforward density is scored
    // exactly (#260 (d)).
    label: 'a single-field record law (no second component to be singular against)',
    src: `
sigma2 ~ Exponential(rate = 1.0)
prior = lawof(record(sigma = sqrt(sigma2)))
lp = logdensityof(prior, record(sigma = 1.5))
`,
  },
  // ── the inheritance rule must not over-reach ──────────────────────────
  {
    label: 'iid over a LEGAL (correlated but full-rank) joint',
    src: `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
S = joint(a = lawof(a), b = lawof(b))
I = iid(S, 2)
ld = logdensityof(I, table(a = [0.1, 0.3], b = [0.2, 0.4]))
`,
  },
  {
    label: 'iid over a scalar law (a plain product measure)',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
I = iid(lawof(y), 2)
ld = logdensityof(I, [0.1, 0.3])
`,
  },
];

for (const c of NOT_SINGULAR) {
  test(`NOT SINGULAR: ${c.label}`, () => {
    assert.deepEqual(singularErrorsOf(c.src), []);
  });
}

// The full-rank shapes the static pass now (correctly) stays silent on are STILL
// refused by clm's coarser runtime check, with a reason that is factually wrong
// for them. Pinned so the divergence between the two predicates is visible rather
// than folklore, and so narrowing clm later flips a test here.
test('clm still refuses the full-rank shapes the static pass declines (divergence '
  + 'is deliberate, and clm\'s stated reason is wrong for them)', async () => {
  const { ctx } = ctxFor(`
y ~ Normal(mu = 0.0, sigma = 1.0)
n1 ~ Normal(mu = 0.0, sigma = 1.0)
n2 ~ Normal(mu = 0.0, sigma = 1.0)
u = y + n1
v = y + n2
S = joint(a = lawof(u), b = lawof(v))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`, 1);
  // A refusal returns no number, so this is sound-but-incapable — which is why
  // narrowing clm was left out of scope: it would hand these shapes to a scoring
  // path nobody has verified.
  await assert.rejects(async () => ctx.getMeasure('ld'), /share the ancestor 'y'/);
});

// ── (2) sampling stays legal ───────────────────────────────────────────────

test('§06 "Sampling is well-defined": a singular joint that is only SAMPLED gets '
  + 'no diagnostic', () => {
  // The joint binding is identical to the first SINGULAR_PAIR case; only the
  // `logdensityof` query is gone. A gate that fired on the joint itself would
  // make a legal model un-writable.
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

// ── query surfaces ────────────────────────────────────────────────────────

test('the 3-arg broadcast(logdensityof, M, pts) surface is gated', () => {
  assert.equal(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
pts = [0.1, 0.2]
g = broadcast(logdensityof, S, pts)
`).length, 1);
});

test('the fn(logdensityof(M, _))-body surface under broadcast is gated', () => {
  // This fixture ALSO carries an unrelated arity error — a record-domain joint
  // cannot be broadcast over a vector of scalar points — so `singularErrorsOf`
  // filters to this pass's own message. The point being pinned is only that the
  // walk reaches a `logdensityof` nested inside an `fn` body, which the module
  // header claims; the surrounding model is not asserted valid.
  assert.equal(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
pts = [0.1, 0.2]
g = broadcast(fn(logdensityof(S, _)), pts)
`).length, 1);
});

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
  // The `logdensityof` line — NOT the joint's own line (1-based, and the leading
  // newline is not a line of its own). The location matters: the joint is legal,
  // so marking its binding would point the user at the wrong line.
  assert.equal(d.loc.start.line, 3);
});

test('a three-component joint on one draw reports ONE error, not one per pair', () => {
  assert.equal(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y), c = lawof(y))
ld = logdensityof(S, record(a = 0.5, b = 0.9, c = 0.1))
`).length, 1);
});

test('two density queries in SEPARATE bindings each report', () => {
  assert.equal(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
ld1 = logdensityof(S, record(a = 0.5, b = 0.9))
ld2 = logdensityof(S, record(a = 1.5, b = 1.9))
`).length, 2);
});

test('two density queries in ONE binding each report (reporting is per query, '
  + 'not per binding)', () => {
  // The earlier revision broke out of the per-binding query loop after the first
  // hit, so this reported one error and the second singular joint was invisible.
  assert.equal(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
T = joint(a = lawof(t), b = lawof(t))
ld = logdensityof(S, record(a = 0.5, b = 0.9)) + logdensityof(T, record(a = 0.5, b = 0.9))
`).length, 2);
});

// ── detection is per-PAIR, not per-joint ─────────────────────────────────

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
  assert.match(errs[0], /the single draw 'y'/);
});

// ── the fail-silent contract ──────────────────────────────────────────────

test('an unclassifiable measure argument yields no diagnostic rather than a guess',
  () => {
    // `kchain` is not a joint, so `_componentsOf` declines it. A pass that
    // guessed here would risk refusing a legal model; declining leaves the
    // runtime to judge it. Asserting the SILENCE documents the direction the
    // pass fails in.
    assert.deepEqual(singularErrorsOf(`
z ~ Normal(mu = 0.0, sigma = 1.0)
K = kernelof(Normal(mu = z, sigma = 1.0), z = z)
C = kchain(lawof(z), K)
ld = logdensityof(C, 0.5)
`), []);
  });

test('a Hall deficiency needing a subset of size 3 is a MISS (documented limit)', () => {
  // Support is {(y, t, y)} ⊂ R³ — 2-dimensional, so genuinely singular. No PAIR
  // of components has equal singleton root sets ({y,t} vs {y}), so the |S| = 2
  // rule cannot see it. Pinned as a known miss rather than left undocumented; if
  // the general Hall check ever lands, this test flips to expecting an error.
  assert.deepEqual(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(inner = record(a = y, b = t), c = y))
ld = logdensityof(R, record(inner = record(a = 0.1, b = 0.2), c = 0.1))
`), []);
});
