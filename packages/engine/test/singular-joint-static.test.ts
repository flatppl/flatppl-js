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
// deterministic functions of ONE shared scalar value AND both components have
// continuous support. The second half is §06 "Reference measure for product
// measures" (verbatim, flatppl-design 9e35262): the reference is built per
// component, "each either `Lebesgue` or `Counting` on the corresponding component
// support" — so nullity is a claim about the COMPONENTS' supports, not about the
// shared draw. Both directions of that distinction are tested below
// (`floor(y)` beside `y`, and two real-typed functions of one Poisson draw).
//
// The NOT-SINGULAR table is the more important half of this file: a false positive
// rejects a model the engine answers exactly, and a user cannot work around a
// compile error.
//
// Two properties are asserted TOGETHER for every singular shape: the STATIC
// diagnostic, and the runtime refusal that accompanies it. Neither alone is §06:
// the diagnostic makes the error visible, and only the refusal makes the query
// reach no number. A test that checked the diagnostic alone would pass against an
// engine that then scored the joint anyway — which is the state the INHERITED
// shapes were in until `clm._refuseIfSingular` learned to recurse through `iid`
// and into nested components.

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
  {
    // §06 "Reference measure for product measures" fixes the reference per
    // COMPONENT support, so what matters is that both components are real-typed —
    // NOT that the shared draw is continuous. Here the draw is DISCRETE and the
    // law is still singular: both components infer `scalar real`, so the reference
    // is Lebesgue ⊗ Lebesgue, and the support {(n, 2n) : n ∈ ℕ₀} is countable and
    // therefore 2-D Lebesgue-null. A draw-keyed continuity test missed this.
    label: 'two real-valued functions of one DISCRETE draw (countable support)',
    ancestor: 'c',
    components: ['a', 'b'],
    src: `
c ~ Poisson(rate = 2.0)
a = c * 1.0
b = c * 2.0
R = lawof(record(a = a, b = b))
ld = logdensityof(R, record(a = 1.0, b = 2.0))
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
// SINGULAR through a shared derived VALUE, not a shared draw
// ════════════════════════════════════════════════════════════════════
//
// The shared generator need not be a draw. Here both components are functions of
// the intermediate value `u`, whose OWN root set has size 2 — so an
// equal-singleton-roots rule missed these, even though the first is §06's "the
// same draw referenced twice" and the third is "a deterministic transform of
// another component" almost verbatim.
//
// These are also the counterexample to reading Hall's condition as an exact rank
// criterion: R₁ = R₂ = {y, n} gives |⋃ Rᵢ| = 2 = |S|, so Hall is satisfied, yet
// the Jacobian has rank 1 because the two component functions are dependent.

const SINGULAR_SHARED_VALUE = [
  {
    label: 'two components that are the SAME expression',
    generator: 'u',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
n ~ Normal(mu = 0.0, sigma = 1.0)
u = y + n
S = joint(a = lawof(u), b = lawof(u))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`,
  },
  {
    label: 'the record spelling of the same-expression pair',
    generator: 'u',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
n ~ Normal(mu = 0.0, sigma = 1.0)
u = y + n
R = lawof(record(a = u, b = u))
ld = logdensityof(R, record(a = 0.5, b = 0.9))
`,
  },
  {
    label: 'one component a deterministic transform of the other',
    generator: 'u',
    // The one shape here that clm's singular check does NOT catch — its inline
    // `lawof(2.0 * u)` component never becomes a named binding in the derivation
    // table, so `_refuseIfSingular` sees no component pair. It still produces no
    // number: the linear-Gaussian recogniser refuses it one step later for being
    // a non-Normal component. Recorded rather than smoothed over, because "which
    // gate stopped it" is the difference between a designed refusal and luck.
    //
    // RISK, and what to do about it: this pins a NON-F5 gate, so widening the
    // recogniser to accept affine components (an obvious future improvement) will
    // redden this test for a reason unrelated to singular joints. The fix then is
    // NOT to loosen this assertion — that would let the shape start scoring
    // unnoticed, and it is genuinely singular. Teach clm to see inline
    // `lawof(2.0 * u)` components first, then re-pin this to the singular
    // refusal. Tracked with the other clm follow-ups in
    // flatppl-dev/TODO-flatppl-js.md.
    backstop: /component 'a' is a 'add', not a Normal/,
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
n ~ Normal(mu = 0.0, sigma = 1.0)
u = y + n
S = joint(a = lawof(u), b = lawof(2.0 * u))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`,
  },
];

for (const c of SINGULAR_SHARED_VALUE) {
  test(`STATIC: ${c.label} is a static error, naming the shared value`, () => {
    const errs = singularErrorsOf(c.src);
    assert.equal(errs.length, 1, 'expected exactly one singular-joint error, got: '
      + JSON.stringify(errs));
    // "value", not "draw" — the generator is a derived binding here, and saying
    // "draw" would name something the user did not write.
    assert.match(errs[0], new RegExp("the single value '" + c.generator + "'"));
  });

  test(`BACKSTOP: ${c.label} still reaches no number at density time`, async () => {
    const { ctx } = ctxFor(c.src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'),
      (c as any).backstop
        || /no density w\.r\.t\. the product reference measure|joint law is singular/);
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
// wrong. At 61c29f0 each SCORED a finite number for a law that has none — the
// value being exactly the product of independent normals, i.e. the singular pair
// scored as two independent coordinates. `clm._refuseIfSingular` read one
// component→binding map, so singularity one level down was invisible to it.
//
// It now recurses through an `iid` derivation to its base measure and into every
// component binding, so all three reach the DESIGNED singular refusal. The
// `backstop` regexes below pin the path clause the recursion adds, since "which
// level was singular" is the difference between a designed refusal and an
// incidental crash.

const SINGULAR_INHERITED = [
  {
    label: 'iid over a singular joint',
    reason: /it is an iid product over a singular joint/,
    // The refusal names the iid's inner measure, then the offending inner pair.
    backstop: /the iid product's inner measure: joint components 'a' and 'b'.*share the ancestor 'y'.*no density w\.r\.t\. the product reference measure/s,
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
    // True support is {(u, u, t)} ⊂ R³ — 2-dimensional, R³-Lebesgue-null.
    backstop: /component 'inner': joint components 'a' and 'b'.*share the ancestor 'y'.*no density w\.r\.t\. the product reference measure/s,
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
    backstop: /component 'inner': joint components 'a' and 'b'.*share the ancestor 'y'.*no density w\.r\.t\. the product reference measure/s,
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

  test(`BACKSTOP: ${c.label} refuses at density time, naming the level that is `
    + 'singular', async () => {
    // Both halves of §06 now hold for the inherited shapes: the static diagnostic
    // makes the error visible AND the query reaches no number. Asserting the path
    // clause, not just any rejection, is what separates the DESIGNED refusal from
    // an incidental crash somewhere further down the density walk.
    const { ctx } = ctxFor(c.src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'), (e: any) => {
      assert.equal(e.code, 'CLM_SINGULAR_JOINT');
      assert.match(String(e.message), c.backstop);
      return true;
    });
  });
}

// ════════════════════════════════════════════════════════════════════
// INHERITANCE reaches every level, and names it
// ════════════════════════════════════════════════════════════════════
//
// The three shapes above are one level deep and keyword-spelled. These are the
// rest of the surface the recursion covers. Each was a finite wrong number or an
// incidental crash before it, and none was pinned.

const INHERITED_SURFACES = [
  {
    label: 'a SINGLE-field outer record whose only field is singular',
    // The sole reason the pair loop's `comps.length < 2` guard had to become a
    // bare `!comps`: a one-field record law is its inner law relabelled, so it is
    // singular whenever the inner is, but no PAIR exists at the outer level.
    // Base scored -1.8628770664093457 (two normals at 0.1, 0.2).
    backstop: /component 'inner': joint components 'a' and 'b'/,
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(inner = record(a = y, b = y)))
ld = logdensityof(R, record(inner = record(a = 0.1, b = 0.2)))
`,
  },
  {
    // §06 "Equivalent record law": "the positional form is the corresponding
    // `cat` law". A tuple derivation nests exactly as a record one does, and the
    // component is labelled by position. Base threw an incidental "cannot consume
    // scalar from value of type object".
    label: 'a POSITIONAL joint nesting a singular joint',
    backstop: /component '#1': joint components 'a' and 'b'/,
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
P = joint(S, lawof(t))
ld = logdensityof(P, record(a = 0.1, b = 0.2))
`,
  },
  {
    // Two `iid` levels, so the path clause repeats. Base threw an incidental
    // "cannot consume named field 'a'".
    label: 'iid of iid over a singular joint',
    backstop: /the iid product's inner measure, inside the iid product's inner measure: joint components 'a' and 'b'/,
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
I = iid(iid(S, 2), 2)
ld = logdensityof(I, table(a = [0.1, 0.3], b = [0.2, 0.4]))
`,
  },
  {
    // Depth 2, and the ONLY test of the path clause's ORDER. It reads
    // innermost-first ("`inner`, inside `deep`"), which is the containment the
    // right way round — an outermost-first join stated it backwards.
    label: 'three levels of nesting',
    backstop: /component 'inner', inside component 'deep': joint components 'a' and 'b'/,
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
r ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(y))
M = joint(inner = S, c = lawof(t))
D = joint(deep = M, d = lawof(r))
ld = logdensityof(D, record(deep = record(inner = record(a = 0.1, b = 0.2), c = 0.3), d = 0.4))
`,
  },
  {
    // `iid` and component nesting compose, and the path clause mixes both kinds.
    label: 'iid over a nested singular record law',
    backstop: /component 'inner', inside the iid product's inner measure: joint components 'a' and 'b'/,
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(inner = record(a = y, b = y), c = t))
I = iid(R, 2)
ld = logdensityof(I, table(inner = [record(a = 0.1, b = 0.2)], c = [0.3]))
`,
  },
];

for (const c of INHERITED_SURFACES) {
  test(`BACKSTOP: ${c.label} refuses, naming the level that is singular`, async () => {
    const { ctx } = ctxFor(c.src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'), (e: any) => {
      assert.equal(e.code, 'CLM_SINGULAR_JOINT');
      assert.match(String(e.message), c.backstop);
      return true;
    });
  });
}

// ════════════════════════════════════════════════════════════════════
// A Hall size-3 deficiency: the container reports its children's roots
// ════════════════════════════════════════════════════════════════════
//
// §06 "Singular joints" is "determined by the others given the shared ancestors",
// and that can hold ACROSS a nesting level with no offending pair at either level
// on its own. `lawof(record(inner = record(a = y, b = t), c = y))` has support
// {(y,t,y)} ⊂ R³ — 2-dimensional, R³-Lebesgue-null — and no PAIR of components has
// overlapping roots while a nested record reports only its own lifted name.
//
// clm's fix is not the general Hall check: a structural (record/tuple) binding is
// a CONTAINER, not a noise source, so its root set is the union of its children's.
// The coarse any-overlap pair test then sees {y,t} against {y} and refuses. This
// stays inside clm's existing sound-but-incapable posture — it only ever converts a
// wrong number into a refusal. The STATIC pass cannot widen the same way without
// false positives, which is why it still declines these.

const HALL3 = [
  {
    label: 'a nested record sharing a draw with a later sibling',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(inner = record(a = y, b = t), c = y))
ld = logdensityof(R, record(inner = record(a = 0.1, b = 0.2), c = 0.1))
`,
    pair: /joint components 'inner' and 'c' .*share the ancestor 'y'/s,
  },
  {
    // Field ORDER must not decide it. This mirror scored -2.7868155996140183 while
    // only the other spelling was covered — the classic "passes on the spelling you
    // tested" shape.
    label: 'the mirror spelling, sibling declared first',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(c = y, inner = record(a = y, b = t)))
ld = logdensityof(R, record(c = 0.1, inner = record(a = 0.1, b = 0.2)))
`,
    pair: /joint components 'c' and 'inner' .*share the ancestor 'y'/s,
  },
  {
    label: 'the joint(inner = S, c = lawof(y)) spelling',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(y), b = lawof(t))
N = joint(inner = S, c = lawof(y))
ld = logdensityof(N, record(inner = record(a = 0.1, b = 0.2), c = 0.1))
`,
    pair: /joint components 'inner' and 'c' .*share the ancestor 'y'/s,
  },
];

for (const c of HALL3) {
  test(`BACKSTOP: Hall size-3 — ${c.label} refuses`, async () => {
    const { ctx } = ctxFor(c.src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'), (e: any) => {
      assert.equal(e.code, 'CLM_SINGULAR_JOINT');
      assert.match(String(e.message), c.pair);
      return true;
    });
  });
}

test('a Hall size-3 shape whose sibling is an INLINE expression is still a MISS', async () => {
  // The remaining limit, and it is the pre-existing inline gap, not the container
  // rule: `c = y + 0.0` lifts to an untyped internal binding, which clm classifies
  // as a constructor measure rather than a variate, so it contributes no roots at
  // all. Naming it (`w = y + 0.0; … c = w`) refuses. Widening the classifier to
  // untyped lifted evaluates would also hand the full-rank nested shapes clm's
  // factually-false nullity message, which is the trade this wave declines.
  // Tracked in flatppl-dev/TODO-flatppl-js.md.
  const { ctx } = ctxFor(`
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(inner = record(a = y, b = t), c = y + 0.0))
ld = logdensityof(R, record(inner = record(a = 0.1, b = 0.2), c = 0.1))
`, 1);
  const m: any = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - -2.7868155996140187) < 1e-13,
    `expected the known-wrong -2.7868155996140187, got ${m.samples[0]}`);
});

test('a full-rank nested shape keeps its PUSHFORWARD refusal, not the singular one',
  async () => {
    // The container rule must not reach a full-rank law. This one already refused
    // before it (the ≥2-latent pushforward gate), and the reason must not drift to
    // the singular message — that message asserts nullity, which is false here.
    const { ctx } = ctxFor(`
y ~ Normal(mu = 0.0, sigma = 1.0)
n1 ~ Normal(mu = 0.0, sigma = 1.0)
n2 ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(inner = record(a = y + n1, b = t), c = y + n2))
ld = logdensityof(R, record(inner = record(a = 0.1, b = 0.2), c = 0.3))
`, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'), (e: any) => {
      assert.notEqual(e.code, 'CLM_SINGULAR_JOINT');
      assert.match(String(e.message), /depends on 2 stochastic ancestors/);
      return true;
    });
  });

// ════════════════════════════════════════════════════════════════════
// The recursion must not over-reach: legal shapes still score
// ════════════════════════════════════════════════════════════════════
//
// The inheritance recursion walks into EVERY component binding, so a legal nested
// or iid shape is one wrong step away from a false refusal — and a false refusal
// leaves the user no workaround. Both values below are pinned to an INDEPENDENT
// oracle (Distributions.jl 0.25, `logpdf(Normal(0,1), ·)` summed), never to the
// engine's own output.

test('a nested record law over three DISTINCT draws still scores, and matches the '
  + 'independent oracle', async () => {
  const { ctx } = ctxFor(`
y ~ Normal(mu = 0.0, sigma = 1.0)
s ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(inner = record(a = y, b = s), c = t))
ld = logdensityof(R, record(inner = record(a = 0.1, b = 0.2), c = 0.3))
`, 1);
  const m: any = await ctx.getMeasure('ld');
  // Three independent standard normals at 0.1, 0.2, 0.3 — the SAME numeric value
  // the singular nested shape used to return, which is exactly why that shape's
  // finite answer was invisible: here the value is correct, there it was not.
  assert.ok(Math.abs(m.samples[0] - -2.8268155996140187) < 1e-13,
    `expected -2.8268155996140187, got ${m.samples[0]}`);
});

test('iid over a scalar law still scores, and matches the independent oracle',
  async () => {
    const { ctx } = ctxFor(`
y ~ Normal(mu = 0.0, sigma = 1.0)
I = iid(lawof(y), 2)
ld = logdensityof(I, [0.1, 0.3])
`, 1);
    const m: any = await ctx.getMeasure('ld');
    assert.ok(Math.abs(m.samples[0] - -1.8878770664093456) < 1e-13,
      `expected -1.8878770664093456, got ${m.samples[0]}`);
  });

test('a LEGAL correlated joint under iid or nesting is not caught by the singular '
  + 'recursion', async () => {
  // Both of these already reach no number, but for the MARGINALISATION refusal
  // (the engine has no exact answer for the shared ancestor `z`), not for
  // singularity. Pinning which gate stops them keeps the recursion from silently
  // becoming the reason: if it ever fires here, that is a false refusal of a
  // full-rank law, and this assertion catches it rather than the error merely
  // changing wording.
  for (const src of [`
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
S = joint(a = lawof(a), b = lawof(b))
I = iid(S, 2)
ld = logdensityof(I, table(a = [0.1, 0.3], b = [0.2, 0.4]))
`, `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
S = joint(a = lawof(a), b = lawof(b))
N = joint(inner = S, c = lawof(t))
ld = logdensityof(N, record(inner = record(a = 0.1, b = 0.2), c = 0.3))
`]) {
    const { ctx } = ctxFor(src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'), (e: any) => {
      assert.notEqual(e.code, 'CLM_SINGULAR_JOINT');
      assert.match(String(e.message), /marginalises the stochastic ancestor/);
      return true;
    });
  }
});

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
  {
    // Equal root sets {y,n} AND a satisfied Hall condition, but full rank: the two
    // component functions are algebraically independent. Cov(p,q) = Var(y)-Var(n) =
    // 0, so this is two independent N(0,√2) variates, determinant 4. THE
    // discriminator for the common-generator test — a rule that fired on equal root
    // sets alone would reject this, and no generator exists here because neither
    // component is a function of the other.
    label: 'equal root sets but full rank (y+n beside y-n, independent)',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
n ~ Normal(mu = 0.0, sigma = 1.0)
p = y + n
q = y - n
S = joint(a = lawof(p), b = lawof(q))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`,
  },
  // ── the shared root is one draw, but not a single coordinate ──────────
  {
    // Root identity is per-draw-BINDING, so it cannot see that the two
    // components read different COORDINATES of `v`. Under identity covariance
    // these are two independent standard normals — full rank. Excluded because
    // an array-valued generator is not one coordinate.
    label: 'two coordinates of one multivariate draw',
    src: `
v ~ MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]])
a = v[1]
b = v[2]
S = joint(a = lawof(a), b = lawof(b))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`,
  },
  // ── a component whose support is COUNTING, not Lebesgue ───────────────
  //
  // §06 "Reference measure for product measures" builds the reference per
  // component support. A counting-referenced component cannot sit on a
  // Lebesgue-null set, so the curve argument does not apply to it — regardless of
  // how continuous the shared draw is. All four of these were static errors under
  // a draw-keyed continuity test.
  {
    // Reference is Lebesgue ⊗ Counting, and the law HAS a density w.r.t. it:
    // f(t, n) = φ(t)·1[⌊t⌋ = n], since μ(A × {n}) = ∫_A 1[⌊t⌋ = n] φ(t) dt for
    // every Borel A. Equivalently the support slices {t : ⌊t⌋ = n} = [n, n+1) are
    // not Lebesgue-null, so the support is not ρ-null.
    label: 'floor(y) beside y (Lebesgue ⊗ Counting reference, density exists)',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(a = y, b = floor(y)))
ld = logdensityof(R, record(a = 0.5, b = 0))
`,
  },
  {
    label: 'round(y) beside y (same argument)',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(a = y, b = round(y)))
ld = logdensityof(R, record(a = 0.5, b = 0))
`,
  },
  {
    label: 'a boolean-valued component beside its continuous draw',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(a = y, b = y > 0.0))
ld = logdensityof(R, record(a = 0.5, b = true))
`,
  },
  {
    label: 'the joint(lawof(y), lawof(floor(y))) spelling of the same shape',
    src: `
y ~ Normal(mu = 0.0, sigma = 1.0)
fy = floor(y)
S = joint(a = lawof(y), b = lawof(fy))
ld = logdensityof(S, record(a = 0.5, b = 0))
`,
  },
  {
    // The all-discrete case, and the cleanest instance of the rule: on a countable
    // support with counting reference the ONLY null set is ∅, so every measure
    // there is absolutely continuous and an all-discrete joint can never be
    // singular. `(k, k)` has an ordinary pmf — p at (1,1), 1-p at (0,0).
    // The variate is spelled `true`, not `1.0`: a Bernoulli joint has a BOOLEAN
    // domain, and scoring it at reals is a type error. It matters here beyond
    // tidiness — a fixture asserting a shape is LEGAL has to type-check, or the
    // `singularErrorsOf` filter quietly hides an unrelated error and the case
    // stops demonstrating what it claims.
    label: 'a duplicated BERNOULLI draw (counting reference, has a pmf)',
    src: `
k ~ Bernoulli(p = 0.3)
S = joint(a = lawof(k), b = lawof(k))
ld = logdensityof(S, record(a = true, b = true))
`,
  },
  {
    label: 'a duplicated POISSON draw (integer support, has a pmf)',
    src: `
c ~ Poisson(rate = 2.0)
S = joint(a = lawof(c), b = lawof(c))
ld = logdensityof(S, record(a = 1, b = 1))
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
    // Asserts ALL errors empty, not just this pass's. A legality fixture must
    // type-check, or `singularErrorsOf`'s filter silently hides an unrelated error
    // and the case stops demonstrating what it claims — which is how two
    // ill-typed discrete fixtures (a Bernoulli joint scored at reals) sat here
    // looking like evidence. Every fixture in this table is clean today; keep it
    // that way rather than reaching for the filter.
    assert.deepEqual(errorsOf(c.src), []);
  });
}

// ── IMPORTANT: every NOT SINGULAR case above asserts STATIC silence only ──
//
// Static silence is not end-to-end support. clm still runs the coarse
// any-overlap test at density time, so it refuses these shapes anyway: the model
// the static pass now declares legal STILL cannot be scored. The two tests below
// pin that, so the exemptions are not misread as "these models work now", and so
// narrowing clm later flips a test here rather than passing unnoticed.

test('a declined FULL-RANK shape is still refused by clm at density time '
  + '(static silence is not end-to-end support)', async () => {
  const { ctx } = ctxFor(`
y ~ Normal(mu = 0.0, sigma = 1.0)
n1 ~ Normal(mu = 0.0, sigma = 1.0)
n2 ~ Normal(mu = 0.0, sigma = 1.0)
u = y + n1
v = y + n2
S = joint(a = lawof(u), b = lawof(v))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`, 1);
  // A refusal returns no number, so clm is sound-but-incapable here — which is
  // why narrowing it was left out of scope: it would hand these shapes to a
  // scoring path nobody has verified.
  await assert.rejects(async () => ctx.getMeasure('ld'), /share the ancestor 'y'/);
});

test('a declined DISCRETE shape is still refused by clm, and with a factually '
  + 'false reason', async () => {
  const { ctx } = ctxFor(`
k ~ Bernoulli(p = 0.3)
S = joint(a = lawof(k), b = lawof(k))
ld = logdensityof(S, record(a = true, b = true))
`, 1);
  // Worth stating plainly, because it is the weakest point of the discrete
  // exemption: clm's message asserts the law "has no density w.r.t. the product
  // reference measure", which is FALSE of `(k, k)` — it has a pmf w.r.t.
  // counting ⊗ counting. So this exemption is static-only and buys the user
  // nothing yet; fixing it means touching the runtime path. Filed in
  // flatppl-dev/TODO-flatppl-js.md.
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /share the ancestor 'k'.*no density w\.r\.t\. the product reference measure/s);
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

// The generator test needs the dependency to pass through a shared NAMED binding,
// because the generator it looks for IS a binding. Two dependent shapes are
// therefore missed. Both are pinned as misses so the module header's narrowed
// claim is backed by a test rather than only asserted, and so closing either one
// flips a test here.
//
// Closing them needs STRUCTURAL comparison of the component expressions (or
// symbolic dependence testing) — deliberately not a wider root-set rule, since
// widening roots is exactly what produced the earlier false positives.

test('a dependent pair with NO shared named binding is a MISS: q = 2y+2n equals '
  + '2p, but never reaches p', () => {
  // Rank 1 (q = 2p exactly), so genuinely singular. `q` refs y and n directly, so
  // no binding is a generator for both components. Still safe: clm's
  // shared-ancestor check refuses it at density time (verified).
  assert.deepEqual(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
n ~ Normal(mu = 0.0, sigma = 1.0)
p = y + n
q = 2.0 * y + 2.0 * n
S = joint(a = lawof(p), b = lawof(q))
ld = logdensityof(S, record(a = 0.5, b = 1.0))
`), []);
});

test('two INLINE identical field expressions are a MISS (no binding to nominate)', () => {
  // Support is the diagonal of R², so singular. The fields are structurally
  // identical but inline, so there is no named generator. Still safe: the
  // multi-latent pushforward path refuses it at density time (verified).
  assert.deepEqual(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
n ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(a = y + n, b = y + n))
ld = logdensityof(R, record(a = 0.5, b = 0.5))
`), []);
});

test('a Hall deficiency needing a subset of size 3 is a MISS (documented limit)', () => {
  // Support is {(y, t, y)} ⊂ R³ — 2-dimensional, so genuinely singular. No PAIR
  // of components has equal singleton root sets ({y,t} vs {y}), so the |S| = 2
  // rule cannot see it. Pinned as a known miss rather than left undocumented; if
  // the general Hall check ever lands, this test flips to expecting an error.
  //
  // STATIC silence only. clm now REFUSES this shape (its container rule unions a
  // nested record's children's roots, so the pair test sees {y,t} against {y}) —
  // see the "Hall size-3" backstop tests above. The static pass cannot copy that
  // widening without false positives, so the two sides diverge here on purpose.
  assert.deepEqual(singularErrorsOf(`
y ~ Normal(mu = 0.0, sigma = 1.0)
t ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(inner = record(a = y, b = t), c = y))
ld = logdensityof(R, record(inner = record(a = 0.1, b = 0.2), c = 0.1))
`), []);
});
