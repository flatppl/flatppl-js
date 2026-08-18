'use strict';

// `joint` over KERNELS — the fan-out kernel (spec §06 "Joint composition" +
// "Uniform kernel extension"). Before this landed, every spelling was a static
// error ("joint kwarg \"p\" expects a measure, got kernel(…)"), which rejected a
// construct §06 permits: the uniform kernel extension applies to every
// measure-to-measure operation except `jointchain` and `kchain`.
//
// SPEC ANCHORS, quoted verbatim (flatppl-design docs/06-measure-algebra.md,
// `joint` entry; the paragraph below is flatppl-design#85, owner-merge pending,
// so this engine is briefly ahead of the published spec — noted in
// flatppl-dev/TODO-flatppl-js.md):
//
//   "For kernels, `joint(K1, K2, ...)` results in a kernel that fans a single
//   input out to all component kernels, so each of them receives the same
//   input. The result's inputs are the union of the component kernels' inputs
//   by name; a component receives the inputs it declares and is unaffected by
//   the others […] Components that share a stochastic node must agree on that
//   node's ancestry: every ancestor of the shared node that any component binds
//   as a boundary input must be bound by every sharing component, under the
//   same input name. A `joint` in which a sharing component binds such an
//   ancestor under a different name, or does not bind it at all — in particular
//   a measure component, which binds nothing — is a static error. Measure
//   components are permitted and are the nullary case: they ignore the input. A
//   measure component may be parameterized and may share stochastic nodes with
//   kernel components; only a shared node with a boundary-bound ancestor is
//   excluded, by the naming clause above.
//   The keyword form applies unchanged, producing a kernel whose output variate
//   is a record. At each input point the result is the `joint` of the component
//   output measures, so the ancestry rule above governs it: component kernels
//   whose traces share a stochastic node yield the correlated record law at
//   each input, and components sharing no stochastic node yield the product —
//   the fanned input is a value, not a stochastic node, and so induces no
//   dependence by itself. The result's total-mass class is the product of the
//   components' classes, as in the measure case; when components sharing a
//   stochastic node include more than one non-normalized member the product is
//   not an upper bound on the composed mass, and no class stronger than unknown
//   is statically justified. A fan-out of Markov kernels is a Markov kernel."
//
// §06 "Uniform kernel extension": "we unify measures and kernels and identify
// measures with nullary kernels", and "This applies to all measure-to-measure
// operations except `jointchain` and `kchain`".
//
// §06 "Singular joints": "the same draw referenced twice […] has no density
// w.r.t. the product reference measure. Sampling is well-defined; a density
// query is a static error where statically detectable".
//
// ORACLES ARE INDEPENDENT of this engine and of flatppl-rust (which evaluates
// no density at all). Each target below was computed twice by routes that share
// no algebra:
//   (a) quadrature of ∫ p(u | z) p(y1 | u) p(y2 | u) du, which never forms a
//       covariance matrix, and
//   (b) the closed-form Gaussian with an explicit 2x2 inverse and determinant.
// The probe: z the fanned input, u ~ Normal(z, 1) shared, a1, a2 ~ Normal(u, 1).
// At each z the record law is MvNormal([z, z], [[2, 1], [1, 2]]).
//   z = 0,   y = (1, -1)  →  -log(2π) - ½log 3 - 1 = -3.3871832107434
//   z = 0.7, y = (1, -1)  →                          -3.5505165440767334
// The trace-DISJOINT control (a private u per component) is the product of the
// two marginals, each Normal(z, √2):
//   z = 0,   y = (1, -1)  →                          -3.0310242469692907
// Those two numbers are 0.36 nats apart, so they discriminate the retained law
// from the product with no tolerance argument needed.

const test = require('node:test');
const assert = require('node:assert');
const { processSource, materialiser, types: T } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const RETAIN_AT_0 = -3.3871832107434;
const RETAIN_AT_07 = -3.5505165440767334;
const DISJOINT_AT_0 = -3.0310242469692907;
const F64_TOL = 1e-14;

// z fanned, `u` shared by both components — the retained-node probe.
const SHARED = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
K2 = kernelof(a2, z = z)
KJ = joint(p = K1, q = K2)
`;

// The same shape with a PRIVATE latent per component: trace-disjoint, so §06
// makes the components independent and the applied joint is the product.
const DISJOINT = `
z = elementof(reals)
u1 ~ Normal(mu = z, sigma = 1.0)
u2 ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u1, sigma = 1.0)
a2 ~ Normal(mu = u2, sigma = 1.0)
K1 = kernelof(a1, z = z)
K2 = kernelof(a2, z = z)
KJ = joint(p = K1, q = K2)
`;

const infer = (src: string) => {
  const { bindings, diagnostics } = processSource(src);
  return {
    bindings,
    errors: (diagnostics || []).filter((d: any) => d.severity === 'error'),
  };
};
const typeOf = (bindings: any, name: string) => bindings.get(name).inferredType;
const inputNamesOf = (t: any) => (t.inputs || []).map((i: any) => i.name);
// A kernel carries its total-mass class on the OUTPUT measure (§11: "the
// total-mass class of the output measure, uniform over all inputs").
const massOf = (t: any) => (t.kind === 'kernel' ? t.result.mass : t.mass);

const scoreOf = async (src: string, N?: number) => {
  const { ctx } = ctxFor(src, N === undefined ? 1 : N);
  return (await ctx.getMeasure('ld')).samples[0];
};

// ── type layer ──────────────────────────────────────────────────────────────

test('a keyword kernel joint is a kernel over a RECORD variate, not an error', () => {
  const { bindings, errors } = infer(SHARED);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  const t = typeOf(bindings, 'KJ');
  assert.equal(t.kind, 'kernel');
  assert.deepEqual(inputNamesOf(t), ['z']);
  assert.equal(t.result.kind, 'measure');
  assert.equal(t.result.domain.kind, 'record');
  assert.deepEqual(Object.keys(t.result.domain.fields), ['p', 'q']);
});

test('a positional kernel joint keeps the `cat` variate the measure case gives',
  () => {
    const { bindings, errors } = infer(SHARED + 'V = joint(K1, K2)\n');
    assert.deepEqual(errors.map((e: any) => e.message), []);
    const t = typeOf(bindings, 'V');
    assert.equal(t.kind, 'kernel');
    assert.deepEqual(inputNamesOf(t), ['z']);
    assert.equal(t.result.domain.kind, 'array');
    assert.deepEqual(t.result.domain.shape, [2]);
  });

test('the result inputs are the union BY NAME — one input per name, in first '
  + 'declaration order', () => {
  // Two components declaring the same name `z` fan one value to both, so the
  // union has one input; two components declaring `z` and `w` give both.
  const shared = infer(SHARED);
  assert.deepEqual(inputNamesOf(typeOf(shared.bindings, 'KJ')), ['z']);
  const { bindings, errors } = infer(`
z = elementof(reals)
w = elementof(reals)
b1 ~ Normal(mu = z, sigma = 1.0)
b2 ~ Normal(mu = w, sigma = 1.0)
K1 = kernelof(b1, z = z)
K2 = kernelof(b2, w = w)
KJ = joint(a = K1, b = K2)
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.deepEqual(inputNamesOf(typeOf(bindings, 'KJ')), ['z', 'w']);
});

test('a MEASURE component is legal and contributes nothing to the input union '
  + '(§06: measures are the nullary case)', () => {
  const { bindings, errors } = infer(`
z = elementof(reals)
a1 ~ Normal(mu = z, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = Normal(mu = 0.0, sigma = 1.0)
KJ = joint(p = K1, q = M)
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  const t = typeOf(bindings, 'KJ');
  assert.equal(t.kind, 'kernel');
  assert.deepEqual(inputNamesOf(t), ['z']);
  assert.deepEqual(Object.keys(t.result.domain.fields), ['p', 'q']);
});

test('an all-measure joint stays a MEASURE — the empty union collapses (§06 '
  + 'kernel↔measure boundary)', () => {
  const { bindings, errors } = infer(`
J = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Normal(mu = 0.0, sigma = 1.0))
V = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 0.0, sigma = 1.0))
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.equal(typeOf(bindings, 'J').kind, 'measure');
  assert.equal(typeOf(bindings, 'V').kind, 'measure');
});

test('a component that is neither a measure nor a kernel still errors — the '
  + 'diagnostics narrow, they do not disappear', () => {
  const kw = infer(`
a = Normal(mu = 0.0, sigma = 1.0)
x = draw(a)
j = joint(p = a, q = x)
`);
  assert.ok(kw.errors.some((e: any) =>
    /joint kwarg "q" expects a measure or a kernel, got real/.test(e.message)),
  'got: ' + kw.errors.map((e: any) => e.message).join(' | '));
  const pos = infer(`
a = Normal(mu = 0.0, sigma = 1.0)
x = draw(a)
j = joint(a, x)
`);
  assert.ok(pos.errors.some((e: any) =>
    /joint component expects a measure or a kernel, got real/.test(e.message)),
  'got: ' + pos.errors.map((e: any) => e.message).join(' | '));
});

test('a component whose own type is not yet inferred leaves its field deferred '
  + 'rather than erroring', () => {
  // §11 `%deferred` is "not yet inferred", so a gap in another op's typing must
  // not turn a well-formed `joint` into a diagnostic.
  const { bindings } = infer(`
R = restrict(Normal(mu = 0.0, sigma = 1.0), interval(0.0, 1.0))
J = joint(a = R, b = Normal(mu = 0.0, sigma = 1.0))
`);
  const t = typeOf(bindings, 'J');
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.fields.a.kind, 'deferred');
  assert.equal(t.domain.fields.b.kind, 'scalar');
});

// ── the boundary-naming static error (§06) ──────────────────────────────────

test('components sharing a stochastic node must reach it through the SAME '
  + 'input name', () => {
  // `u` is retained, and its boundary ancestor `z` arrives as input `s` in one
  // component and `t` in the other. At an application point with s ≠ t the
  // single node `u` has no well-defined parent value.
  const { errors } = infer(`
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, s = z)
K2 = kernelof(a2, t = z)
KJ = joint(p = K1, q = K2)
`);
  assert.ok(errors.some((e: any) =>
    /share the stochastic node 'u'.*bound as input 's'.*and 't'/s.test(e.message)),
  'got: ' + errors.map((e: any) => e.message).join(' | '));
});

test('differently-named inputs over DISJOINT traces are legal — the clause is '
  + 'about shared nodes, not about names', () => {
  const { bindings, errors } = infer(`
z = elementof(reals)
w = elementof(reals)
b1 ~ Normal(mu = z, sigma = 1.0)
b2 ~ Normal(mu = w, sigma = 1.0)
K1 = kernelof(b1, s = z)
K2 = kernelof(b2, t = w)
KJ = joint(p = K1, q = K2)
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.deepEqual(inputNamesOf(typeOf(bindings, 'KJ')), ['s', 't']);
});

test('a shared node reached through ONE input name is legal', () => {
  const { errors } = infer(`
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, s = z)
K2 = kernelof(a2, s = z)
KJ = joint(p = K1, q = K2)
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
});

// ── total-mass class ────────────────────────────────────────────────────────

test('a fan-out of Markov kernels is a Markov kernel', () => {
  const { bindings } = infer(SHARED);
  assert.equal(massOf(typeOf(bindings, 'KJ')), T.MASS_NORMALIZED);
});

test('the class is the PRODUCT of the component classes, kernel components '
  + 'included', () => {
  const { bindings, errors } = infer(`
z = elementof(reals)
a1 ~ Normal(mu = z, sigma = 1.0)
K1 = kernelof(a1, z = z)
KW = functionof(weighted(2.0, Normal(mu = z, sigma = 1.0)), z = z)
KJ = joint(p = K1, q = KW)
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.equal(massOf(typeOf(bindings, 'KW')), T.MASS_FINITE);
  assert.equal(massOf(typeOf(bindings, 'KJ')), T.MASS_FINITE);
});

test('two non-normalized components sharing a stochastic node downgrade to '
  + 'unknown; disjoint ones keep the product', () => {
  // The product is not an upper bound under sharing: the composed mass is
  // E[w₁(a₁)·w₂(a₂)], which E[w₁]·E[w₂] does not bound. A Student-t (ν = 3)
  // ancestor with wᵢ(y) = y² is the counterexample — each component finite, the
  // composition infinite.
  const { bindings, errors } = infer(`
z = elementof(reals)
u ~ StudentT(nu = 3.0)
e ~ Normal(mu = 0.0, sigma = 1.0)
K1 = functionof(weighted(2.0, lawof(u)), z = z)
K2 = functionof(weighted(3.0, lawof(u + e)), z = z)
K3 = functionof(weighted(3.0, lawof(e)), z = z)
KSHARED = joint(p = K1, q = K2)
KDISJOINT = joint(p = K1, q = K3)
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.equal(massOf(typeOf(bindings, 'KSHARED')), T.MASS_UNKNOWN);
  assert.equal(massOf(typeOf(bindings, 'KDISJOINT')), T.MASS_FINITE);
});

test('the sharing test stops at a kernel component\'s BOUNDARY — a common '
  + 'ancestor above it is not in either trace', () => {
  // §04 "Specifying reification boundaries" replaces boundary nodes with fresh
  // `elementof` inputs, so `w` is not in either component's trace: below the
  // boundary they are {x1} and {x2}, disjoint. Given the fanned `m` the
  // components are independent and the product (mass 6, `finite`) is exact.
  // Walking past the boundary read `w` as shared and downgraded to `unknown`.
  const { bindings, errors } = infer(`
w ~ Normal(mu = 0.0, sigma = 1.0)
m = w * 2.0
x1 ~ Normal(mu = m, sigma = 1.0)
x2 ~ Normal(mu = m, sigma = 1.0)
K1 = functionof(weighted(2.0, lawof(x1)), m = m)
K2 = functionof(weighted(3.0, lawof(x2)), m = m)
KJ = joint(p = K1, q = K2)
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.equal(massOf(typeOf(bindings, 'K1')), T.MASS_FINITE);
  assert.equal(massOf(typeOf(bindings, 'K2')), T.MASS_FINITE);
  assert.equal(massOf(typeOf(bindings, 'KJ')), T.MASS_FINITE);
});

test('the same downgrade applies at arity zero — the counterexample is a '
  + 'MEASURE joint', () => {
  const { bindings, errors } = infer(`
u ~ StudentT(nu = 3.0)
e ~ Normal(mu = 0.0, sigma = 1.0)
u2 = u + e
v ~ StudentT(nu = 3.0)
JSHARED = joint(a = weighted(2.0, lawof(u)), b = weighted(3.0, lawof(u2)))
JDISJOINT = joint(a = weighted(2.0, lawof(u)), b = weighted(3.0, lawof(v)))
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.equal(massOf(typeOf(bindings, 'JSHARED')), T.MASS_UNKNOWN);
  assert.equal(massOf(typeOf(bindings, 'JDISJOINT')), T.MASS_FINITE);
});

// ── density of the APPLIED fan-out ──────────────────────────────────────────

test('a shared internal node gives the CORRELATED record law at the input', async () => {
  const got = await scoreOf(SHARED
    + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))\n');
  assert.ok(Math.abs(got - RETAIN_AT_0) < F64_TOL,
    `got ${got}, quadrature over the shared latent and MvNormal([0,0],[[2,1],[1,2]]) `
    + `both give ${RETAIN_AT_0}. The product of the marginals would be `
    + `${DISJOINT_AT_0} — copying the shared node is a DIFFERENT measure`);
});

test('the applied fan-out reads a NON-ZERO input', async () => {
  const got = await scoreOf(SHARED
    + 'ld = logdensityof(KJ(z = 0.7), record(p = 1.0, q = -1.0))\n');
  assert.ok(Math.abs(got - RETAIN_AT_07) < F64_TOL,
    `got ${got}, oracle ${RETAIN_AT_07} (the law shifts to MvNormal([0.7,0.7],…), `
    + 'so an ignored input would score the z = 0 value)');
});

test('the POSITIONAL spelling scores the same law over a vector variate', async () => {
  const got = await scoreOf(SHARED
    + 'V = joint(K1, K2)\nld = logdensityof(V(z = 0.0), [1.0, -1.0])\n');
  assert.ok(Math.abs(got - RETAIN_AT_0) < F64_TOL, `got ${got}, oracle ${RETAIN_AT_0}`);
});

test('TRACE-DISJOINT components are independent: the product of the marginals',
  async () => {
    const got = await scoreOf(DISJOINT
      + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))\n');
    assert.ok(Math.abs(got - DISJOINT_AT_0) < F64_TOL,
      `got ${got}, product of two Normal(0, √2) marginals ${DISJOINT_AT_0}. The `
      + `correlated law would be ${RETAIN_AT_0} — the fanned INPUT is a value, not `
      + 'a stochastic node, so it induces no dependence by itself');
  });

test('a MEASURE component contributes its own density, unaffected by the input',
  async () => {
    // K1(0) is Normal(0, √2) and M is Normal(0, 1): the components share no
    // stochastic node, so the score is the sum of the two logpdfs.
    // scipy: norm.logpdf(1, 0, √2) + norm.logpdf(0.5, 0, 1) = -2.559450656689318
    const oracle = -2.559450656689318;
    const got = await scoreOf(`
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
M = Normal(mu = 0.0, sigma = 1.0)
KJ = joint(p = K1, q = M)
ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = 0.5))
`);
    assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
  });

test('each component reads only the inputs it declares', async () => {
  // b1 ~ Normal(z, 1) at z = 0.5 and b2 ~ Normal(w, 1) at w = -0.5.
  // scipy: norm.logpdf(1, 0.5, 1) + norm.logpdf(0.5, -0.5, 1) = -2.4628770664093453
  const oracle = -2.4628770664093453;
  const got = await scoreOf(`
z = elementof(reals)
w = elementof(reals)
b1 ~ Normal(mu = z, sigma = 1.0)
b2 ~ Normal(mu = w, sigma = 1.0)
K1 = kernelof(b1, z = z)
K2 = kernelof(b2, w = w)
KJ = joint(p = K1, q = K2)
ld = logdensityof(KJ(z = 0.5, w = -0.5), record(p = 1.0, q = 0.5))
`);
  assert.ok(Math.abs(got - oracle) < F64_TOL,
    `got ${got}, oracle ${oracle} — a component that also read the other's input `
    + 'would score a shifted mean');
});

test('a SINGULAR fan-out — joint(K, K) — refuses the density query (§06 '
  + '"Singular joints")', async () => {
  const { ctx } = ctxFor(`
z = elementof(reals)
a1 ~ Normal(mu = z, sigma = 1.0)
K1 = kernelof(a1, z = z)
KJ = joint(p = K1, q = K1)
ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /components 'p' and 'q' are reified laws of the same draw.*lower-dimensional subset/s,
    'the fan-out of one reified kernel references one draw twice, so it has no '
    + 'density w.r.t. the product reference measure');
});

test('an applied kernel whose reified RECORD names one draw twice refuses too — '
  + 'a collateral fix, off the fan-out path', async () => {
  // Not a `joint` over kernels at all: one `kernelof` whose record body names the
  // same draw in two fields. §06 "Singular joints" refuses it — the pair
  // concentrates on the diagonal — and until the closure copy was recognised as a
  // variate the engine SCORED it. Both spellings, measured on origin/main:
  //   no latent   → -2.8378770664093453, the same value the DISTINCT-draw
  //                 program gives, so the repeat was simply invisible;
  //   with latent → -3.3871832107434, the value that belongs to two distinct
  //                 children of the shared node.
  // The refusal message names an internal binding (filed gap 6), so it is not
  // pinned here.
  const singular = [`
z = elementof(reals)
a1 ~ Normal(mu = z, sigma = 1.0)
KR = kernelof(record(p = a1, q = a1), z = z)
ld = logdensityof(KR(z = 0.0), record(p = 1.0, q = -1.0))
`, `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
KR = kernelof(record(p = a1, q = a1), z = z)
ld = logdensityof(KR(z = 0.0), record(p = 1.0, q = -1.0))
`];
  for (const src of singular) {
    const { ctx } = ctxFor(src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'),
      /are reified laws of the same draw/);
  }
  // The controls must keep scoring: two DISTINCT draws under the same boundary
  // are two coordinates, and the recognition must not reach them.
  const noLatent = await scoreOf(`
z = elementof(reals)
a1 ~ Normal(mu = z, sigma = 1.0)
a2 ~ Normal(mu = z, sigma = 1.0)
KR = kernelof(record(p = a1, q = a2), z = z)
ld = logdensityof(KR(z = 0.0), record(p = 1.0, q = -1.0))
`);
  assert.ok(Math.abs(noLatent - -2.8378770664093453) < F64_TOL,
    `got ${noLatent}, two independent Normal(0,1) coordinates -2.8378770664093453`);
  const withLatent = await scoreOf(`
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
KR = kernelof(record(p = a1, q = a2), z = z)
ld = logdensityof(KR(z = 0.0), record(p = 1.0, q = -1.0))
`);
  assert.ok(Math.abs(withLatent - RETAIN_AT_0) < F64_TOL,
    `got ${withLatent}, correlated record law ${RETAIN_AT_0}`);
});

// ── a kernel is not a closed measure ────────────────────────────────────────

test('scoring the UNAPPLIED fan-out is a static error (only closed measures '
  + 'have a density)', () => {
  const { errors } = infer(SHARED
    + 'ld = logdensityof(KJ, record(p = 1.0, q = -1.0))\n');
  assert.ok(errors.some((e: any) =>
    /logdensityof: arg 1 expects measure, got kernel/.test(e.message)),
  'got: ' + errors.map((e: any) => e.message).join(' | '));
});

test('sampling the UNAPPLIED fan-out refuses', async () => {
  const { ctx } = ctxFor(SHARED, 8);
  await assert.rejects(async () => materialiser.materialiseMeasure('KJ', ctx),
    /not materialisable as a closed measure/);
});

// ── shapes the density lowering does not reach (each refuses honestly) ──────

test('a component reached through an ALIAS scores the same law (§04 "Aliasing '
  + 'is just assignment")', async () => {
  const got = await scoreOf(SHARED
    + 'G = K1\nKA = joint(p = G, q = K2)\n'
    + 'ld = logdensityof(KA(z = 0.0), record(p = 1.0, q = -1.0))\n');
  assert.ok(Math.abs(got - RETAIN_AT_0) < F64_TOL, `got ${got}, oracle ${RETAIN_AT_0}`);
});

test('LAMBDA components type as a fan-out kernel, but the applied density is a '
  + 'KNOWN GAP that refuses', async () => {
  // A lambda's boundary is a placeholder, not a named outer node, so the
  // boundary hoist this lowering performs has nothing to bind the union input
  // to. The type layer is complete for the shape; only the density lowering
  // defers, and it refuses rather than answering. Filed in
  // flatppl-dev/TODO-flatppl-js.md; when it lands, this test flips.
  const src = `
KL1 = z -> Normal(mu = z, sigma = 1.0)
KL2 = z -> Normal(mu = z, sigma = 2.0)
KJ = joint(p = KL1, q = KL2)
ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))
`;
  const { bindings, errors } = infer(src);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  const t = typeOf(bindings, 'KJ');
  assert.equal(t.kind, 'kernel');
  assert.deepEqual(inputNamesOf(t), ['z']);
  const { ctx } = ctxFor(src, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'), /no derivation for 'ld'/);
});

test('a NESTED kernel joint as a component types correctly but the applied '
  + 'density is a KNOWN GAP that refuses', async () => {
  // Nesting is legal and flattening-equivalent — node identity is global, so
  // nesting order cannot change which nodes coincide — but the hoist's component
  // reader only accepts `kernelof` / `functionof` heads, so a `joint`-built
  // kernel component is not hoistable. Both spellings refuse. Filed in
  // flatppl-dev/TODO-flatppl-js.md.
  const pre = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
a3 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
K2 = kernelof(a2, z = z)
K3 = kernelof(a3, z = z)
`;
  const point = 'record(p = 1.0, q = record(r = 1.0, s = -1.0))';
  for (const decl of [
    'KI = joint(r = K2, s = K3)\nKN = joint(p = K1, q = KI)',
    'KN = joint(p = K1, q = joint(r = K2, s = K3))',
  ]) {
    const src = pre + decl + `\nld = logdensityof(KN(z = 0.0), ${point})\n`;
    const { bindings, errors } = infer(src);
    assert.deepEqual(errors.map((e: any) => e.message), []);
    const t = typeOf(bindings, 'KN');
    assert.equal(t.kind, 'kernel');
    assert.deepEqual(inputNamesOf(t), ['z']);
    assert.equal(t.result.domain.fields.q.kind, 'record',
      'the nested joint contributes a nested record field, not merged fields');
    const { ctx } = ctxFor(src, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'), /no derivation for 'ld'/);
  }
});

test('a LEGAL one-node-under-two-names pair types correctly but the applied '
  + 'density is a KNOWN GAP that refuses', async () => {
  // §06's clause constrains SHARING components only, and these two are
  // trace-disjoint, so `{s, t}` is the right signature and typeinfer is silent.
  // The hoist still refuses: one substitution map cannot give `wv` two fed
  // values. Filed in flatppl-dev/TODO-flatppl-js.md.
  const src = `
wv = elementof(reals)
b1 ~ Normal(mu = wv, sigma = 1.0)
b2 ~ Normal(mu = wv, sigma = 1.0)
K1 = kernelof(b1, s = wv)
K2 = kernelof(b2, t = wv)
KJ = joint(p = K1, q = K2)
ld = logdensityof(KJ(s = 0.0, t = 0.0), record(p = 1.0, q = -1.0))
`;
  const { bindings, errors } = infer(src);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.deepEqual(inputNamesOf(typeOf(bindings, 'KJ')), ['s', 't']);
  const { ctx } = ctxFor(src, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'), /no derivation for 'ld'/);
});

test('applying an ALL-MEASURE joint is a static error — it has no inputs', () => {
  const { errors } = infer(`
J = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Normal(mu = 0.0, sigma = 1.0))
ld = logdensityof(J(z = 0.0), record(a = 1.0, b = -1.0))
`);
  assert.ok(errors.some((e: any) => /"J" is not callable/.test(e.message)),
    'got: ' + errors.map((e: any) => e.message).join(' | '));
});

test('the boundary-naming conflict also refuses at score time, not just at '
  + 'typeinfer', async () => {
  const { ctx } = ctxFor(`
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, s = z)
K2 = kernelof(a2, t = z)
KJ = joint(p = K1, q = K2)
ld = logdensityof(KJ(s = 0.0, t = 0.0), record(p = 1.0, q = -1.0))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'), /no derivation for 'ld'/,
    'a shape §06 makes a static error must not also produce a number');
});

// ── nested boundary scopes: the identity is not licensed, so refuse ─────────
//
// The commuting identity joint(kernelof(a1, B1), kernelof(a2, B2)) =
// kernelof(record(p = a1, q = a2), B1 ∪ B2) rests on each boundary set being an
// all-or-none complete cut OF ITS OWN COMPONENT (§04 "Specifying reification
// boundaries"). When a node in B1 is an INTERIOR ancestor of component 2, the
// union cuts a path component 2 marginalizes over: component 2's own input goes
// dead and it is scored at the wrong law.
//
// No ruling is needed to call that wrong. It breaks the marginalization floor —
// the i-th marginal of the composed law at an input must equal K_i at that same
// input — which BOTH candidate conventions (retain and copy) satisfy. So the
// hoist refuses these shapes; the truths below are what an implementation that
// carries per-component boundary scopes must produce, not what this engine
// answers today. Filed in flatppl-dev/TODO-flatppl-js.md.

const NESTED_BOUNDARY = `
z = elementof(reals)
v ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = v, sigma = 1.0)
a2 ~ Normal(mu = v, sigma = 1.0)
K1 = kernelof(a1, s = v)
K2 = kernelof(a2, t = z)
KJ = joint(p = K1, q = K2)
`;

test('the components of a nested-boundary fan-out still score correctly ONE AT '
  + 'A TIME — the reference the composed law must reproduce', async () => {
  // K1(s) is Normal(s, 1); K2(t) marginalizes v and is Normal(t, √2).
  // scipy: norm.logpdf(1, 0, 1) = -1.4189385332046727
  //        norm.logpdf(-1, 0, √2) = -1.5155121234846453
  //        norm.logpdf(-1, 5, √2) = -10.265512123484644
  const one = await scoreOf(NESTED_BOUNDARY + 'ld = logdensityof(K1(s = 0.0), 1.0)\n');
  assert.ok(Math.abs(one - -1.4189385332046727) < F64_TOL, `K1: got ${one}`);
  const two = await scoreOf(NESTED_BOUNDARY + 'ld = logdensityof(K2(t = 0.0), -1.0)\n');
  assert.ok(Math.abs(two - -1.5155121234846453) < F64_TOL, `K2 at t=0: got ${two}`);
  const far = await scoreOf(NESTED_BOUNDARY + 'ld = logdensityof(K2(t = 5.0), -1.0)\n');
  assert.ok(Math.abs(far - -10.265512123484644) < F64_TOL, `K2 at t=5: got ${far}`);
});

test('a nested boundary scope REFUSES the applied density instead of scoring a '
  + 'law with a dead input', async () => {
  // These components share NO stochastic node, so §06 makes the composed law
  // the product of the per-component conditionals:
  //   (s=0, t=0)  →  -1.4189385332046727 + -1.5155121234846453 = -2.934450656689318
  //   (s=0, t=5)  →  -1.4189385332046727 + -10.265512123484644 = -11.684450656689316
  // Both confirmed by quadrature over v and by the closed form. Before the
  // complete-cut check the engine returned -2.8378770664093453 at BOTH feed
  // points — the fed `t` was ignored entirely, and the q-marginal had variance 1
  // where K2(t) has variance 2.
  for (const feed of ['s = 0.0, t = 0.0', 's = 0.0, t = 5.0']) {
    const { ctx } = ctxFor(NESTED_BOUNDARY
      + `ld = logdensityof(KJ(${feed}), record(p = 1.0, q = -1.0))\n`, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'), /no derivation for 'ld'/,
      `KJ(${feed}) must refuse, not answer`);
  }
});

test('a nested boundary scope refuses SAMPLING too, so no wrong moments escape',
  async () => {
    // The truth at (s = 0, t = 5) is E[p] = 0, Var(p) = 1, E[q] = 5, Var(q) = 2.
    // The pre-fix sampler gave q mean ≈ -0.02, var ≈ 0.96 — the same dead input
    // as the density, so sampling ≡ density agreement held while both were wrong.
    const { ctx } = ctxFor(NESTED_BOUNDARY + 'S = KJ(s = 0.0, t = 5.0)\n', 64);
    await assert.rejects(async () => materialiser.materialiseMeasure('S', ctx),
      /no derivation for 'S'/);
  });

test('the SHARING nested-boundary variant is a STATIC error, and still refuses',
  async () => {
  // u is shared, and its boundary ancestor `v` is bound as `s` by K1 and not
  // bound at all by K2, so the union cuts K2's path through v. Before the
  // complete-cut check this scored -3.3871832107434 at every feed point, while
  // K2(t = 0) alone is -1.6349113442053942 = Normal(0, √3): the composed
  // q-marginal had variance 2 against K2's 3. This is the KERNEL non-binder —
  // the same missing case as W1 (kernel-joint-w1-maths.md §6, "the kernel-side
  // analogue … is the same missing case and currently refuses only by the C1
  // complete-cut mechanism, not by the clause"), so the clarified clause now
  // catches it statically and the runtime refusal remains as the second layer.
  const src = `
z = elementof(reals)
v ~ Normal(mu = z, sigma = 1.0)
u ~ Normal(mu = v, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, s = v)
K2 = kernelof(a2, t = z)
KJ = joint(p = K1, q = K2)
ld = logdensityof(KJ(s = 0.0, t = 0.0), record(p = 1.0, q = -1.0))
`;
  const { errors } = infer(src);
  assert.ok(errors.some((e: any) =>
    /share the stochastic node 'u'.*ancestor 'v' is bound as input 's'.*not bound at all by another/s
      .test(e.message)),
  'got: ' + errors.map((e: any) => e.message).join(' | '));
  const { ctx } = ctxFor(src, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'), /no derivation for 'ld'/);
});

test('a component that declares an EXTRA input the other lacks is not a nested '
  + 'scope and still scores', async () => {
  // The complete-cut check must not fire on a legal union: K2 binds {z, w} and
  // K1 binds {z}, but neither component's boundary node lies inside the other.
  // At w = 0 the law is the shared probe's.
  const got = await scoreOf(`
z = elementof(reals)
w = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u + w, sigma = 1.0)
K1 = kernelof(a1, z = z)
K2 = kernelof(a2, z = z, w = w)
KJ = joint(p = K1, q = K2)
ld = logdensityof(KJ(z = 0.0, w = 0.0), record(p = 1.0, q = -1.0))
`);
  assert.ok(Math.abs(got - RETAIN_AT_0) < F64_TOL, `got ${got}, oracle ${RETAIN_AT_0}`);
});

// ── the draw classifier must stay confined to reification closures ──────────

test('two distinct inline draw(m) components are INDEPENDENT, not one draw',
  async () => {
    // §04 "Reification to measures": "a `joint` of two reified laws of the same
    // draw is the singular diagonal joint, while `joint(m, m)` is the product of
    // two independent draws". No kernel and no fan-out is involved here — this
    // pins that the singular-fan-out fix stays inside reification closures.
    // scipy: 2 * norm.logpdf(1, 0, 1) = -2.8378770664093453
    const oracle = -2.8378770664093453;
    const got = await scoreOf(`
m = Normal(mu = 0.0, sigma = 1.0)
L = lawof(record(a = draw(m), b = draw(m)))
ld = logdensityof(L, record(a = 1.0, b = -1.0))
`);
    assert.ok(Math.abs(got - oracle) < F64_TOL,
      `got ${got}, product of two independent Normal(0,1) draws ${oracle}. A `
      + 'refusal here would mean the draw classifier reaches ordinary lifted '
      + 'anons, not just reification-closure copies');
  });

// ── the mass class survives application ─────────────────────────────────────

test('applying the fan-out does not erase its class — the application carries '
  + 'the kernel output class (§11)', () => {
  // `inferUserCall` returns the callee's `result` type OBJECT, so an application
  // that classified as `deferred` stamped that back over the kernel's own class.
  // `deferred` is the class the draw gate PASSES where `unknown` fails, so the
  // erasure hid the Q5 downgrade on every fan-out that is actually applied.
  const applied = infer(SHARED + 'A = KJ(z = 0.0)\n');
  assert.equal(massOf(typeOf(applied.bindings, 'KJ')), T.MASS_NORMALIZED);
  assert.equal(massOf(typeOf(applied.bindings, 'A')), T.MASS_NORMALIZED);
  const scored = infer(SHARED
    + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))\n');
  assert.equal(massOf(typeOf(scored.bindings, 'KJ')), T.MASS_NORMALIZED);
});

test('the Q5 downgrade survives application and reaches the draw gate', () => {
  const Q5 = `
z = elementof(reals)
u ~ StudentT(nu = 3.0)
e ~ Normal(mu = 0.0, sigma = 1.0)
K1 = functionof(weighted(2.0, lawof(u)), z = z)
K2 = functionof(weighted(3.0, lawof(u + e)), z = z)
KS = joint(p = K1, q = K2)
A = KS(z = 0.0)
`;
  const { bindings } = infer(Q5);
  assert.equal(massOf(typeOf(bindings, 'KS')), T.MASS_UNKNOWN);
  assert.equal(massOf(typeOf(bindings, 'A')), T.MASS_UNKNOWN);
  // §04 "Reification": a draw needs a normalized measure. With the class erased
  // to `deferred` this passed silently.
  const gated = infer(Q5 + 'd = draw(A)\n');
  assert.ok(gated.errors.some((e: any) =>
    /draw requires a probability measure.*%unknown/.test(e.message)),
  'got: ' + gated.errors.map((e: any) => e.message).join(' | '));
  // Control: an applied fan-out of Markov kernels still draws clean.
  const ok = infer(SHARED + 'A = KJ(z = 0.0)\nd = draw(A)\n');
  assert.deepEqual(ok.errors.map((e: any) => e.message), []);
});

test('carrying a class through application is not fan-out-specific, and a '
  + 'non-kernel callee still has no verdict', () => {
  // Any applied kernel benefits: a kernel-first `jointchain` applied at its
  // input now reports the chain's class instead of losing it.
  const chain = infer(`
p = elementof(reals)
q0 ~ Normal(mu = p, sigma = 1.0)
K0 = kernelof(q0, p = p)
K1 = y -> Normal(mu = y, sigma = 1.0)
CH = jointchain(K0, K1)
A = CH(p = 0.0)
`);
  assert.deepEqual(chain.errors.map((e: any) => e.message), []);
  assert.equal(massOf(typeOf(chain.bindings, 'CH')), T.MASS_NORMALIZED);
  assert.equal(massOf(typeOf(chain.bindings, 'A')), T.MASS_NORMALIZED);
  // A callee that is not kernel-typed answers `deferred` — §11's "not yet
  // inferred", which is what the `broadcasted` wrapper's untyped head gives and
  // what it gave before this arm existed. `deferred` here is a gap, not a claim.
  const wrapped = infer(`
f = x -> Normal(mu = x, sigma = 1.0)
bc = broadcasted(f)
M = bc([1.0, 2.0])
`);
  assert.equal(massOf(typeOf(wrapped.bindings, 'M')), T.MASS_DEFERRED);
});

// ── sampling the applied fan-out ────────────────────────────────────────────

test('sampling the applied fan-out draws the shared node ONCE, so the '
  + 'coordinates are correlated', async () => {
  // Cov(p, q) = Var(u | z) = 1 under the retained law and 0 if each component
  // redrew its own copy. At 4000 atoms the sampling error on the covariance is
  // ~0.04, so the two hypotheses are ~25 standard errors apart.
  const { ctx } = ctxFor(SHARED + 'S = KJ(z = 0.0)\n', 4000);
  const m = await materialiser.materialiseMeasure('S', ctx);
  const p = m.fields.p.samples, q = m.fields.q.samples;
  assert.equal(p.length, 4000);
  let sp = 0, sq = 0, spq = 0;
  for (let i = 0; i < p.length; i++) { sp += p[i]; sq += q[i]; spq += p[i] * q[i]; }
  const cov = spq / p.length - (sp / p.length) * (sq / p.length);
  assert.ok(Math.abs(cov - 1) < 0.15,
    `sample covariance ${cov}; the retained law has Cov = Var(u | z) = 1, a `
    + 'per-component copy would give 0');
});
