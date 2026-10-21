'use strict';

// §06 kernel fan-out applies each component at its own boundary and takes
// the independent product of the resulting measures. Dependence inside one
// kernel record remains intact. The density oracles are Gaussian products
// and the correlated record control has covariance [[2,1],[1,2]].

const test = require('node:test');
const assert = require('node:assert');
const { processSource, materialiser, types: T } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const CORRELATED_AT_0 = -3.3871832107434;
const PRODUCT_AT_0 = -Math.log(4 * Math.PI) - 0.5;
const PRODUCT_AT_07 = -Math.log(4 * Math.PI) - (0.3 ** 2 + 1.7 ** 2) / 4;
const F64_TOL = 1e-14;

// Both components absorb u separately when applied at the fanned input.
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

// Boundary isolation is exercised numerically in kernel-joint-w1-refusal.test.ts.

test('differently-named kernel inputs contribute to the input union', () => {
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

test('finite weighted law components retain the product mass class', () => {
  const { bindings, errors } = infer(`
z = elementof(reals)
u ~ StudentT(nu = 3.0)
e ~ Normal(0.0, 1.0)
K1 = functionof(weighted(2.0, lawof(u)), z = z)
K2 = functionof(weighted(3.0, lawof(u + e)), z = z)
KJ = joint(p = K1, q = K2)
`);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.equal(massOf(typeOf(bindings, 'KJ')), T.MASS_FINITE);
});

test('weighted kernels retain the product mass class across an explicit boundary', () => {
  // §04 "Specifying reification boundaries" replaces boundary nodes with fresh
  // `elementof` inputs, so `w` is not in either component's trace: below the
  // boundary they are {x1} and {x2}, disjoint. Given the fanned `m` the
  // components are independent and the product (mass 6, `finite`) is exact.
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

test('weighted law products retain their finite mass at arity zero', async () => {
  const src = `
u ~ Normal(0.0, 1.0)
J = joint(a = weighted(2.0, lawof(u)), b = weighted(3.0, lawof(u)))
ld = logdensityof(J, record(a = 0.0, b = 1.0))
`;
  const { bindings, errors } = infer(src);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.equal(massOf(typeOf(bindings, 'J')), T.MASS_FINITE);
  const oracle = Math.log(6) - Math.log(2 * Math.PI) - 0.5;
  assert.ok(Math.abs(await scoreOf(src) - oracle) < F64_TOL);
});

// ── density of the APPLIED fan-out ──────────────────────────────────────────

test('fan-out components independently absorb their common source ancestry', async () => {
  const got = await scoreOf(SHARED
    + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))\n');
  assert.ok(Math.abs(got - PRODUCT_AT_0) < F64_TOL,
    `got ${got}, product of two N(0,√2) laws ${PRODUCT_AT_0}`);
});

test('the applied fan-out reads a NON-ZERO input', async () => {
  const got = await scoreOf(SHARED
    + 'ld = logdensityof(KJ(z = 0.7), record(p = 1.0, q = -1.0))\n');
  assert.ok(Math.abs(got - PRODUCT_AT_07) < F64_TOL,
    `got ${got}, oracle ${PRODUCT_AT_07} (both marginal means shift to 0.7, `
    + 'so an ignored input would score the z = 0 value)');
});

test('the POSITIONAL spelling scores the same law over a vector variate', async () => {
  const got = await scoreOf(SHARED
    + 'V = joint(K1, K2)\nld = logdensityof(V(z = 0.0), [1.0, -1.0])\n');
  assert.ok(Math.abs(got - PRODUCT_AT_0) < F64_TOL, `got ${got}, oracle ${PRODUCT_AT_0}`);
});

test('separate source ancestry gives the same product of marginal laws', async () => {
  const got = await scoreOf(DISJOINT
    + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))\n');
  assert.ok(Math.abs(got - PRODUCT_AT_0) < F64_TOL);
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

test('repeating one kernel gives two independent density coordinates', async () => {
  const got = await scoreOf(`
z = elementof(reals)
a1 ~ Normal(z, 1.0)
K1 = kernelof(a1, z = z)
KJ = joint(p = K1, q = K1)
ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))
`);
  assert.ok(Math.abs(got - (-Math.log(2 * Math.PI) - 1)) < F64_TOL);
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
  assert.ok(Math.abs(withLatent - CORRELATED_AT_0) < F64_TOL,
    `got ${withLatent}, correlated record law ${CORRELATED_AT_0}`);
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

// ── aliases, lambda components and nested fan-out ─────────────────────────

test('a component reached through an ALIAS scores the same law (§04 "Aliasing '
  + 'is just assignment")', async () => {
  const got = await scoreOf(SHARED
    + 'G = K1\nKA = joint(p = G, q = K2)\n'
    + 'ld = logdensityof(KA(z = 0.0), record(p = 1.0, q = -1.0))\n');
  assert.ok(Math.abs(got - PRODUCT_AT_0) < F64_TOL, `got ${got}, oracle ${PRODUCT_AT_0}`);
});

test('named lambda kernels contribute independent density factors', async () => {
  const src = `
KL1 = z -> Normal(mu = z, sigma = 1.0)
KL2 = z -> Normal(mu = z, sigma = 2.0)
KJ = joint(p = KL1, q = KL2)
ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))
`;
  const { bindings, errors } = infer(src);
  assert.deepEqual(errors.map((e: any) => e.message), []);
  assert.deepEqual(inputNamesOf(typeOf(bindings, 'KJ')), ['z']);
  const oracle = -Math.log(2 * Math.PI) - Math.log(2) - 0.625;
  assert.ok(Math.abs(await scoreOf(src) - oracle) < F64_TOL);
});

for (const [label, component, inputs] of [
  ['lambda', 'z -> Normal(mu = z, sigma = 2.0)', 'z = 3.0'],
  ['fn', 'fn(Normal(mu = _, sigma = 2.0))', 'z = 3.0, arg1 = 3.0'],
]) {
  test(`inline ${label} kernel components score their independent output laws`, async () => {
    const got = await scoreOf(`
z = elementof(reals)
K = joint(a = functionof(Normal(mu = z, sigma = 1.0), z = z), b = ${component})
ld = logdensityof(K(${inputs}), record(a = 4.0, b = 2.0))
`);
    const oracle = -Math.log(2 * Math.PI) - Math.log(2) - 0.625;
    assert.ok(Math.abs(got - oracle) < F64_TOL);
  });
}

test('nested kernel products score simple factors and refuse unsupported nested marginals', async () => {
  const pre = `
z = elementof(reals)
u ~ Normal(z, 1.0)
a1 ~ Normal(u, 1.0)
a2 ~ Normal(u, 1.0)
a3 ~ Normal(u, 1.0)
K1 = kernelof(a1, z = z)
K2 = kernelof(a2, z = z)
K3 = kernelof(a3, z = z)
`;
  for (const decl of [
    'KI = joint(r = K2, s = K3)\nKN = joint(p = K1, q = KI)',
    'KN = joint(p = K1, q = joint(r = K2, s = K3))',
  ]) {
    const query = '\nld = logdensityof(KN(z = 0.0), record(p = 1.0, q = record(r = 1.0, s = -1.0)))\n';
    const got = await scoreOf(pre.replace('u ~ Normal(z, 1.0)', 'u = z') + decl + query);
    const oracle = -1.5 * Math.log(2 * Math.PI) - 1.5;
    assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
    // The hierarchy would give three independent N(z,√2) factors. The current
    // nested marginal recognizer safely refuses that shape.
    await assert.rejects(() => scoreOf(pre + decl + query),
      /marginalises the stochastic ancestor/);
  }
});

test('applying an ALL-MEASURE joint is a static error — it has no inputs', () => {
  const { errors } = infer(`
J = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Normal(mu = 0.0, sigma = 1.0))
ld = logdensityof(J(z = 0.0), record(a = 1.0, b = -1.0))
`);
  assert.ok(errors.some((e: any) => /"J" is not callable/.test(e.message)),
    'got: ' + errors.map((e: any) => e.message).join(' | '));
});

// ── each component keeps its own boundary cut ──────────────────────────

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

test('nested boundary scopes preserve both input-dependent marginal densities', async () => {
  for (const [t, oracle] of [[0, -2.934450656689318], [5, -11.684450656689316]]) {
    const got = await scoreOf(NESTED_BOUNDARY
      + `ld = logdensityof(KJ(s = 0.0, t = ${t}.0), record(p = 1.0, q = -1.0))\n`);
    assert.ok(Math.abs(got - oracle) < F64_TOL, `t=${t}: got ${got}, oracle ${oracle}`);
  }
});

test('nested boundary scopes preserve both input-dependent sampling laws', async () => {
  const { ctx } = ctxFor(NESTED_BOUNDARY + 'S = KJ(s = 0.0, t = 5.0)\n', 4096);
  const { p, q } = (await ctx.getMeasure('S')).fields;
  for (const [xs, mu, variance] of [[p.samples, 0, 1], [q.samples, 5, 2]]) {
    const mean = xs.reduce((sum: number, x: number) => sum + x, 0) / xs.length;
    const v = xs.reduce((sum: number, x: number) => sum + (x - mean) ** 2, 0) / xs.length;
    assert.ok(Math.abs(mean - mu) < 0.15);
    assert.ok(Math.abs(v - variance) < 0.25);
  }
});

test('overlapping source ancestry keeps separate component boundary cuts', async () => {
  const got = await scoreOf(`
z = elementof(reals)
v ~ Normal(z, 1.0)
u ~ Normal(v, 1.0)
a1 ~ Normal(u, 1.0)
a2 ~ Normal(u, 1.0)
K1 = kernelof(a1, s = v)
K2 = kernelof(a2, t = z)
KJ = joint(p = K1, q = K2)
ld = logdensityof(KJ(s = 0.0, t = 5.0), record(p = 1.0, q = -1.0))
`);
  // K1(0)=N(0,√2), K2(5)=N(5,√3).
  const oracle = -Math.log(2 * Math.PI) - 0.5 * Math.log(6) - 0.25 - 6;
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
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
  assert.ok(Math.abs(got - PRODUCT_AT_0) < F64_TOL, `got ${got}, oracle ${PRODUCT_AT_0}`);
});

// ── the draw classifier must stay confined to reification closures ──────────

test('two distinct inline draw(m) components are INDEPENDENT, not one draw',
  async () => {
    // Separate draw calls inside one law provide independent coordinates.
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
  // preserving the output class keeps the applied draw gate meaningful.
  const applied = infer(SHARED + 'A = KJ(z = 0.0)\n');
  assert.equal(massOf(typeOf(applied.bindings, 'KJ')), T.MASS_NORMALIZED);
  assert.equal(massOf(typeOf(applied.bindings, 'A')), T.MASS_NORMALIZED);
  const scored = infer(SHARED
    + 'ld = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))\n');
  assert.equal(massOf(typeOf(scored.bindings, 'KJ')), T.MASS_NORMALIZED);
});

test('the finite product class survives application and the draw gate', () => {
  const src = `
z = elementof(reals)
u ~ StudentT(nu = 3.0)
e ~ Normal(0.0, 1.0)
K1 = functionof(weighted(2.0, lawof(u)), z = z)
K2 = functionof(weighted(3.0, lawof(u + e)), z = z)
KS = joint(p = K1, q = K2)
A = KS(z = 0.0)
`;
  const { bindings } = infer(src);
  assert.equal(massOf(typeOf(bindings, 'KS')), T.MASS_FINITE);
  assert.equal(massOf(typeOf(bindings, 'A')), T.MASS_FINITE);
  const gated = infer(src + 'd = draw(A)\n');
  assert.ok(gated.errors.some((e: any) =>
    /draw requires a probability measure.*%finite/.test(e.message)));
  const normalized = infer(SHARED + 'A = KJ(z = 0.0)\nd = draw(A)\n');
  assert.deepEqual(normalized.errors.map((e: any) => e.message), []);
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

test('sampling the fan-out independently copies each absorbed law', async () => {
  const { ctx } = ctxFor(SHARED + 'S = KJ(z = 0.0)\n', 4096);
  const { p, q } = (await ctx.getMeasure('S')).fields;
  let sp = 0, sq = 0, spq = 0;
  for (let i = 0; i < p.samples.length; i++) {
    sp += p.samples[i]; sq += q.samples[i]; spq += p.samples[i] * q.samples[i];
  }
  const n = p.samples.length;
  const cov = spq / n - (sp / n) * (sq / n);
  assert.ok(Math.abs(cov) < 0.2, `sample covariance ${cov}, oracle 0`);
});
