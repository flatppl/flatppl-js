'use strict';

// §04 reifies a law with its internal dependence intact; §06 composes independent
// copies of law values. A raw constructor's free parameters remain conditional
// until an enclosing lawof absorbs them.
//
// How the resulting marginal may be evaluated is stated in that section for
// `kchain` — "This is generally intractable; an engine evaluates it in closed
// form, or by enumeration of a discrete latent, and otherwise reports a static
// error" — and reaches this construct only as an ANALOGY. Nothing normative
// forces exact-or-refuse: flatppl-design#72 was closed unmerged and the owner's
// 2026-08-06 call leaves the method unruled for now, superseding the earlier
// 2026-08-05 no-stochastic-estimate decision. These tests therefore pin an
// ENGINEERING CHOICE of this path, not a conformance requirement: a shape it
// cannot answer by one of the two exact devices must REFUSE rather than estimate.
//
// Oracles are INDEPENDENT (Distributions.jl, not the engine's own output, not
// the Rust determiniser):
//   z ~ Normal(0,1); a,b ~ Normal(z,1)  ⇒  MvNormal([0,0], [[2,1],[1,2]])
//     logpdf(0.5, 0.7)                    = -2.5171832107434002
//   theta ~ Normal(0,1); obs ~ Normal(theta,1) ⇒ Normal(0, √2)
//     logpdf(0.5)                         = -1.3280121234846454
//   z ~ Normal(1,2); a ~ Normal(3z+1, 0.5); b ~ Normal(z, 1.5)
//     ⇒ MvNormal([4,1], [[36.25,12],[12,6.25]])
//     logpdf(0.5, 0.7)                    = -4.375464849980595
// (julia> logpdf(MvNormal(mu, Sigma), x) — the covariances are the textbook
// linear-Gaussian moments, re-derived in linear-gaussian.ts's header.)
//
// The CHAIN covariances below are each confirmed twice, by two derivations that
// share no algebra: the MvNormal moments, and the chain rule p(a)·p(b|a) over
// the same draws. The mixtures are the finite sum written out term by term.

const test = require('node:test');
const assert = require('node:assert');
const { ctxFor } = require('./_ctx-factory.ts');
const { materialiser } = require('..');

const MVN_ORACLE = -2.5171832107434002;
const SCALAR_ORACLE = -1.3280121234846454;
const AFFINE_ORACLE = -4.375464849980595;
const PRODUCT_ORACLE = -Math.log(4 * Math.PI) - (0.5 ** 2 + 0.7 ** 2) / 4;
// Both spellings go through the engine's canonical MvNormal closed form
// (density-prims), whose Cholesky ordering differs from Distributions.jl's by
// a couple of ulp — an f64 comparison, not a loosened statistical tolerance.
const F64_TOL = 1e-14;

const RECORD_LAW = `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`;

const JOINT_OF_LAWS = `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
J = joint(a = lawof(a), b = lawof(b))
ld = logdensityof(J, record(a = 0.5, b = 0.7))
`;

const scoreOf = async (src: string, N: number) => {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('ld');
  return m.samples[0];
};

test('a fresh draw from a reified transform has an independent density coordinate', async () => {
  const got = await scoreOf(`
x ~ Normal(0.0, 1.0)
M = functionof(lawof(2.0*x))()
y ~ M
ld = logdensityof(lawof(record(x=x, y=y)), record(x=0.0, y=1.0))
`, 1);
  // Independent N(0,1) and N(0,2), not a diagonal transform of one draw.
  assert.ok(Math.abs(got - (-Math.log(2*Math.PI) - Math.log(2) - 1/8)) < F64_TOL);
  const fresh = await scoreOf(`
x ~ Normal(0.0, 1.0)
M = functionof(lawof(2.0*x))()
y ~ M
z ~ M
ld = logdensityof(lawof(record(y=y, z=z)), record(y=1.0, z=2.0))
`, 1);
  assert.ok(Math.abs(fresh - (-Math.log(2*Math.PI) - 2*Math.log(2) - 5/8)) < F64_TOL);
  // Reusing one fresh draw is still diagonal, including the direct lawof
  // spelling whose fixed measure phase must not erase the draw identity.
  await assert.rejects(() => scoreOf(`
x ~ Normal(0.0, 1.0)
M = lawof(2.0*x)
y ~ M
z = y
ld = logdensityof(lawof(record(x=x, y=y, z=z)), record(x=0.0, y=1.0, z=1.0))
`, 1), (e: any) => e.code === 'CLM_SINGULAR_JOINT');
});

test('draws from a named stochastic constructor retain distinct conditional noise', async () => {
  const prefix = `
p ~ Normal(0.0, 2.0)
M = Normal(p, 1.0)
a ~ M
b ~ M
`;
  const full = await scoreOf(prefix + `
ld = logdensityof(lawof(record(p=p, a=a, b=b)), record(p=0.0, a=1.0, b=2.0))
`, 1);
  assert.ok(Math.abs(full - (-1.5*Math.log(2*Math.PI) - Math.log(2) - 2.5)) < F64_TOL);
  const marginal = await scoreOf(prefix + `
ld = logdensityof(lawof(record(a=a, b=b)), record(a=1.0, b=2.0))
`, 1);
  // Covariance [[5,4],[4,5]] has determinant 9 and quadratic form 1.
  assert.ok(Math.abs(marginal - (-Math.log(2*Math.PI) - Math.log(3) - 0.5)) < F64_TOL);
});

test('shared-ancestor record law scores the analytic MvNormal marginal (N=1)', async () => {
  const got = await scoreOf(RECORD_LAW, 1);
  assert.ok(Math.abs(got - MVN_ORACLE) < F64_TOL,
    `got ${got}, Distributions.jl MvNormal([0,0],[[2,1],[1,2]]) logpdf(0.5,0.7) = ${MVN_ORACLE}`);
});

test('a product of marginal laws differs from one law of the correlated record', async () => {
  const viaJoint = await scoreOf(JOINT_OF_LAWS, 1);
  const viaRecord = await scoreOf(RECORD_LAW, 1);
  assert.ok(Math.abs(viaJoint - PRODUCT_ORACLE) < F64_TOL,
    `got ${viaJoint}, independent N(0,√2) product ${PRODUCT_ORACLE}`);
  assert.ok(Math.abs(viaRecord - MVN_ORACLE) < F64_TOL);
});

test('affine constructor laws retain their marginal when copied into a product', async () => {
  const got = await scoreOf(`
z ~ Normal(0, 1)
x ~ locscale(Normal(0, 1), z, 1)
J = joint(a = lawof(x), b = lawof(x))
ld = logdensityof(J, record(a = 0.5, b = 0.7))
`, 1);
  assert.ok(Math.abs(got - PRODUCT_ORACLE) < F64_TOL);
});

test('nested negative affine scales preserve Gaussian marginal and joint moments', async () => {
  const source = `
z ~ Normal(0, 1)
x ~ locscale(locscale(Normal(0, 1), z, -2), 1, -3)
`;
  // x = 1 - 3z + 6ε has variance 45. The joint (z,x) has determinant 36.
  const product = await scoreOf(source + `
J = joint(a = lawof(x), b = lawof(x))
ld = logdensityof(J, record(a = 1, b = 1))
`, 1);
  assert.ok(Math.abs(product + Math.log(2 * Math.PI * 45)) < F64_TOL);
  const joint = await scoreOf(source + `
ld = logdensityof(lawof(record(z = z, x = x)), record(z = 0, x = 1))
`, 1);
  assert.ok(Math.abs(joint + Math.log(2 * Math.PI * 6)) < F64_TOL);
});

test('an affine transform with a latent scale does not claim a Gaussian marginal', async () => {
  await assert.rejects(() => scoreOf(`
z ~ Normal(0, 1)
x ~ locscale(Normal(0, 1), 0, exp(z))
ld = logdensityof(lawof(x), 1)
`, 1), /constant nonzero scale/);
});

test('an IID matrix integrates one independent Gaussian law per coordinate', async () => {
  const got = await scoreOf(`
z ~ Normal(0, 1)
x ~ Normal(z, 1)
M = iid(lawof(x), [2, 3])
ld = logdensityof(M, rowstack([[0, 0, 0], [0, 0, 0]]))
`, 1);
  assert.ok(Math.abs(got + 3 * Math.log(4 * Math.PI)) < F64_TOL);
});

test('the analytic marginal does not depend on sampleCount (no MC estimate)', async () => {
  const atOne = await scoreOf(RECORD_LAW, 1);
  const atMany = await scoreOf(RECORD_LAW, 250);
  assert.equal(atOne, atMany,
    'a closed-form density is a function of the point alone; an N-dependent '
    + 'value would mean an MC estimator came back');
});

test('a scalar marginal law (lawof of a draw with a stochastic ancestor) is the '
  + 'closed-form Normal(0, √2)', async () => {
  const got = await scoreOf(`
theta ~ Normal(mu = 0.0, sigma = 1.0)
obs ~ Normal(mu = theta, sigma = 1.0)
P = lawof(obs)
ld = logdensityof(P, 0.5)
`, 1);
  assert.ok(Math.abs(got - SCALAR_ORACLE) < F64_TOL,
    `got ${got}, Distributions.jl Normal(0,√2) logpdf(0.5) = ${SCALAR_ORACLE}`);
});

test('an affine location and a non-unit latent scale still close analytically', async () => {
  const got = await scoreOf(`
z ~ Normal(mu = 1.0, sigma = 2.0)
a ~ Normal(mu = 3.0 * z + 1.0, sigma = 0.5)
b ~ Normal(mu = z, sigma = 1.5)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`, 1);
  assert.ok(Math.abs(got - AFFINE_ORACLE) < F64_TOL,
    `got ${got}, Distributions.jl MvNormal([4,1],[[36.25,12],[12,6.25]]) logpdf(0.5,0.7) = ${AFFINE_ORACLE}`);
});

test('the POSITIONAL distribution spelling Normal(mu, sigma) reaches the same '
  + 'closed form', async () => {
  const got = await scoreOf(`
z ~ Normal(0.0, 1.0)
a ~ Normal(z, 1.0)
b ~ Normal(z, 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`, 1);
  assert.ok(Math.abs(got - MVN_ORACLE) < F64_TOL, `got ${got}, oracle ${MVN_ORACLE}`);
});

test('a hierarchy of latents (z2 ~ Normal(z1, 1)) closes analytically too', async () => {
  // z1 ~ N(0,1); z2 ~ N(z1,1); a ~ N(z2,1); b ~ N(z1,1) ⇒ Var(a) = 3,
  // Var(b) = 2, Cov(a,b) = Cov(z2,z1) = 1.
  // julia> logpdf(MvNormal([0,0], [3 1; 1 2]), [0.5,0.7]) = -2.769596022626396
  const got = await scoreOf(`
z1 ~ Normal(mu = 0.0, sigma = 1.0)
z2 ~ Normal(mu = z1, sigma = 1.0)
a ~ Normal(mu = z2, sigma = 1.0)
b ~ Normal(mu = z1, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`, 1);
  assert.ok(Math.abs(got - (-2.769596022626396)) < F64_TOL,
    `got ${got}, Distributions.jl MvNormal([0,0],[[3,1],[1,2]]) logpdf(0.5,0.7) = -2.769596022626396`);
});

test('the POSITIONAL product scores independent marginal laws over a vector variate',
  async () => {
    const got = await scoreOf(`
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
V = joint(lawof(a), lawof(b))
ld = logdensityof(V, [0.5, 0.7])
`, 1);
    assert.ok(Math.abs(got - PRODUCT_ORACLE) < F64_TOL,
      `got ${got}, oracle ${PRODUCT_ORACLE}`);
  });

test('a fixed-phase location and the remaining affine ops (sub / neg / divide) '
  + 'resolve as constant coefficients', async () => {
  // a ~ Normal(-(z/2) - m, 1), b ~ Normal(z, 1) with m = 0.25 fixed
  //   ⇒ mean = (-0.25, 0), Var(a) = 0.25 + 1 = 1.25, Cov = -0.5, Var(b) = 2.
  // julia> logpdf(MvNormal([-0.25,0], [1.25 -0.5; -0.5 2]), [0.5,0.7])
  const oracle = -2.7461199522952877;
  const got = await scoreOf(`
m = 0.25
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = -(z / 2.0) - m, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`, 1);
  assert.ok(Math.abs(got - oracle) < F64_TOL,
    `got ${got}, Distributions.jl MvNormal([-0.25,0],[[1.25,-0.5],[-0.5,2]]) logpdf(0.5,0.7) = ${oracle}`);
});

test('a non-affine op in the location refuses (no linearisation)', async () => {
  const { ctx } = ctxFor(`
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = exp(z), sigma = 1.0)
L = lawof(record(a = a))
ld = logdensityof(L, record(a = 0.5))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /location is not an affine function/);
});

// ── singular joints (§06 "Singular joints") ─────────────────────────────────

const SINGULAR_SAME_DRAW = `
y ~ Normal(mu = 0.0, sigma = 1.0)
S = lawof(record(a = y, b = y))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`;

test('one law of a repeated coordinate refuses the density query', async () => {
  const { ctx } = ctxFor(SINGULAR_SAME_DRAW, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /share the ancestor 'y'.*lower-dimensional subset/s);
});

test('a deterministic transform of another component refuses the density query', async () => {
  const { ctx } = ctxFor(`
y ~ Normal(mu = 0.0, sigma = 1.0)
w = exp(y)
T = lawof(record(a = y, b = w))
ld = logdensityof(T, record(a = 0.5, b = 0.9))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /share the ancestor 'y'/);
});

test('sampling a singular joint stays legal (§06: "Sampling is well-defined")', async () => {
  const { ctx } = ctxFor(SINGULAR_SAME_DRAW, 64);
  const m = await materialiser.materialiseMeasure('S', ctx);
  const a = m.fields.a.samples;
  const b = m.fields.b.samples;
  assert.equal(a.length, 64);
  for (let i = 0; i < a.length; i++) {
    assert.equal(a[i], b[i], 'the same draw referenced twice samples on the diagonal');
  }
});

test('joint(m, m) over a CONSTRUCTOR measure is NOT singular — it stays the '
  + 'independent product (§04 "Identity law")', async () => {
  const logpdfNormal = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;
  const got = await scoreOf(`
m = Normal(mu = 0.0, sigma = 1.0)
Q = joint(a = m, b = m)
ld = logdensityof(Q, record(a = 0.5, b = 0.9))
`, 1);
  const expect = logpdfNormal(0.5) + logpdfNormal(0.9);
  assert.ok(Math.abs(got - expect) < F64_TOL,
    `got ${got}, independent product ${expect} — the singularity gate must not `
    + 'fire on two independent draws of one constructor measure');
});

// ── non-analytic marginals refuse rather than estimate ──────────────────────

test('a non-Gaussian ancestor refuses the density query at ANY sampleCount', async () => {
  const src = `
z ~ Gamma(shape = 2.0, rate = 2.0)
y ~ Normal(mu = z, sigma = 1.0)
G = lawof(record(y = y))
ld = logdensityof(G, record(y = 0.5))
`;
  for (const N of [1, 5000]) {
    const { ctx } = ctxFor(src, N);
    await assert.rejects(async () => ctx.getMeasure('ld'),
      /no exact answer for it here.*'z' is a 'Gamma'/s,
      `sampleCount ${N} must refuse, not estimate`);
  }
});

test('a latent in the SCALE is not linear-Gaussian and refuses', async () => {
  const { ctx } = ctxFor(`
s ~ Normal(mu = 2.0, sigma = 0.1)
a ~ Normal(mu = 0.0, sigma = s)
L = lawof(record(a = a))
ld = logdensityof(L, record(a = 0.5))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /scale is not a positive constant/);
});

test('a nonlinear (latent × latent) location is not affine and refuses', async () => {
  const { ctx } = ctxFor(`
u ~ Normal(mu = 0.0, sigma = 1.0)
v ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = u * v, sigma = 1.0)
L = lawof(record(a = a))
ld = logdensityof(L, record(a = 0.5))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /location is not an affine function/);
});

// ── a caller-substituted boundary must not be silently ignored ─────────────
//
// `lowerMeasure(name, ctx, {boundaries})` is the viewer's kernel/profile route
// (spec §04 functionof boundary substitution): the CALLER supplies the value for
// a named input instead of the model's own. The closed form reads its constants
// out of `bindings`/`fixedValues`, so a marginal that depends on a substituted
// name must refuse — answering from the un-substituted value would be a
// confidently wrong density, and the MC reduce this wave replaced at least fed
// the boundary to the worker. Threading the substitution into the moments is the
// capability fix, filed in flatppl-dev/TODO-flatppl-js.md.

const OVERRIDDEN_MEAN = `
m = 0.0
z ~ Normal(mu = m, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`;

const reduceFor = (src: string, opts?: any) => {
  const clm = require('../clm.ts');
  const { ctx } = ctxFor(src, 1);
  return clm.lowerMeasure('L', ctx, opts).reduce;
};

test('an explicit boundary the marginal depends on refuses instead of scoring '
  + 'the un-substituted value', () => {
  // Baseline: with no substitution the same model closes analytically, and the
  // mean comes from the model's own `m = 0.0`.
  const plain = reduceFor(OVERRIDDEN_MEAN);
  assert.equal(plain.method, 'analytic-gaussian');
  assert.deepEqual(plain.gaussian.mean, [0, 0]);
  // Substituting m = 5 makes the true mean [5, 5]. The closed form cannot see
  // the fed value, so it must refuse rather than return [0, 0] again.
  const fed = reduceFor(OVERRIDDEN_MEAN, { boundaries: { m: 5 } });
  assert.equal(fed.method, 'refuse');
  assert.match(fed.reason, /depends on 'm', which the caller substitutes/);
});

test('a free profile axis the marginal depends on refuses the same way', () => {
  const fed = reduceFor(OVERRIDDEN_MEAN, { freeInputs: ['m'] });
  assert.equal(fed.method, 'refuse');
  assert.match(fed.reason, /depends on 'm', which the caller substitutes/);
});

test('a substituted input the marginal does NOT depend on leaves the closed '
  + 'form intact', () => {
  const fed = reduceFor(`
k = 3.0
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`, { boundaries: { k: 9 } });
  assert.equal(fed.method, 'analytic-gaussian');
  assert.deepEqual(fed.gaussian.cov, [[2, 1], [1, 2]]);
});

// ── hierarchical chains: a component in another component's location ───────
//
// z ~ N(0,1); a ~ N(z,1); b ~ N(a,1) is the plainest hierarchical shape there
// is, and it IS linear-Gaussian: a = z + ε_a and b = a + ε_b, so Var(a) = 2,
// Var(b) = 3 and Cov(a,b) = Var(a) = 2, giving MvNormal([0,0], [[2,2],[2,3]]).
//
// Two derivations that share no algebra agree on that covariance:
//   julia> logpdf(MvNormal([0,0], [2 2; 2 3]), [0.5,0.7]) = -2.2669506566893185
//   julia> logpdf(Normal(0,√2), 0.5) + logpdf(Normal(0.5,1), 0.7)
//                                                         = -2.266950656689318
// The second is the chain rule p(a)·p(b|a) over the same draws — it never forms
// a covariance matrix — so it is an independent check that the recogniser keeps
// the noise `a` and `b` SHARE. Dropping it (composing b's location down to `z`
// and giving b fresh noise) would give [[2,1],[1,2]] and a different number.
const CHAIN_ORACLE = -2.2669506566893185;

const CHAIN_DRAWS = `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = a, sigma = 1.0)
`;

test('a component in another component\'s location closes analytically '
  + '(hierarchical chain, record spelling)', async () => {
  const got = await scoreOf(CHAIN_DRAWS + `
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`, 1);
  assert.ok(Math.abs(got - CHAIN_ORACLE) < F64_TOL,
    `got ${got}, MvNormal([0,0],[[2,2],[2,3]]) logpdf(0.5,0.7) = ${CHAIN_ORACLE}`);
});

test('record and vector spellings of one chain law score the SAME marginal — a component '
  + 'is identified by its DRAW, not by its label', async () => {
  // The record spelling exposes the field under the draw's own name `a`; the
  // other three do not, and before draw-identity keying the recogniser
  // integrated `a` out as a latent AND redrew it as a component, silently
  // scoring MvNormal([0,0],[[2,1],[1,3]]) — a WRONG number, not a refusal.
  const spellings = [
    ['record', 'L = lawof(record(a = a, b = b))', 'record(a = 0.5, b = 0.7)'],
    ['relabelled record', 'L = lawof(record(p = a, q = b))', 'record(p = 0.5, q = 0.7)'],
    ['vector', 'L = lawof([a, b])', '[0.5, 0.7]'],
  ];
  for (const [what, decl, point] of spellings) {
    const got = await scoreOf(CHAIN_DRAWS + decl + `\nld = logdensityof(L, ${point})\n`, 1);
    assert.ok(Math.abs(got - CHAIN_ORACLE) < F64_TOL,
      `${what}: got ${got}, oracle ${CHAIN_ORACLE}`);
  }
});

test('the chain marginal does not depend on sampleCount', async () => {
  const src = CHAIN_DRAWS + `
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`;
  assert.equal(await scoreOf(src, 1), await scoreOf(src, 250));
});

test('an affine chain composes the maps through the intermediate node', async () => {
  // z ~ N(1,2); a ~ N(2z-1, 0.5); b ~ N(3a+1, 1.5)
  //   mean(a) = 1, Var(a) = 4*4 + 0.25 = 16.25
  //   mean(b) = 4, Var(b) = 9*16.25 + 2.25 = 148.5, Cov(a,b) = 3*16.25 = 48.75
  // julia> logpdf(MvNormal([1,4], [16.25 48.75; 48.75 148.5]), [0.5,0.7])
  //   = -4.3650809365976935
  // cross-check (chain rule, no covariance matrix):
  //   logpdf(Normal(1,√16.25), 0.5) + logpdf(Normal(3*0.5+1, 1.5), 0.7)
  //   = -4.365080936597691
  const oracle = -4.3650809365976935;
  const got = await scoreOf(`
z ~ Normal(mu = 1.0, sigma = 2.0)
a ~ Normal(mu = 2.0 * z - 1.0, sigma = 0.5)
b ~ Normal(mu = 3.0 * a + 1.0, sigma = 1.5)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`, 1);
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
});

test('an UNEXPOSED intermediate latent is closed over too (transitive '
  + 'ancestor closure)', async () => {
  // z1 ~ N(0,1); z2 ~ N(z1,1); a,b ~ N(z2,1): only `z2` is a body self-ref, so
  // `z1` reaches the recogniser through z2's law alone. Var(z2) = 2 ⇒
  // Var(a) = Var(b) = 3, Cov(a,b) = Var(z2) = 2.
  // julia> logpdf(MvNormal([0,0], [3 2; 2 3]), [0.5,0.7]) = -2.7245960226263954
  // cross-check: logpdf(Normal(0,√3),0.5) + logpdf(Normal(0.5*2/3, √(3-4/3)),0.7)
  //   = -2.724596022626396
  const oracle = -2.7245960226263954;
  const got = await scoreOf(`
z1 ~ Normal(mu = 0.0, sigma = 1.0)
z2 ~ Normal(mu = z1, sigma = 1.0)
a ~ Normal(mu = z2, sigma = 1.0)
b ~ Normal(mu = z2, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`, 1);
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
});

// ── enumeration of a finite discrete latent (§06's second exact device) ────
//
// §06 "Density of composed measures" states the marginal rule for `kchain`: "an
// engine evaluates it in closed form, or by enumeration of a discrete latent,
// and otherwise reports a static error". Enumeration is DETERMINISTIC and EXACT
// — density(y) = Σ_k P(z = k) · density(y | z = k) — so it satisfies this path's
// exact-or-refuse choice without any Monte Carlo. The support cap mirrors the
// determiniser's (`flatppl-rust/crates/determinizer/src/marginal.rs`,
// MAX_ATOMS = 256), which is a CAP VALUE reference only, not a semantics oracle;
// the cross-latent product cap is this engine's own, tighter reading.

test('a Bernoulli ancestor enumerates to the exact two-atom mixture', async () => {
  // julia> log(0.7*pdf(Normal(0,1),0.5) + 0.3*pdf(Normal(1,1),0.5))
  const oracle = -1.0439385332046727;
  const got = await scoreOf(`
s ~ Bernoulli(p = 0.3)
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 0.5)
`, 1);
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
});

test('the enumerated mixture does not depend on sampleCount (no MC estimate)',
  async () => {
    const src = `
s ~ Bernoulli(p = 0.3)
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 0.5)
`;
    assert.equal(await scoreOf(src, 1), await scoreOf(src, 400));
  });

test('a Categorical ancestor enumerates its 1-based atoms (§08)', async () => {
  // §08 Categorical: support interval(1, n) — "Categories are numbered starting
  // from 1". Atoms {1,2,3} with p = [0.2,0.5,0.3].
  // julia> log(0.2*pdf(Normal(1,1),1.4) + 0.5*pdf(Normal(2,1),1.4)
  //            + 0.3*pdf(Normal(3,1),1.4))
  const oracle = -1.296297984007803;
  const got = await scoreOf(`
s ~ Categorical(p = [0.2, 0.5, 0.3])
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 1.4)
`, 1);
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
});

test('a Categorical0 ancestor enumerates its 0-based atoms (§08)', async () => {
  // §08 Categorical0: support interval(0, n-1), density p_{k+1}. Atoms {0,1,2}
  // for the SAME p the 1-based test above uses, scored at the SAME point, so the
  // pair pins the offset rather than just the arithmetic.
  // julia> log(0.2*pdf(Normal(0,1),1.4) + 0.5*pdf(Normal(1,1),1.4)
  //            + 0.3*pdf(Normal(2,1),1.4))
  //   = -1.1582096163653917
  // The 1-based sibling scores -1.296297984007803, so an off-by-one in the
  // offset moves this by 0.138 — far outside F64_TOL.
  const oracle = -1.1582096163653917;
  const got = await scoreOf(`
s ~ Categorical0(p = [0.2, 0.5, 0.3])
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 1.4)
`, 1);
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
});

test('a Binomial ancestor enumerates its n+1 atoms', async () => {
  // julia> log(sum(pdf(Binomial(4,0.4),k) * pdf(Normal(k,1),1.7) for k in 0:4))
  const oracle = -1.276691405132729;
  const got = await scoreOf(`
s ~ Binomial(n = 4, p = 0.4)
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 1.7)
`, 1);
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
});

test('a discrete ancestor SHARED by two components is a mixture of MvNormals',
  async () => {
    // Given s the components are independent, so each block is MvNormal([s,s], I).
    // julia> log(0.7*pdf(MvNormal([0,0],I),[0.5,0.7])
    //            + 0.3*pdf(MvNormal([1,1],I),[0.5,0.7]))
    const oracle = -2.143569046102502;
    const got = await scoreOf(`
s ~ Bernoulli(p = 0.3)
a ~ Normal(mu = s, sigma = 1.0)
b ~ Normal(mu = s, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5, b = 0.7))
`, 1);
    assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
  });

test('enumeration and the chain recogniser compose: a discrete root above a '
  + 'Gaussian chain', async () => {
  // s ~ Bernoulli(0.3); z ~ N(s,1); a ~ N(z,1); b ~ N(a,1). Per atom the block
  // is MvNormal([s,s], [[2,2],[2,3]]).
  // julia> log(0.7*pdf(MvNormal([0,0],[2 2; 2 3]),[0.9,1.9])
  //            + 0.3*pdf(MvNormal([1,1],[2 2; 2 3]),[0.9,1.9]))
  const oracle = -2.822642636382475;
  const got = await scoreOf(`
s ~ Bernoulli(p = 0.3)
z ~ Normal(mu = s, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = a, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.9, b = 1.9))
`, 1);
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
});

test('two discrete ancestors enumerate their product support', async () => {
  // s ~ Bernoulli(0.3) × t ~ Categorical([0.2,0.5,0.3]) = 6 atom combinations.
  // julia> log(sum(pdf(Bernoulli(0.3),k) * [0.2,0.5,0.3][j]
  //                * pdf(Normal(k + 2j, 1), 3.1) for k in 0:1, j in 1:3))
  const oracle = -1.841643789646692;
  const got = await scoreOf(`
s ~ Bernoulli(p = 0.3)
t ~ Categorical(p = [0.2, 0.5, 0.3])
x ~ Normal(mu = s + 2.0 * t, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 3.1)
`, 1);
  assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
});

test('a support above the enumeration cap refuses rather than enumerating',
  async () => {
    const { ctx } = ctxFor(`
s ~ Binomial(n = 400, p = 0.5)
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 0.5)
`, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'),
      /401 atoms, above the enumeration cap of 256/);
  });

test('an INFINITE discrete ancestor is not enumerable and refuses', async () => {
  const { ctx } = ctxFor(`
s ~ Poisson(rate = 3.0)
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 0.5)
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /'s' is a 'Poisson', not a Normal/);
});

test('the product support of several discrete ancestors is capped too', async () => {
  // 21 × 21 = 441 atom combinations, each latent under the per-latent cap.
  const { ctx } = ctxFor(`
s ~ Binomial(n = 20, p = 0.5)
t ~ Binomial(n = 20, p = 0.5)
x ~ Normal(mu = s + t, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 0.5)
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /441 atom combinations, above the cap of 256/);
});

test('a zero-mass atom drops out of the mixture instead of contributing -Inf',
  async () => {
    // Categorical([0, 1]) puts no mass on atom 1, so the sum is the atom-2 term
    // alone: log(1 * pdf(Normal(2,1), 1.4)) = logpdf(Normal(2,1), 1.4).
    // julia> logpdf(Normal(2,1), 1.4) = -1.0989385332046728
    const oracle = -1.0989385332046728;
    const got = await scoreOf(`
s ~ Categorical(p = [0.0, 1.0])
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 1.4)
`, 1);
    assert.ok(Math.abs(got - oracle) < F64_TOL, `got ${got}, oracle ${oracle}`);
  });

// A discrete latent whose own parameters are stochastic has no statically known
// support, so there is nothing to enumerate — refuse per distribution.

test("a Bernoulli ancestor with a stochastic `p` is not enumerable", async () => {
  const { ctx } = ctxFor(`
u ~ Normal(mu = 0.5, sigma = 0.1)
s ~ Bernoulli(p = u)
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 0.5)
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /`p` is not a constant in \[0, 1\]/);
});

test("a Binomial ancestor with a stochastic `p` is not enumerable", async () => {
  const { ctx } = ctxFor(`
u ~ Normal(mu = 0.5, sigma = 0.1)
s ~ Binomial(n = 4, p = u)
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 0.5)
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /`n`\/`p` are not constants/);
});

test("a Categorical ancestor with a stochastic `p` has no static category count",
  async () => {
    const { ctx } = ctxFor(`
w ~ Dirichlet(alpha = [1.0, 1.0, 1.0])
s ~ Categorical(p = w)
x ~ Normal(mu = s, sigma = 1.0)
P = lawof(x)
ld = logdensityof(P, 0.5)
`, 1);
    await assert.rejects(async () => ctx.getMeasure('ld'),
      /`p` is not a constant probability vector/);
  });

// ── draw-identity guards, exercised on the recogniser directly ─────────────
//
// Both are SAFETY guards against double-counting a shared draw. Neither is
// reachable through source today (`lowerMeasure` recovers a positional body's
// identities from its tuple derivation, and `_refuseIfSingular` rejects a
// same-draw joint before the body is built), so they are driven here from a
// hand-built body — the shape a by-IR `lowerMeasure` call could still present.

const recogniserOn = (body: any, marg: string[], src: string, identity: any) => {
  const lg = require('../linear-gaussian.ts');
  const { ctx } = ctxFor(src, 1);
  return lg.recogniseGaussianMarginal(body, marg, ctx, undefined, identity);
};

const NORMAL_OF = (name: string) => ({
  kind: 'call', op: 'Normal',
  kwargs: { mu: { kind: 'ref', ns: 'self', name }, sigma: { kind: 'lit', value: 1 } },
});

const TWO_DRAWS = `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
`;

test('a multi-component body whose components carry no draw identity refuses',
  () => {
    const body = { kind: 'call', op: 'joint', args: [NORMAL_OF('z'), NORMAL_OF('z')] };
    const g = recogniserOn(body, ['z'], TWO_DRAWS, { keyOf: (n: string) => n, componentKeys: null });
    assert.match(g.refuse, /carries no draw identity/);
  });

test('two components resolving to ONE draw refuses as singular', () => {
  const body = {
    kind: 'call', op: 'joint',
    fields: [{ name: 'p', value: NORMAL_OF('z'), source: 'a' },
             { name: 'q', value: NORMAL_OF('z'), source: 'a' }],
  };
  const g = recogniserOn(body, ['z'], TWO_DRAWS, { keyOf: (n: string) => n });
  assert.match(g.refuse, /are the same draw, so the joint law is singular/);
});

test('omitting identity.keyOf THROWS rather than falling back to name keying',
  () => {
    // Defaulting keyOf to `nm => nm` is name keying, which is what scored
    // [[2,1],[1,3]] for the chain before this recogniser was rewritten. A caller
    // that cannot supply draw identities must get an error, never a number.
    const body = {
      kind: 'call', op: 'joint',
      fields: [{ name: 'a', value: NORMAL_OF('z'), source: 'a' },
               { name: 'b', value: NORMAL_OF('a'), source: 'b' }],
    };
    for (const bad of [undefined, {}, { componentKeys: null }]) {
      assert.throws(() => recogniserOn(body, ['z'], TWO_DRAWS, bad),
        /identity\.keyOf is required/);
    }
  });

// A stochastic binding with NO inferred type cannot be classified as a draw
// variate or a constructor measure. Answering "constructor" would be fail-open:
// the component gets a fresh coordinate and a singular joint SCORES instead of
// refusing. Only internal lifted names are untyped today, so the classifier
// asserts that rather than trusting it. Not reachable from source — driven from a
// hand-built ctx, like the recogniser guards above.

test('an untyped stochastic binding under a USER-FACING name throws rather than '
  + 'being read as a constructor', () => {
  const clm = require('../clm.ts');
  const ctxWith = (nm: string) => ({
    derivations: { J: { kind: 'record', fields: { a: nm, b: nm } } },
    bindings: new Map([[nm, { phase: 'stochastic' }]]),   // no inferredType
    fixedValues: new Map(),
  });
  assert.throws(() => clm.lowerMeasure('J', ctxWith('q')),
    /stochastic binding 'q' has no inferred type/);
  // The control: the same shape under an internal lifted name is the ordinary
  // untyped-constructor case and must NOT hit the guard.
  try {
    clm.lowerMeasure('J', ctxWith('__anon7'));
  } catch (e: any) {
    assert.doesNotMatch(String(e.message), /has no inferred type/,
      'an internal lifted name is the expected untyped case, not a guard trip');
  }
});

// ── the point scored must match the variate ────────────────────────────────
//
// §06 "Singular joints" sets the pattern for the whole density path: "a static
// error where statically detectable, and is otherwise refused by the engine".
// A mismatched point IS statically detectable (the analyzer emits a
// `logdensityof: arg 2 expects …` diagnostic), and the closed form refuses it
// again at score time rather than reading a NaN out of the wrong shape.

const compileErrors = (src: string) => {
  const { processSource } = require('..');
  return (processSource(src).diagnostics || [])
    .filter((d: any) => d.severity === 'error').map((d: any) => d.message);
};

test('a record-variate marginal refuses a positional point', async () => {
  const src = `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, [0.5, 0.7])
`;
  assert.match(compileErrors(src).join('\n'), /expects record with fields/);
  const { ctx } = ctxFor(src, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'), /the point scored is not a record/);
});

test('a record-variate marginal refuses a point missing a field', async () => {
  const { ctx } = ctxFor(`
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 0.5))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'), /no finite field 'b'/);
});

test('a positional marginal refuses a short vector and a record point', async () => {
  const model = (point: string) => `
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
V = joint(lawof(a), lawof(b))
ld = logdensityof(V, ${point})
`;
  const short = ctxFor(model('[0.5]'), 1).ctx;
  await assert.rejects(async () => short.getMeasure('ld'),
    /has 2 components, but the point scored has 1 element/);
  const rec = ctxFor(model('record(a = 0.5, b = 0.7)'), 1).ctx;
  await assert.rejects(async () => rec.getMeasure('ld'),
    /cannot read the point scored as a numeric variate/);
});

// ── free constructor parameters and enclosing laws ──────────────────────────
// The outer law integrates z once. Raw joint constructors retain z as a free
// parameter, so their log density is the conditional sum at the supplied z.
// Compound oracles were independently checked by Gaussian inversion and by
// quadrature of ∫ p(z) p(a|z) p(b|z) dz.
const COMPOUND_ORACLE = -8.748747354129808;
const COMPOUND_EQUAL_SD_ORACLE = -10.903201177191129;
const COMPOUND_AFFINE_LOC_ORACLE = -11.982596349192143;
const DISJOINT_ORACLE = -4.0426427710908985;
const MARGINAL_1D_ORACLE = -2.1138901582154195;
const ULP_BUDGET = 64;
const ulpsClose = (got: number, want: number) =>
  Math.abs(got - want) <= ULP_BUDGET * Number.EPSILON * Math.abs(want);
const normalLogpdf = (x: number, mu: number, sigma: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2;

const CONSTRUCTOR_JOINT = `
z ~ Normal(mu = 0.5, sigma = 2.0)
J = lawof(joint(a = Normal(mu = z, sigma = 0.6), b = Normal(mu = z, sigma = 0.8)))
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`;

test('an enclosing law integrates a shared constructor parameter once', async () => {
  assert.ok(ulpsClose(await scoreOf(CONSTRUCTOR_JOINT, 1), COMPOUND_ORACLE));
  const positional = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
V = lawof(joint(Normal(mu = z, sigma = 0.6), Normal(mu = z, sigma = 0.8)))
ld = logdensityof(V, [2.5, -1.0])
`, 1);
  assert.ok(ulpsClose(positional, COMPOUND_ORACLE));
});

test('raw constructor products remain conditional on an explicit outer input', async () => {
  for (const input of [-1.0, 1.25]) {
    const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
J = joint(a = Normal(mu = z, sigma = 0.6), b = Normal(mu = z, sigma = 0.8))
score = logdensityof(J, record(a = 2.5, b = -1.0))
F = functionof(score, z = z)
ld = F(z = ${input})
`, 1);
    const oracle = normalLogpdf(2.5, input, 0.6) + normalLogpdf(-1, input, 0.8);
    assert.ok(ulpsClose(got, oracle), `z=${input}: got ${got}, oracle ${oracle}`);
  }
});

test('an enclosing constructor law agrees with one law of its fresh draws', async () => {
  const viaRecord = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
a ~ Normal(mu = z, sigma = 0.6)
b ~ Normal(mu = z, sigma = 0.8)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 2.5, b = -1.0))
`, 1);
  assert.ok(ulpsClose(viaRecord, COMPOUND_ORACLE));
  assert.ok(ulpsClose(await scoreOf(CONSTRUCTOR_JOINT, 1), viaRecord));
  const product = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
a ~ Normal(mu = z, sigma = 0.6)
b ~ Normal(mu = z, sigma = 0.8)
J = joint(a = lawof(a), b = lawof(b))
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`, 1);
  assert.ok(Math.abs(product - DISJOINT_ORACLE) < F64_TOL);
});

test('the enclosing constructor law does not depend on sampleCount', async () => {
  assert.equal(await scoreOf(CONSTRUCTOR_JOINT, 1), await scoreOf(CONSTRUCTOR_JOINT, 250));
});

test('an outer law over disjoint constructor parameters is the product of marginals', async () => {
  const got = await scoreOf(`
z1 ~ Normal(mu = 0.5, sigma = 2.0)
z2 ~ Normal(mu = 0.5, sigma = 2.0)
J = lawof(joint(a = Normal(mu = z1, sigma = 0.6), b = Normal(mu = z2, sigma = 0.8)))
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`, 1);
  assert.ok(Math.abs(got - DISJOINT_ORACLE) < F64_TOL);
});

test('an outer law over one constructor integrates its one-dimensional marginal', async () => {
  for (const [measure, point] of [
    ['lawof(joint(a = Normal(mu = z, sigma = 0.6)))', 'record(a = 2.5)'],
    ['lawof(Normal(mu = z, sigma = 0.6))', '2.5'],
  ]) {
    const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
J = ${measure}
ld = logdensityof(J, ${point})
`, 1);
    assert.ok(Math.abs(got - MARGINAL_1D_ORACLE) < F64_TOL);
  }
});

test('an enclosing law gives reused and inline constructors the same fresh noise', async () => {
  for (const product of ['joint(a = q, b = q)',
    'joint(a = Normal(z, 0.6), b = Normal(z, 0.6))']) {
    const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
q = Normal(mu = z, sigma = 0.6)
J = lawof(${product})
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`, 1);
    assert.ok(ulpsClose(got, COMPOUND_EQUAL_SD_ORACLE));
  }
});

test('the enclosing law composes an affine constructor location', async () => {
  const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
t = 3.0 * z
q = Normal(mu = t, sigma = 0.6)
J = lawof(joint(a = q, b = q))
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`, 1);
  assert.ok(ulpsClose(got, COMPOUND_AFFINE_LOC_ORACLE));
});

test('nested raw constructor products retain their free parameter', async () => {
  const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
q = Normal(mu = z, sigma = 0.6)
u = joint(a = q, b = q)
J = joint(u = u, c = q)
score = logdensityof(J, record(u = record(a = 2.5, b = -1.0), c = 0.3))
F = functionof(score, z = z)
ld = F(z = 0.5)
`, 1);
  const oracle = [2.5, -1, 0.3].reduce((sum, x) => sum + normalLogpdf(x, 0.5, 0.6), 0);
  assert.ok(ulpsClose(got, oracle));
});

test('repeated and distinct standard-normal law values have independent product densities', async () => {
  for (const fields of ['a = lawof(x), b = lawof(x)', 'a = lawof(y), b = lawof(x)',
    'a = L, b = L', 'a = self.L, b = L']) {
    for (const b of [0.0, 1.0]) {
      const got = await scoreOf(`
x ~ Normal(0.0, 1.0)
y ~ Normal(0.0, 1.0)
L = lawof(x)
J = joint(${fields})
ld = logdensityof(J, record(a = 0.0, b = ${b}))
`, 1);
      const oracle = -Math.log(2 * Math.PI) - b * b / 2;
      assert.ok(Math.abs(got - oracle) < F64_TOL, `${fields} at (0,${b}): ${got}`);
    }
  }
});

test('a repeated hierarchical law integrates separate copies of its ancestor', async () => {
  const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
y ~ Normal(mu = z, sigma = 0.6)
L = lawof(y)
J = joint(a = L, b = L)
ld = logdensityof(J, record(a = 0.5, b = 0.9))
`, 1);
  const sigma = Math.sqrt(4.36);
  assert.ok(Math.abs(got - (normalLogpdf(0.5, 0.5, sigma)
    + normalLogpdf(0.9, 0.5, sigma))) < F64_TOL);
});

test('a raw constructor and a non-Gaussian sibling score conditionally', async () => {
  const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
J = joint(a = Normal(mu = z, sigma = 0.6), b = Exponential(rate = 1.0))
score = logdensityof(J, record(a = 2.5, b = 1.0))
F = functionof(score, z = z)
ld = F(z = 0.5)
`, 1);
  assert.ok(Math.abs(got - (normalLogpdf(2.5, 0.5, 0.6) - 1)) < F64_TOL);
});

test('iid over a reified hierarchical law scores the product of marginals', async () => {
  const got = await scoreOf(`
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
L = iid(lawof(a), 2)
ld = logdensityof(L, [0.5, 0.7])
`, 1);
  assert.ok(Math.abs(got - PRODUCT_ORACLE) < F64_TOL,
    `got ${got}, independent marginal product ${PRODUCT_ORACLE}`);
});
