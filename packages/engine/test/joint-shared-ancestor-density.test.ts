'use strict';

// Density of a joint law over components that SHARE a stochastic ancestor.
// Spec anchors, quoted verbatim from flatppl-design 52df5de: §06 "Joint
// composition" → "Equivalent record law" and "Singular joints", §06 "Density of
// composed measures", §04 "Identity law" and "Trace of the reified law".
//
// §06 "Equivalent record law": "`joint(a = lawof(a), b = lawof(b))` is
// equivalent to `lawof(record(a = a, b = b))`; the positional form is the
// corresponding `cat` law". §06 "Density of composed measures": "A `joint` with
// shared ancestry reduces as its equivalent record law; a singular joint has no
// density and the query is refused."
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

test('shared-ancestor record law scores the analytic MvNormal marginal (N=1)', async () => {
  const got = await scoreOf(RECORD_LAW, 1);
  assert.ok(Math.abs(got - MVN_ORACLE) < F64_TOL,
    `got ${got}, Distributions.jl MvNormal([0,0],[[2,1],[1,2]]) logpdf(0.5,0.7) = ${MVN_ORACLE}`);
});

test('joint(a = lawof(a), b = lawof(b)) scores the SAME analytic marginal '
  + '(§06 "Equivalent record law")', async () => {
  const viaJoint = await scoreOf(JOINT_OF_LAWS, 1);
  const viaRecord = await scoreOf(RECORD_LAW, 1);
  assert.ok(Math.abs(viaJoint - MVN_ORACLE) < F64_TOL,
    `got ${viaJoint}, oracle ${MVN_ORACLE}`);
  assert.equal(viaJoint, viaRecord,
    'the two spellings are one measure and must score bit-identically');
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

test('the POSITIONAL joint spelling scores the same marginal over a vector variate',
  async () => {
    const got = await scoreOf(`
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
b ~ Normal(mu = z, sigma = 1.0)
V = joint(lawof(a), lawof(b))
ld = logdensityof(V, [0.5, 0.7])
`, 1);
    assert.ok(Math.abs(got - MVN_ORACLE) < F64_TOL, `got ${got}, oracle ${MVN_ORACLE}`);
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
S = joint(a = lawof(y), b = lawof(y))
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`;

test('joint(a = lawof(y), b = lawof(y)) refuses the density query', async () => {
  const { ctx } = ctxFor(SINGULAR_SAME_DRAW, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /share the ancestor 'y'.*lower-dimensional subset/s);
});

test('the record spelling of the same-draw joint refuses identically', async () => {
  const { ctx } = ctxFor(`
y ~ Normal(mu = 0.0, sigma = 1.0)
R = lawof(record(a = y, b = y))
ld = logdensityof(R, record(a = 0.5, b = 0.9))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /share the ancestor 'y'.*no density w\.r\.t\. the product reference measure/s);
});

test('a deterministic transform of another component refuses the density query', async () => {
  const { ctx } = ctxFor(`
y ~ Normal(mu = 0.0, sigma = 1.0)
w = exp(y)
T = joint(a = lawof(y), b = lawof(w))
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

test('every spelling of the chain joint scores the SAME marginal — a component '
  + 'is identified by its DRAW, not by its label', async () => {
  // The record spelling exposes the field under the draw's own name `a`; the
  // other three do not, and before draw-identity keying the recogniser
  // integrated `a` out as a latent AND redrew it as a component, silently
  // scoring MvNormal([0,0],[[2,1],[1,3]]) — a WRONG number, not a refusal.
  const spellings = [
    ['record', 'L = lawof(record(a = a, b = b))', 'record(a = 0.5, b = 0.7)'],
    ['joint of laws', 'L = joint(a = lawof(a), b = lawof(b))', 'record(a = 0.5, b = 0.7)'],
    ['relabelled joint', 'L = joint(p = lawof(a), q = lawof(b))', 'record(p = 0.5, q = 0.7)'],
    ['positional joint', 'L = joint(lawof(a), lawof(b))', '[0.5, 0.7]'],
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

// ── constructor components with a stochastic parameter ─────────────────────
//
// §06 "Joint composition", quoted verbatim from flatppl-design 9e35262: "A
// component contributes a fresh coordinate; a stochastic node shared between
// component traces (through a reified component — `lawof`, `kernelof` — or a
// stochastic constructor parameter) remains a single node of the composed trace.
// Components that share no stochastic node are independent, and their `joint` is
// the product measure".
//
// A distribution CONSTRUCTOR whose parameters reach a draw is the second of the
// two routes that sentence names, so `z ~ Normal(mu0, s0); joint(a = Normal(mu =
// z, sigma = sa), b = Normal(mu = z, sigma = sb))` is the COMPOUND (correlated)
// law: each coordinate is fresh, `z` is one node, giving mean (mu0, mu0) and
// Sigma = s0^2 * 11' + diag(sa^2, sb^2). That is the same law §06 "Equivalent
// record law" gives `lawof(record(a = a, b = b))` over fresh draws, so the two
// spellings are pinned to ONE oracle below.
//
// Oracles are INDEPENDENT of this engine and of flatppl-rust (which emits
// symbolic lowerings and evaluates no density at all, so it is not an oracle
// here). Each was computed at 60 decimal digits by two routes that share no
// algebra — the closed-form Gaussian with an explicit 2x2 inverse and
// determinant, and numerical quadrature of the compound integral
// ∫ p(z) p(a|z) p(b|z) dz, which never forms a covariance matrix. For every value
// below the two routes agree to 60 significant digits.
//
// The reference is computed from the parameters as EXACT DECIMALS, which is what
// the source literals denote; feeding the reference the float64 of `0.6` instead
// shifts it by about 1 ulp — right at the level these tests assert, so it is not
// a detail that can be skipped.
//   mu0 = 0.5, s0 = 2, point (2.5, -1.0)
//     (sa, sb) = (0.6, 0.8):  -8.74874735412980761937447869029
//     sa = sb = 0.6:         -10.9032011771911282956792920692
//   mu0 = 1.5, s0 = 6 (an affine location t = 3z), sa = sb = 0.6, same point
//                            -11.9825963491921424570272318537
// The disjoint-latent and marginal controls are products of 1-D Normals from the
// same computation, and the engine matches those bit-exactly.

const COMPOUND_ORACLE = -8.748747354129808;          // nearest f64 of the above
const COMPOUND_EQUAL_SD_ORACLE = -10.903201177191129;
const COMPOUND_AFFINE_LOC_ORACLE = -11.982596349192143;
const DISJOINT_ORACLE = -4.0426427710908985;
const MARGINAL_1D_ORACLE = -2.1138901582154195;

// WHICH TOLERANCE TO USE, and why there are two.
//
// Most tests in this file compare against the absolute `F64_TOL = 1e-14`, which
// suits the F1-era blocks: [[2,1],[1,2]] and friends are well conditioned, so the
// engine lands within ~2 ulp of the exact value.
//
// The COMPOUND blocks are not. Sigma = s0^2 * 11' + diag(sigma_i^2) is dominated
// by the rank-1 shared-latent term (4.36 against 4 off-diagonal; 36.36 against 36
// once the location is scaled), so the engine's Cholesky cancels far more and
// lands 5-17 ulp out. Worse, the error grows with s0^2/sigma^2 — the affine-
// location sibling below sits at 16 EPS on a CORRECT answer — so an absolute
// budget would have to be re-tuned per shape. A compound-law test therefore uses
// `ulpsClose` (a RELATIVE budget), and every other test keeps F64_TOL.
//
// `ulpsClose` is still an exactness claim, not a loosened statistical tolerance.
// The budget is 64 EPS ~ 1.4e-14 relative, and the failure mode it must catch —
// dropping the shared node, so the law becomes the product of the marginals —
// sits 2.4e15 EPS away (6.45 nats on the affine sibling). There are thirteen
// orders of magnitude of daylight between the budget and the nearest wrong
// answer, and the worst correct value observed uses a quarter of the budget.
const ULP_BUDGET = 64;
const ulpsClose = (got: number, want: number) =>
  Math.abs(got - want) <= ULP_BUDGET * Number.EPSILON * Math.abs(want);

const CONSTRUCTOR_JOINT = `
z ~ Normal(mu = 0.5, sigma = 2.0)
J = joint(a = Normal(mu = z, sigma = 0.6), b = Normal(mu = z, sigma = 0.8))
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`;

test('a keyword joint of CONSTRUCTORS sharing a latent scores the compound law',
  async () => {
    const got = await scoreOf(CONSTRUCTOR_JOINT, 1);
    assert.ok(ulpsClose(got, COMPOUND_ORACLE),
      `got ${got}, compound-law oracle ${COMPOUND_ORACLE} (40-digit closed form and `
      + 'quadrature agree). The product of the two marginals would be '
      + `${DISJOINT_ORACLE} — a component sharing no node is a DIFFERENT law`);
  });

test('the POSITIONAL constructor-joint spelling scores the same compound law',
  async () => {
    const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
V = joint(Normal(mu = z, sigma = 0.6), Normal(mu = z, sigma = 0.8))
ld = logdensityof(V, [2.5, -1.0])
`, 1);
    assert.ok(ulpsClose(got, COMPOUND_ORACLE), `got ${got}, oracle ${COMPOUND_ORACLE}`);
  });

test('the constructor joint and its equivalent record law over fresh draws are '
  + 'ONE measure (§06 "Equivalent record law")', async () => {
  // The constructor spelling shares `z` through a stochastic PARAMETER, the
  // record spelling through REIFIED components. §06 names both routes in one
  // sentence, so the two must not merely agree to a tolerance — they are the
  // same measure and must score bit-identically.
  const viaConstructors = await scoreOf(CONSTRUCTOR_JOINT, 1);
  const viaRecordLaw = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
a ~ Normal(mu = z, sigma = 0.6)
b ~ Normal(mu = z, sigma = 0.8)
L = lawof(record(a = a, b = b))
ld = logdensityof(L, record(a = 2.5, b = -1.0))
`, 1);
  const viaJointOfLaws = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
a ~ Normal(mu = z, sigma = 0.6)
b ~ Normal(mu = z, sigma = 0.8)
J = joint(a = lawof(a), b = lawof(b))
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`, 1);
  assert.equal(viaConstructors, viaRecordLaw);
  assert.equal(viaConstructors, viaJointOfLaws);
});

test('the constructor joint does not depend on sampleCount (no MC estimate)',
  async () => {
    assert.equal(await scoreOf(CONSTRUCTOR_JOINT, 1),
      await scoreOf(CONSTRUCTOR_JOINT, 250));
  });

test('constructor components over DISJOINT latents are the product of their '
  + 'marginals', async () => {
  // §06: "Components that share no stochastic node are independent, and their
  // `joint` is the product measure". Each component still marginalises its OWN
  // latent, so this is the product of two 1-D compound marginals — not the
  // ancestor-free product, and not the correlated law.
  const got = await scoreOf(`
z1 ~ Normal(mu = 0.5, sigma = 2.0)
z2 ~ Normal(mu = 0.5, sigma = 2.0)
J = joint(a = Normal(mu = z1, sigma = 0.6), b = Normal(mu = z2, sigma = 0.8))
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`, 1);
  assert.ok(Math.abs(got - DISJOINT_ORACLE) < F64_TOL,
    `got ${got}, product of the two 1-D marginals ${DISJOINT_ORACLE}`);
});

test('a SOLE constructor component with a latent is that component\'s 1-D '
  + 'marginal', async () => {
  // The one-component joint must keep the direct path rather than becoming a
  // one-field record, and a bare stochastic-parameter measure is the same law.
  // (The `relabel(Normal(mu = z, …), ["a"])` spelling of this is NOT covered: it
  // refuses on the separately filed S5 relabel typing gap, at origin/main too.)
  for (const decl of ['J = joint(a = Normal(mu = z, sigma = 0.6))']) {
    const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
${decl}
ld = logdensityof(J, record(a = 2.5))
`, 1);
    assert.ok(Math.abs(got - MARGINAL_1D_ORACLE) < F64_TOL,
      `${decl}: got ${got}, Normal(0.5, √(4 + 0.36)) at 2.5 = ${MARGINAL_1D_ORACLE}`);
  }
  const bare = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
J = Normal(mu = z, sigma = 0.6)
ld = logdensityof(J, 2.5)
`, 1);
  assert.ok(Math.abs(bare - MARGINAL_1D_ORACLE) < F64_TOL,
    `bare stochastic-parameter measure: got ${bare}, oracle ${MARGINAL_1D_ORACLE}`);
});

// A constructor NAMED ONCE and used as two components. §06 gives each component
// a fresh coordinate and keeps the shared parameter node single, so this is the
// compound law with sa = sb — NOT the singular diagonal. Before this was fixed
// the two components resolved to one binding name, which the singularity gate
// read as the same draw and refused; naming a constructor must not change the
// measure it denotes.

const REUSED_CONSTRUCTOR = `
z ~ Normal(mu = 0.5, sigma = 2.0)
q = Normal(mu = z, sigma = 0.6)
J = joint(a = q, b = q)
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`;

test('a constructor with a latent used as TWO components is the compound law, '
  + 'not a singular joint', async () => {
  const got = await scoreOf(REUSED_CONSTRUCTOR, 1);
  assert.ok(ulpsClose(got, COMPOUND_EQUAL_SD_ORACLE),
    `got ${got}, compound-law oracle ${COMPOUND_EQUAL_SD_ORACLE}`);
});

test('writing the reused constructor out twice denotes the same measure',
  async () => {
  const inline = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
J = joint(a = Normal(mu = z, sigma = 0.6), b = Normal(mu = z, sigma = 0.6))
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`, 1);
  assert.equal(await scoreOf(REUSED_CONSTRUCTOR, 1), inline);
});

test('the reused constructor still composes through an AFFINE location',
  async () => {
    // `t = 3z` scales the shared node, so Sigma = 36 * 11' + diag(0.36, 0.36) and
    // the mean is (1.5, 1.5). This is the shape that fixes the tolerance budget:
    // s0^2/sigma^2 = 100 here against 11 above, and the engine's Cholesky lands
    // 16 EPS out on this CORRECT value — which is why ULP_BUDGET is not 16.
    const got = await scoreOf(`
z ~ Normal(mu = 0.5, sigma = 2.0)
t = 3.0 * z
q = Normal(mu = t, sigma = 0.6)
J = joint(a = q, b = q)
ld = logdensityof(J, record(a = 2.5, b = -1.0))
`, 1);
    assert.ok(ulpsClose(got, COMPOUND_AFFINE_LOC_ORACLE),
      `got ${got}, oracle ${COMPOUND_AFFINE_LOC_ORACLE}. Dropping the shared node `
      + 'would give -5.531043805465598, 6.45 nats away');
  });

// KNOWN SAMPLING DEFECT, filed in flatppl-dev/TODO-flatppl-js.md and
// measure-algebra-audit.md: NESTING a constructor joint inside another joint that
// also names the same constructor. `joint(u = joint(a = q, b = q), c = q)` should
// be three fresh coordinates over one shared `z`, so all three pairwise
// correlations are s0²/(s0²+σ²). The inner pair is correct, but `c` comes back
// BIT-IDENTICAL to `u.a` (correlation exactly 1) because the outer joint's `q` is
// not a duplicate BY NAME of `u`, so it takes the first-occurrence branch in
// `materialiser._materialiseFactorsIndependent` and reuses the cached batch that
// `u.a` already drew. That is a singular pair where §06 wants a fresh coordinate.
//
// Pre-existing: identical at clean 0ab097d, before this wave. It is a SAMPLING
// defect only — the density path refuses the shape (its component `u` is a
// `joint`, not a Normal), so no wrong number is ever scored. This test pins the
// refusal, which is what currently keeps the defect unreachable from a density.
// The correlation itself is deliberately NOT asserted here: the only value
// available to assert is the wrong one, and pinning a wrong value as expected is
// how it would become permanent. The correlation guard lives in
// sampling-density-agreement.test.ts, tagged WILL-FLIP.

test('a nested constructor joint refuses on the density side (which is what '
  + 'keeps the known nested sampling defect unscorable)', async () => {
  const { ctx } = ctxFor(`
z ~ Normal(mu = 0.5, sigma = 2.0)
q = Normal(mu = z, sigma = 0.6)
u = joint(a = q, b = q)
J = joint(u = u, c = q)
ld = logdensityof(J, record(u = record(a = 2.5, b = -1.0), c = 0.3))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /component 'u' is a 'joint', not a Normal/,
    'if this starts scoring, the nested SAMPLING defect (c bit-identical to u.a) '
    + 'becomes reachable from a density and must be fixed first');
});

test('a named REIFIED law used as two components still refuses as singular',
  async () => {
    // The counterpart the fix must not break: `lawof(y)` names a DRAW, so two
    // components of it are the same coordinate and the joint is singular (§06
    // "Singular joints"). The discriminator is variate-vs-measure, not whether
    // the binding was given a name.
    for (const model of [`
y ~ Normal(mu = 0.0, sigma = 1.0)
Ly = lawof(y)
S = joint(a = Ly, b = Ly)
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`, `
z ~ Normal(mu = 0.5, sigma = 2.0)
y ~ Normal(mu = z, sigma = 0.6)
Ly = lawof(y)
S = joint(a = Ly, b = Ly)
ld = logdensityof(S, record(a = 0.5, b = 0.9))
`]) {
      const { ctx } = ctxFor(model, 1);
      await assert.rejects(async () => ctx.getMeasure('ld'),
        /share the ancestor 'y'.*lower-dimensional subset/s);
    }
  });

// KNOWN GAP (filed in flatppl-dev/TODO-flatppl-js.md): a latent-carrying
// constructor beside an ancestor-free NON-Gaussian sibling. §06 makes the sibling
// independent, so the exact answer is the product of the closed-form marginal and
// the sibling's own density, but the recogniser classifies every component in one
// linear-Gaussian block and refuses on the first non-Normal one. It refuses
// rather than answering wrongly, so this is a capability gap; the test pins the
// refusal so implementing the factorisation flips a red test instead of drifting.
test('an ancestor-free NON-Gaussian sibling refuses rather than factorising '
  + '(known gap)', async () => {
  const { ctx } = ctxFor(`
z ~ Normal(mu = 0.5, sigma = 2.0)
J = joint(a = Normal(mu = z, sigma = 0.6), b = Exponential(rate = 1.0))
ld = logdensityof(J, record(a = 2.5, b = 1.0))
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /component 'b' is a 'Exponential', not a Normal/,
    'when this starts scoring, the exact value is -3.1138901582154195 — the 1-D '
    + 'marginal -2.1138901582154195 plus Exponential(1) at 1.0, which is -1');
});

// ── iid is independent by construction (§06 "iid") ─────────────────────────

test('iid over a reified law is unchanged: the copies redraw the ancestor, so '
  + 'the shared-ancestor marginal does not apply', async () => {
  // §06 iid: "each of the N copies redraws the reified sub-DAG afresh,
  // including its stochastic ancestors — `iid` never shares ancestors between
  // copies". The engine has no closed form for that product of marginals, so
  // the density query refuses rather than reusing ONE ancestor draw for every
  // copy (which would silently score the shared-ancestor joint instead).
  const { ctx } = ctxFor(`
z ~ Normal(mu = 0.0, sigma = 1.0)
a ~ Normal(mu = z, sigma = 1.0)
L = iid(lawof(a), 2)
ld = logdensityof(L, [0.5, 0.7])
`, 1);
  await assert.rejects(async () => ctx.getMeasure('ld'),
    /no exact answer for it here/);
});
