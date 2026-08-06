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
// error" — and the owner's 2026-08-05 decision applies it to this construct.
// Under either there is no Monte-Carlo branch to test: a shape this engine
// cannot answer by ONE of those two exact devices must REFUSE.
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
// — density(y) = Σ_k P(z = k) · density(y | z = k) — so it is not the
// Monte-Carlo estimate the 2026-08-05 decision rules out. The support cap
// mirrors the determiniser's (`flatppl-rust/crates/determinizer/src/marginal.rs`,
// MAX_ATOMS = 256), which is a CAP VALUE reference only, not a semantics oracle.

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
