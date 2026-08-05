'use strict';

// Density of a joint law over components that SHARE a stochastic ancestor
// (spec §06 "Joint composition" → "Reified components share their ancestry",
// §06 "Density of composed measures", §06 "Singular joints", §04 "Identity
// law" / "Trace of the reified law").
//
// §06: "`joint(a = lawof(a), b = lawof(b))` is equivalent to `lawof(record(a =
// a, b = b))`: the shared ancestor is traced once and the dependence is
// retained", and "A `joint` with shared ancestry reduces as its equivalent
// record law; a singular joint has no density and the query is refused."
// §06 allows an engine exactly two ways to evaluate the resulting marginal —
// "in closed form, or by enumeration of a discrete latent, and otherwise
// reports a static error" — so there is no Monte-Carlo branch to test: a shape
// this engine cannot close analytically must REFUSE.
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
  + '(§06 "Reified components share their ancestry")', async () => {
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
      /cannot evaluate in closed form.*'z' is a 'Gamma'/s,
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
    /cannot evaluate in closed form/);
});
