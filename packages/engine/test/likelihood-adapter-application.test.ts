'use strict';

// A likelihood adapter `a -> K(theta = e(pars))` applies one kernel. It is
// scored against K's own graph with K's boundary replaced by e(pars), so lift
// copies none of K's closure. Each oracle below is closed form. The arguments
// rename and transform the input, so a dropped substitution changes the number.

const test = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const normal = (x: number, mu: number, sigma: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - (x - mu) ** 2 / (2 * sigma * sigma);
const close = (got: number, want: number) =>
  assert.ok(Math.abs(got - want) < 1e-12, `${got} vs ${want}`);
const copies = (built: any) => [...built.bindings.values()].filter((b: any) => b.closureOf);
async function scalar(ctx: any, name: string) {
  const m = await ctx.getMeasure(name);
  return (m.samples || m.value.data)[0];
}

const KERNEL = `
theta ~ Normal(0, 1)
y ~ Normal(theta, 0.5)
K = kernelof(record(y = y), theta = theta)
a ~ Normal(0, 2)
prior = lawof(record(a = a))
`;

test('a single kernel application is scored without copying the kernel closure', async () => {
  const { ctx, built } = makeMatCtx(`
theta ~ Uniform(unitinterval)
counts(n, theta) = iid(Binomial(n, theta), 2)
sizes = indicesof(zeros(3))
w = ones(3)
y ~ normalize(ksuperpose(counts, w)(sizes, theta))
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = [1, 2]))
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
s = logdensityof(L, (theta = 0.3))
`, { sampleCount: 4 });
  assert.deepEqual(copies(built), []);
  assert.equal([...built.bindings.values()].filter((b: any) => b.ir?.op === 'Binomial').length, 3);
  const theta = 0.3;
  const expected = Math.log((2 * theta ** 3 * (1 - theta) + 9 * theta ** 3 * (1 - theta) ** 3) / 3);
  close(await scalar(ctx, 's'), expected);
  close((await buildLogPi(ctx, ctx.derivations.posterior)).logPi({ theta }), expected);
});

test('renamed, transformed and positional arguments bind the applied boundary', async () => {
  for (const kernel of ['a -> K(theta = a / 2)', 'a -> K(a / 2)']) {
    const { ctx, built } = makeMatCtx(`${KERNEL}
L = likelihoodof(${kernel}, record(y = 1.0))
posterior = bayesupdate(L, prior)
s = logdensityof(L, (a = 0.6))
p = logdensityof(posterior, (a = 0.6))
`, { sampleCount: 4 });
    assert.deepEqual(copies(built), [], kernel);
    const lik = normal(1.0, 0.3, 0.5);
    close(await scalar(ctx, 's'), lik);
    close(await scalar(ctx, 'p'), lik + normal(0.6, 0, 2));
    close((await buildLogPi(ctx, ctx.derivations.posterior)).logPi({ a: 0.6 }), lik + normal(0.6, 0, 2));
  }
});

test('importance weights and joint likelihoods carry the applied boundary', async () => {
  const { ctx } = makeMatCtx(`${KERNEL}
L1 = likelihoodof(a -> K(theta = a / 2), record(y = 1.0))
L2 = likelihoodof(a -> K(theta = 3 * a), record(y = -0.4))
J = joint_likelihood(L1, L2)
posterior = bayesupdate(L1, prior)
s = logdensityof(J, (a = 0.6))
`, { sampleCount: 64 });
  close(await scalar(ctx, 's'), normal(1.0, 0.3, 0.5) + normal(-0.4, 1.8, 0.5));
  const post = await ctx.getMeasure('posterior');
  const atoms = post.fields.a.samples;
  const lw = post.logWeights;
  for (let i = 1; i < atoms.length; i++) {
    const want = normal(1.0, atoms[i] / 2, 0.5) - normal(1.0, atoms[0] / 2, 0.5);
    assert.ok(Math.abs((lw[i] - lw[0]) - want) < 1e-9, `atom ${i}`);
  }
});

test('a normalizer inside the applied kernel takes its mass at the argument', async () => {
  // Constant and transformed arguments both bind theta = 2.5. The module
  // theta is a Gamma draw, so a mass taken at the module node would differ.
  const { ctx, built } = makeMatCtx(`
theta ~ Gamma(2, 1)
y ~ normalize(truncate(Normal(theta, 1), interval(2, 4)))
K = kernelof(record(y = y), theta = theta)
a ~ Gamma(2, 1)
prior = lawof(record(a = a))
L1 = likelihoodof(a -> K(theta = 2.5), record(y = 3))
L2 = likelihoodof(a -> K(theta = a / 2), record(y = 3))
posterior = bayesupdate(L2, prior)
s1 = logdensityof(L1, (a = 1.0))
s2 = logdensityof(L2, (a = 5.0))
`, { sampleCount: 4 });
  assert.deepEqual(copies(built), []);
  // log φ(0.5) − log(Φ(1.5) − Φ(−0.5)), from the error function.
  const expected = -0.573383167788773;
  close(await scalar(ctx, 's1'), expected);
  close(await scalar(ctx, 's2'), expected);
  close((await buildLogPi(ctx, ctx.derivations.posterior)).likOf({ a: 5.0 }), expected);
});

test('a fallback normalizer mass reads the substituted inner measure, not its module name', async () => {
  // No current shape reaches this fallback with a boundary-dependent inner,
  // so drive it directly. The named binding has the module weight 0.5. The
  // substituted inner has weight 0.25.
  const { massFromMeasures } = require('../normalize-mass.ts');
  const { ctx } = makeMatCtx('w = weighted(0.5, Normal(0, 1))\n', { sampleCount: 4 });
  const lit = (value: number) => ({ kind: 'lit', value });
  const ir = { kind: 'call', op: 'weighted', args: [lit(0.25),
    { kind: 'call', op: 'Normal', args: [lit(0), lit(1)] }] };
  const [applied, named] = await massFromMeasures(
    [{ massFrom: { ref: 'w', ir } }, { massFrom: { ref: 'w' } }], ctx);
  close(applied.logTotalmass, Math.log(0.25));
  close(named.logTotalmass, Math.log(0.5));
});

test('a fixed normalizer inside the applied kernel takes the materialised mass', async () => {
  // No symbolic mass exists for a truncated Poisson, so buildLogPi materialises
  // it. The lower bound sits below 0 so the support is {0, …, 5}.
  const { ctx, built } = makeMatCtx(`
theta ~ Normal(0, 1)
y ~ normalize(truncate(Poisson(3.0), interval(-1, 5)))
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = 2))
posterior = bayesupdate(L, lawof(record(theta = theta)))
`, { sampleCount: 4 });
  assert.deepEqual(copies(built), []);
  const logPmf = (k: number) => k * Math.log(3) - 3 - Math.log([1, 1, 2, 6, 24, 120][k]);
  const logZ = Math.log([0, 1, 2, 3, 4, 5].reduce((s, k) => s + Math.exp(logPmf(k)), 0));
  close((await buildLogPi(ctx, ctx.derivations.posterior)).likOf({ theta: 0.2 }), logPmf(2) - logZ);
});

test('the scored tree refuses a boundary that substitution cannot reach', () => {
  // Lift never takes this path for a closure with a latent draw. Build the term
  // by hand to show the assertion catches a criterion that admits one.
  const { ctx, built } = makeMatCtx(`
theta ~ Normal(0, 1)
mu ~ Normal(theta, 1)
y ~ Normal(mu, 1)
K = kernelof(record(y = y), theta = theta)
`, { sampleCount: 4 });
  const K = built.bindings.get('K').ir;
  const term = { kind: 'likelihood_density', bodyName: null, bodyIR: K.body,
    obsIR: { kind: 'lit', value: 0.5 }, params: ['_pars_'], paramKwargs: ['pars'], orderedInputs: true,
    boundaryArgs: { theta: { kind: 'call', op: 'get_field',
      args: [{ kind: 'ref', ns: '%local', name: '_pars_' }, { kind: 'lit', value: 'theta' }] } } };
  assert.throws(() => require('../clm.ts').lowerMeasure(term.bodyIR, ctx, { derivation: term }),
    /an applied kernel's boundary still reaches its scored body through mu/);
});

test('an applied callable whose body is not its variates keeps the copy path', async () => {
  const { ctx, built } = makeMatCtx(`${KERNEL}
N = functionof(Normal(theta, 0.5), theta = theta)
L = likelihoodof(a -> N(theta = a / 2), 1.0)
s = logdensityof(L, (a = 0.6))
`, { sampleCount: 4 });
  assert.equal(built.derivations.s.boundaryArgs, undefined);
  close(await scalar(ctx, 's'), normal(1.0, 0.3, 0.5));
});

test('an ill-formed adapter keeps the copy path and its diagnostic', () => {
  const { processSource, orchestrator } = require('../index.ts');
  const cases: Array<[string, string]> = [
    [`${KERNEL}L = likelihoodof(a -> K(theta = a, scale = 2), record(y = 1.0))\n`,
      'call to "K" has unknown argument "scale"'],
    [`${KERNEL}L = likelihoodof(a -> K(a, 2), record(y = 1.0))\n`,
      'call to "K" does not accept 2 positional arguments'],
    // Lift leaves a recursive call in place, inside K's closure.
    [`f(x) = f(x)
theta ~ Normal(0, 1)
y ~ Normal(f(theta), 0.5)
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(a -> K(theta = a), record(y = 1.0))
`, 'cyclic binding dependency at "f"'],
  ];
  for (const [source, message] of cases) {
    const proc = processSource(source);
    assert.ok(proc.diagnostics.some((d: any) => d.message === message), message);
    const built = orchestrator.buildDerivations(proc.linkedBindings, { moduleRegistry: proc.linkedModuleRegistry });
    assert.ok(copies(built).length > 0, message);
  }
});

test('a body with two applications keeps the copy path', async () => {
  const { ctx, built } = makeMatCtx(`
theta ~ Normal(0, 1)
y ~ Normal(theta, 0.5)
K = kernelof(y, theta = theta)
L = likelihoodof((a, b) -> joint(u = K(theta = a), v = K(theta = b)), record(u = 0.2, v = -0.1))
s = logdensityof(L, (a = 0.4, b = -0.3))
`, { sampleCount: 4 });
  assert.ok(copies(built).length >= 2);
  close(await scalar(ctx, 's'), normal(0.2, 0.4, 0.5) + normal(-0.1, -0.3, 0.5));
});

test('a kernel closure with a latent draw keeps the copy path', () => {
  const { built } = makeMatCtx(`
theta ~ Uniform(interval(0.1, 0.9))
mu ~ Normal(theta, 1)
y ~ Normal(mu, 1)
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(a -> K(theta = a), record(y = 1))
`, { sampleCount: 4 });
  // The latent `mu` would stay a marginalized ref that substitution cannot reach.
  assert.ok(copies(built).some((b: any) => b.closureOf === 'mu'));
});
