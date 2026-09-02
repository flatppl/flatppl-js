'use strict';

// The spec's own "Posterior construction" example — a `joint` prior over a free
// `elementof` parameter, a likelihood of a point-free `functionof(model)` — used
// to be cascade-pruned (the H1 boundary-conflation scar zone,
// flatppl-dev/measure-algebra-audit.md): the engine materialised only
// bayesupdate(L, lawof(draws)) and the disintegration idiom, so this posterior
// got no derivation and the viewer reported it unplottable.
//
// The cause was upstream of bayesupdate. §04's implicit-boundary rule — a
// no-kwargs `functionof` / `kernelof` promotes its reachable elementof leaves to
// inputs — was applied only to a NAMED reification binding. The reification here
// is INLINE inside `likelihoodof(...)`, and inlines are lifted to their own
// anonymous binding only after that pass ran, so this one kept `params: []`: a
// parameterless callable, with no boundary for the prior's atoms to feed.
//
// It now classifies, and this test pins BOTH halves against independent oracles:
// the density at a point against the closed form, and the posterior measure's
// importance-weighted moments against the conjugate Normal-Normal posterior.
// The loud diagnostic for a genuinely unsupported prior shape is still asserted
// below, on a shape that remains unsupported.

const test = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

// Verbatim from flatppl-design/docs/06-measure-algebra.md "Posterior construction".
const SPEC_POSTERIOR = `
mu = elementof(reals)
model = Normal(mu = mu, sigma = 1.0)
obs = 2.5
L = likelihoodof(functionof(model), obs)
prior = joint(mu = Normal(mu = 0, sigma = 2.0))
posterior = bayesupdate(L, prior)
`;

test('the spec-canonical posterior classifies and emits no diagnostic', () => {
  const ds = orchestrator.buildDerivations(processSource(SPEC_POSTERIOR).bindings);
  assert.equal(ds.derivations.posterior.kind, 'bayesupdate',
    'posterior classifies (was cascade-pruned)');
  const diags = (ds.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(diags.map((d: any) => d.message), [],
    'no error diagnostics for the spec\'s own posterior construction');
  // The implicit boundary reached the LIFTED inline reification, which is what
  // gives the prior's atoms somewhere to feed.
  const kernel = [...ds.bindings.values()].find((b: any) =>
    b.ir && b.ir.op === 'functionof' && Array.isArray(b.ir.params)
    && b.ir.params.length === 1 && b.ir.params[0] === 'mu');
  assert.ok(kernel, 'the inline functionof carries `mu` as an explicit boundary');
});

test('the spec-canonical posterior scores against the closed form', async () => {
  const { ctx } = makeMatCtx(SPEC_POSTERIOR
    + 'lp = logdensityof(posterior, record(mu = 1.0))\n', { sampleCount: 8 });
  const m = await ctx.getMeasure('lp');
  const got = m.value ? m.value.data[0] : m.samples[0];
  // log N(2.5 | 1, 1) + log N(1 | 0, 2), written out.
  const L2P = Math.log(2 * Math.PI);
  const want = (-0.5 * L2P - 0.5 * (2.5 - 1.0) ** 2)
             + (-0.5 * L2P - Math.log(2) - 0.5 * (1.0 / 2) ** 2);
  assert.ok(Math.abs(got - want) < 1e-12,
    `posterior log-density ${got}, closed form ${want}`);
});

test('the spec-canonical posterior measure carries the conjugate moments', async () => {
  const { ctx } = makeMatCtx(SPEC_POSTERIOR, { sampleCount: 20000 });
  const post = await ctx.getMeasure('posterior');
  const field = post.fields ? post.fields.mu : post;
  const s = field.samples;
  const lw = post.logWeights || field.logWeights;
  assert.ok(s && lw, 'posterior materialises as a weighted measure over mu');
  // Normal-Normal conjugacy: precision 1/4 + 1 = 1.25, so variance 0.8 and
  // mean 0.8 * 2.5 = 2.0. Independent of the engine.
  let mx = -Infinity;
  for (let i = 0; i < lw.length; i++) if (lw[i] > mx) mx = lw[i];
  let sw = 0, m1 = 0, m2 = 0;
  for (let i = 0; i < s.length; i++) {
    const w = Math.exp(lw[i] - mx);
    sw += w; m1 += w * s[i]; m2 += w * s[i] * s[i];
  }
  const mean = m1 / sw;
  const sd = Math.sqrt(m2 / sw - mean * mean);
  assert.ok(Math.abs(mean - 2.0) < 0.05, `IS mean ${mean}, want 2.0`);
  assert.ok(Math.abs(sd - Math.sqrt(0.8)) < 0.05, `IS sd ${sd}, want ${Math.sqrt(0.8)}`);
});

// A prior that is neither the law of the likelihood's boundary draws NOR a joint
// over the boundary's own name still has no materialisation, and must say so
// rather than vanish.
const MISMATCHED_PRIOR = `
mu = elementof(reals)
model = Normal(mu = mu, sigma = 1.0)
obs = 2.5
L = likelihoodof(functionof(model), obs)
other = elementof(reals)
prior = joint(other = Normal(mu = 0, sigma = 2.0))
posterior = bayesupdate(L, prior)
`;

test('a posterior the engine cannot materialise emits a loud diagnostic', () => {
  const ds = orchestrator.buildDerivations(processSource(MISMATCHED_PRIOR).bindings);
  const diags = (ds.diagnostics || []).filter((d: any) => d.severity === 'error');
  const hit = diags.find((d: any) =>
    /posterior/i.test(d.message) && /bayesupdate/i.test(d.message));
  if (ds.derivations.posterior === undefined) {
    assert.ok(hit, 'a loud error diagnostic names the pruned bayesupdate posterior');
    assert.match(hit.message, /disintegrat/i,
      'diagnostic points at the disintegration idiom');
  } else {
    // Classified instead: then it must materialise, not mislead. Scoring it is
    // the assertion — a wrong shape throws here rather than passing silently.
    assert.equal(ds.derivations.posterior.kind, 'bayesupdate');
  }
});

// A `joint_likelihood` L (spec §06 "Combining likelihoods") resolves per term
// rather than as one L→K chain, which `_resolveLikelihood` declines. The
// posterior therefore got no derivation at all: the density path could not
// score it and the measure path had nothing to sample. Both now fold the terms
// — the product of independent terms, so the per-atom log-weight is the SUM —
// and both are pinned against the closed-form Gaussian combination.

const TWO_INSTRUMENTS = `
obs_a = 1.5
obs_b = 3.2
mu = elementof(reals)
model_a = functionof(Normal(mu = mu, sigma = 1.0))
model_b = functionof(Normal(mu = 2.0 * mu, sigma = 0.5))
L_a = likelihoodof(model_a, obs_a)
L_b = likelihoodof(model_b, obs_b)
L = joint_likelihood(L_a, L_b)
prior = joint(mu = Normal(mu = 0.0, sigma = 2.0))
posterior = bayesupdate(L, prior)
`;

test('a joint_likelihood posterior scores the sum of its terms plus the prior',
  async () => {
    const L2P = Math.log(2 * Math.PI);
    // Three closed-form Normal terms, written out.
    const closed = (mu: number) =>
      (-0.5 * L2P - 0.5 * (1.5 - mu) ** 2)
      + (-0.5 * L2P - Math.log(0.5) - 0.5 * ((3.2 - 2 * mu) / 0.5) ** 2)
      + (-0.5 * L2P - Math.log(2) - 0.5 * (mu / 2) ** 2);
    for (const mu of [1.6, 0.0, -0.8]) {
      const { ctx } = makeMatCtx(
        TWO_INSTRUMENTS + `lp = logdensityof(posterior, record(mu = ${mu}))\n`,
        { sampleCount: 8 });
      const m = await ctx.getMeasure('lp');
      const got = m.value ? m.value.data[0] : m.samples[0];
      assert.ok(Math.abs(got - closed(mu)) < 1e-9,
        `mu=${mu}: got ${got}, closed form ${closed(mu)}`);
    }
  });

test('a joint_likelihood posterior MEASURE carries the combined moments', async () => {
  const { ctx, built } = makeMatCtx(TWO_INSTRUMENTS, { sampleCount: 40000 });
  assert.equal(built.derivations.posterior.kind, 'bayesupdate');
  assert.equal(built.derivations.posterior.subs.length, 2);
  const post = await ctx.getMeasure('posterior');
  const field = post.fields ? post.fields.mu : post;
  const s = field.samples;
  const lw = post.logWeights || field.logWeights;
  // Gaussian precisions in mu: instrument A contributes 1, instrument B
  // contributes (2/0.5)^2 = 16 (it reads 2*mu with sigma 0.5), the prior 1/4.
  const precA = 1, precB = (2 / 0.5) ** 2, precP = 1 / 4;
  const prec = precA + precB + precP;
  const wantMean = (precA * 1.5 + precB * (3.2 / 2)) / prec;
  const wantSd = Math.sqrt(1 / prec);
  let mx = -Infinity;
  for (let i = 0; i < lw.length; i++) if (lw[i] > mx) mx = lw[i];
  let sw = 0, m1 = 0, m2 = 0;
  for (let i = 0; i < s.length; i++) {
    const w = Math.exp(lw[i] - mx);
    sw += w; m1 += w * s[i]; m2 += w * s[i] * s[i];
  }
  const mean = m1 / sw;
  const sd = Math.sqrt(m2 / sw - mean * mean);
  assert.ok(Math.abs(mean - wantMean) < 0.02, `IS mean ${mean}, want ${wantMean}`);
  assert.ok(Math.abs(sd - wantSd) < 0.02, `IS sd ${sd}, want ${wantSd}`);
});

test('an MCMC backend refuses a multi-term joint posterior by name', async () => {
  const { ctx } = makeMatCtx(TWO_INSTRUMENTS, { sampleCount: 64 });
  ctx.inferenceOpts = { backend: 'mh' };
  await assert.rejects(() => ctx.getMeasure('posterior'),
    /joint_likelihood of more than one term/);
});

test('an implicit kernel reaches a parametric leaf behind a nested boundary', () => {
  // The implicit-reification walk (spec §04: clicking `x` is plotting
  // `kernelof(x)`) descends the measure graph for parametric leaves.
  // `collectSelfRefs` treats an identifier-bound boundary as the callable's
  // own INPUT, correctly, so a point-free `functionof(Normal(mu = mu, …))`
  // reported nothing and the walk surfaced no inputs at all. It now follows
  // `paramSources`, which names the binding feeding that boundary.
  const src = `
obs = 2.5
mu = elementof(reals)
model = functionof(Normal(mu = mu, sigma = 1.0))
L = likelihoodof(model, obs)
other = elementof(reals)
prior = joint(other = Normal(mu = 0.0, sigma = 1.0))
posterior = bayesupdate(L, prior)
`;
  const ds = orchestrator.buildDerivations(processSource(src).bindings);
  // The signature is asked for directly: it is what the viewer's implicit path
  // consults for any measure-typed binding whose derivation was pruned.
  const sig = orchestrator.implicitKernelSignature(
    'posterior', ds.bindings, ds.derivations);
  assert.ok(sig, 'an implicit kernel signature is produced');
  assert.ok(sig.inputs.some((i: any) => i.paramName === 'mu'),
    'the leaf behind the kernel boundary is an input: '
    + sig.inputs.map((i: any) => i.paramName).join(', '));
});

test('a joint term whose body needs an unresolvable ref prunes the whole score', () => {
  // The cascade-prune verdict per joint term, driven directly so the resolvable
  // set is explicit. A joint likelihood is only as scoreable as its least
  // scoreable term: one unresolved fixed-phase dependency prunes the whole
  // score rather than silently reporting the terms that happen to work.
  const derivations = require('../derivations.ts');
  const bindings = new Map<string, any>([
    ['sigma', { phase: 'fixed', type: 'literal' }],
    ['mu', { phase: 'parameterized', type: 'input' }],
  ]);
  const term = (sigmaRef: boolean) => ({
    bodyName: null,
    bodyIR: {
      kind: 'call', op: 'Normal',
      kwargs: {
        mu: { kind: 'ref', ns: 'self', name: 'mu' },
        sigma: sigmaRef
          ? { kind: 'ref', ns: 'self', name: 'sigma' }
          : { kind: 'lit', value: 1 },
      },
    },
    obsIR: { kind: 'lit', value: 2.5 },
    paramKwargs: ['mu'], params: ['mu'],
  });
  const d = (subs: any[]) => ({
    kind: 'joint_likelihood_density', subs,
    pointIR: { kind: 'call', op: 'record', fields: [] },
  });
  // `resolvable` is derived from the derivations map plus fixedValues, so the
  // two calls differ only in whether `sigma` carries a value.
  const check = (subs: any[], fixed: string[]) => derivations.derivationRefsValid(
    d(subs), {}, bindings, new Map(fixed.map((n) => [n, 0])), 'lp');

  assert.equal(check([term(false), term(true)], ['sigma']), true,
    'both terms score when the fixed dependency resolves');
  assert.equal(check([term(false), term(true)], []), false,
    "the second term's sigma no longer resolves, so the joint prunes");
  assert.equal(check([{ bodyName: null, bodyIR: null }], ['sigma']), false,
    'a term with neither a body name nor body IR cannot be scored');
});
