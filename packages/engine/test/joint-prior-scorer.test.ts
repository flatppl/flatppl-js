'use strict';
// joint-prior-scorer.test.ts — the sampler path must handle a prior built
// directly as a joint/record of distributions (06-measure-algebra.md §joint:
// `prior = joint(mu = Normal(0,1), sigma = Exponential(1))`), where the latent
// variates are `elementof` boundary inputs of the kernel, NOT stochastic `~`
// draws. The latent NAMES are the joint's field labels (they match the kernel's
// parametric inputs); each field's distribution is the measure to score.
//
// Before the fix, buildLogPi/enumerateLatents only collected `type==='draw'`
// bindings transitively feeding the prior; a joint-of-distributions prior has
// none, so the samplers threw "prior 'prior' has no stochastic draws to sample".
//
// Oracle: Distributions.jl (independent), at theta1=-0.5, theta2=2.0:
//   prior = logpdf(Normal(0,1),-0.5) + logpdf(Exponential(rate=1),2.0)
//   a = 5*theta2 = 10 ; b = |theta1|*theta2 = 1
//   lik   = sum(logpdf(Normal(a,b), data))
//   → prior = -3.0439385332046727
//     lik   = -221.63938533204677
//     logPi = -224.68332386525145

const { test }       = require('node:test');
const assert         = require('node:assert/strict');
const fs             = require('node:fs');
const path           = require('node:path');
const { ctxFor }     = require('./density/regression-baseline.test.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const modelSpec      = require('../model-spec.ts');

const TOL = 1e-9;
const SRC = fs.readFileSync(path.join(__dirname, 'fixtures/bayesian_inference_1.flatppl'), 'utf8');

function posteriorDerivOf(ctx: any): any {
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') return v;
  }
  return null;
}

test('enumerateLatents reads a joint-of-distributions prior as field latents', () => {
  const { ctx } = ctxFor(SRC, 1);
  const d = posteriorDerivOf(ctx);
  assert.ok(d, 'bayesupdate derivation present');
  const latents = modelSpec.enumerateLatents(d, ctx);
  const names = latents.map((l: any) => l.name).sort();
  assert.deepEqual(names, ['theta1', 'theta2'], `latents: ${JSON.stringify(latents)}`);
  const byName: Record<string, any> = {};
  for (const l of latents) byName[l.name] = l;
  assert.equal(byName.theta1.support.kind, 'real', 'theta1 ~ Normal → real support');
  assert.equal(byName.theta2.support.kind, 'positive', 'theta2 ~ Exponential → positive support');
});

test('buildLogPi scores a joint-of-distributions prior to the Distributions.jl oracle (1e-9)', async () => {
  const { ctx } = ctxFor(SRC, 1);
  const d = posteriorDerivOf(ctx);
  const { logPi, priorOf, likOf } = await buildLogPi(ctx, d);
  const pt = { theta1: -0.5, theta2: 2.0 };
  const gotPrior = priorOf(pt);
  const gotLik   = likOf(pt);
  const gotLogPi = logPi(pt);
  assert.ok(Math.abs(gotPrior - (-3.0439385332046727)) <= TOL, `prior: got ${gotPrior}`);
  assert.ok(Math.abs(gotLik   - (-221.63938533204677)) <= TOL, `lik: got ${gotLik}`);
  assert.ok(Math.abs(gotLogPi - (-224.68332386525145)) <= TOL, `logPi: got ${gotLogPi}`);
});
