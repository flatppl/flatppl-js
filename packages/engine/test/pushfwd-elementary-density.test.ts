'use strict';
// pushfwd-elementary-density.test.ts — spec §06 "Engine contract for pushfwd
// density evaluation", case 1 (known-bijection registry): a conforming engine
// MUST evaluate `densityof(pushfwd(f, M), y)` analytically when f is composed
// of the built-in elementary bijections (exp/log, affine from add/sub/neg/mul/
// divide with positive scaling, pow with literal exponent) — WITHOUT an
// explicit `bijection(f, f_inv, logvolume)` annotation.
//
// Before the fix, density.walkPushfwd only saw bijection metadata for an
// explicitly `bijection`-annotated binding; a plain `fn(0.1 * exp(_))` got
// none and threw "pushfwd … requires a bijection annotation". That blocked
// every score-the-prior sampler (mh/emcee/smc/amis/ess) on the litter
// Beta-Binomial model whose `a_plus_b ~ iid(pushfwd(fn(0.1*exp(_)),
// Exponential(1.5)), G)` is exactly this case. derivations.ts now auto-derives
// the inverse + forward log-volume via bijection-registry.invertExpr.
//
// Oracle: `pushfwd(x -> 0.1·exp(x), Exponential(rate=1.5))` is Pareto with
// scale x_m=0.1, shape α=1.5. Distributions.jl Pareto(α, θ) = Pareto(1.5, 0.1):
//   logpdf(0.5) = -1.3155445799830405
//   logpdf(1.0) = -3.048412531382904
//   logpdf(2.0) = -4.781280482782767
// Support is [0.1, ∞): x < 0.1 ⇒ preimage log(10x) < 0, outside Exponential
// support ⇒ −∞.

const { test }       = require('node:test');
const assert         = require('node:assert/strict');
const { ctxFor }     = require('./density/regression-baseline.test.ts');
const orchestrator   = require('../orchestrator.ts');
const density        = require('../density.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const TOL = 1e-9;

test('pushfwd(0.1·exp, Exponential(1.5)) auto-derives the Pareto density (1e-9, no annotation)', () => {
  const { ctx } = ctxFor(`
pareto = pushfwd(fn(0.1 * exp(_)), Exponential(1.5))
a_plus_b ~ iid(pareto, 1)
prior = lawof(record(a_plus_b = a_plus_b))
posterior = prior
`, 1);
  const raw = orchestrator.expandMeasure('pareto', { derivations: ctx.derivations, bindings: ctx.bindings });
  const opts = { parseSet: () => null, resolveMeasureRef: () => null, baseEnv: { __moduleRegistry: ctx.moduleRegistry } };
  const cases: Array<[number, number]> = [
    [0.5, -1.3155445799830405],
    [1.0, -3.048412531382904],
    [2.0, -4.781280482782767],
  ];
  for (const [x, want] of cases) {
    const lp = density.logDensityN(raw, x, {}, 1, opts)[0];
    assert.ok(Math.abs(lp - want) <= TOL, `logpdf(${x}): got ${lp}, oracle ${want}`);
  }
  // Below the support bound x_m = 0.1 → −∞ (preimage out of Exponential support).
  assert.equal(density.logDensityN(raw, 0.05, {}, 1, opts)[0], -Infinity);
});

test('elementary pushfwd prior is tractable for the litter Beta-Binomial samplers', async () => {
  const { ctx } = ctxFor(`
G = 2
N = 16
n_data = [[13,12,9,9,8,8,13,12,10,10,9,13,5,7,10,10],[12,11,10,9,11,10,10,9,9,5,9,7,10,6,10,7]]
r_data = [[13,12,9,9,8,8,12,11,9,9,8,11,4,5,7,7],[12,11,10,9,10,9,9,8,8,4,7,4,5,3,3,0]]
pareto = pushfwd(fn(0.1 * exp(_)), Exponential(1.5))
a_plus_b ~ iid(pareto, G)
mu ~ iid(Beta(1, 1), G)
a = mu .* a_plus_b
b = (1 .- mu) .* a_plus_b
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
binomial_row_K = (n_row, p_row) -> Binomial.(n_row, p_row)
r ~ binomial_row_K.(n_data, p)
prior = lawof(record(a_plus_b = a_plus_b, mu = mu, p = p))
forward_kernel = kernelof(record(r = r), a_plus_b = a_plus_b, mu = mu, p = p)
L = likelihoodof(forward_kernel, record(r = r_data))
posterior = bayesupdate(L, prior)
`, 1);
  let d: any = null;
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') d = v;
  }
  const { probePrior } = await buildLogPi(ctx, d);
  // The probe must NOT throw "pushfwd … requires a bijection annotation".
  probePrior({
    a_plus_b: Float64Array.from([1.0, 2.0]),
    mu: Float64Array.from([0.5, 0.5]),
    p: Float64Array.from(new Array(32).fill(0.5)),
  });
});
