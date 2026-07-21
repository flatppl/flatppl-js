'use strict';

// =====================================================================
// mcmc-live-truncate-normalizer.test.ts — Buffy #384
// =====================================================================
//
// mcmc-density.ts's buildLogPi resolves each draw's own prior measure via
// resolveNormalizeMasses ONCE at SETUP (an async pre-pass — the whole point
// of buildLogPi is a SYNCHRONOUS per-step scorer for the MCMC hot loop).
// When a `normalize(truncate(D, S))` prior's massFrom-carrying node is
// resolved by MATERIALISING the inner measure once (ctx.getMeasure) and
// baking the resulting −log Z as a literal, that is correct ONLY when D's
// params are θ-INDEPENDENT (e.g. eight-schools' tau ~
// normalize(truncate(Cauchy(0,5), interval(0,inf))), a genuine constant).
// When D's params reference a LIVE latent — here `sigma`, itself a draw —
// the true log-mass Z(σ) varies every MCMC step, but the old code baked one
// constant at setup and reused it for every subsequent step: every sampler
// backend built on buildLogPi (mh/ram/slice/emcee/amis/smc/nested) scored a
// silently-WRONG density away from the setup-time σ.
//
// The fix classifies a massFrom normalize(truncate(...)) node as θ-dependent
// (its base's params, or the truncation bounds, reference a draw or a value
// derived from one) vs θ-independent. θ-independent keeps the pre-existing
// bake (unchanged, no perf regression). θ-dependent instead strips massFrom
// and leaves the node as a BARE normalize(truncate(...)); density.ts's own
// walkNormalize already resolves −log Z(θ) FRESH on every logDensityN call,
// from the point's own env, via tryResolveTruncateNormalizerShift (the same
// mechanism the per-cell kernel-broadcast walk already relies on for this
// exact shape) — so no new CDF glue code is needed in mcmc-density.ts.
//
// Model: mu = 1.0 (fixed); sigma ~ Gamma(2.0, 1.0) (live latent);
// x ~ normalize(truncate(Normal(mu, sigma), interval(0.0, inf))). The
// correct log-density of a Normal(mu,sigma) truncated to [0,inf) at x is
// normal_logpdf(x;mu,sigma) − log(Φ((inf−mu)/sigma) − Φ((0−mu)/sigma))
// = normal_logpdf(x;mu,sigma) − log(1 − Φ(−mu/sigma)), genuinely
// σ-dependent since mu ≠ 0.
//
// Oracle (scipy.stats, INDEPENDENT of the engine):
//   full(σ) = scipy.stats.truncnorm.logpdf(x=0.5, a=(0-mu)/σ, b=inf, loc=mu,
//             scale=σ) + scipy.stats.gamma.logpdf(σ, a=2.0, scale=1.0)
// computed at 4 distinct σ (pre-fix diverges at every σ except the
// materialisation point; post-fix matches at all of them to 1e-9).

const { test }       = require('node:test');
const assert         = require('node:assert/strict');
const { ctxFor }     = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const SRC = `
mu = 1.0
sigma ~ Gamma(2.0, 1.0)
x ~ normalize(truncate(Normal(mu, sigma), interval(0.0, inf)))
prior = lawof(record(sigma = sigma, x = x))

y_data = 0.5
y ~ Normal(x, 1.0)

forward_kernel = kernelof(record(y = y), sigma = sigma, x = x)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`;

const X_FIXED = 0.5;

// [sigma, oracle full priorOf(sigma,x) = truncnorm_logpdf(x) + gamma_logpdf(sigma)]
// scipy.stats.truncnorm.logpdf(0.5, a=(0-1)/sigma, b=inf, loc=1, scale=sigma)
//   + scipy.stats.gamma.logpdf(sigma, a=2.0, scale=1.0), computed independently.
const ORACLE: Array<[number, number]> = [
  [0.3, -2.6073982696876428],
  [0.7, -1.79438708795525],
  [1.5, -2.18348309820792],
  [3.0, -3.4716783311724484],
];

async function buildPriorOf(): Promise<(pt: Record<string, any>) => number> {
  const { ctx } = ctxFor(SRC, 1);
  const derivations: Record<string, any> = ctx.derivations;
  let d: any = null;
  for (const [, v] of Object.entries(derivations)) {
    if (v && (v as any).kind === 'bayesupdate') { d = v; break; }
  }
  assert.ok(d != null, 'no bayesupdate derivation found in ctx.derivations');
  const { priorOf } = await buildLogPi(ctx, d);
  return priorOf;
}

for (const [sigma, oracle] of ORACLE) {
  test(`#384 θ-dependent normalize(truncate(Normal(mu,σ),[0,inf))) at σ=${sigma} matches scipy oracle`, async () => {
    const priorOf = await buildPriorOf();
    const got = priorOf({ sigma, x: X_FIXED });
    assert.ok(
      Math.abs(got - oracle) <= 1e-9,
      `σ=${sigma}: engine ${got} vs scipy oracle ${oracle} (Δ ${Math.abs(got - oracle)}) — `
      + `a σ-constant Z bake at setup (the #384 bug) mis-scores here`,
    );
  });
}

test('#384 the four σ scores are DISTINCT (the normalizer is not baked σ-constant)', async () => {
  const priorOf = await buildPriorOf();
  const scores = ORACLE.map(([sigma]) => priorOf({ sigma, x: X_FIXED }));
  for (let i = 1; i < scores.length; i++) {
    assert.ok(
      Math.abs(scores[i] - scores[i - 1]) > 1e-3,
      `consecutive σ scores must differ; got ${JSON.stringify(scores)}`,
    );
  }
});

// Regression guard for the FIXED-params path (eight-schools' tau ~
// normalize(truncate(Cauchy(0,5), interval(0,inf))) — literal Cauchy params,
// genuinely θ-independent): must still take the bake-once fast path
// unchanged, i.e. produce a finite, self-consistent score. This does not
// re-derive an independent oracle (regression-baseline.test.ts /
// mcmc-scorer-baseline.test.ts already pin eight-schools to Distributions.jl
// golden values); it only guards that classifying θ-dependence didn't
// misclassify a genuinely-fixed truncate normalizer as deferrable (or vice
// versa) and break the existing fast path.
test('#384 fixed-params normalize(truncate(Cauchy(0,5),[0,inf))) (eight-schools tau) is unaffected', async () => {
  const fixedSrc = `
y_data = [28, 8, -3, 7, -1, 1, 18, 12]
std_errs_data = [15, 10, 16, 11, 9, 11, 10, 18]
J = 8

mu ~ Normal(0, 5)
tau ~ normalize(truncate(Cauchy(0, 5), interval(0, inf)))
theta ~ iid(Normal(mu, tau), J)

prior = lawof(record(mu = mu, tau = tau, theta = theta))

y ~ Normal.(theta, std_errs_data)

forward_kernel = kernelof(record(y = y), mu = mu, tau = tau, theta = theta)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(fixedSrc, 1);
  const derivations: Record<string, any> = ctx.derivations;
  let d: any = null;
  for (const [, v] of Object.entries(derivations)) {
    if (v && (v as any).kind === 'bayesupdate') { d = v; break; }
  }
  const { priorOf } = await buildLogPi(ctx, d);
  const pt = {
    mu: 0.0, tau: 1.0,
    theta: Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
  };
  const got = priorOf(pt);
  // Golden value from mcmc-scorer-baseline.test.ts's eight-schools case
  // (Distributions.jl-validated), asserted there to 1e-9 already; re-asserted
  // here at a looser tolerance purely as a "the fast path still runs and
  // still agrees" guard against this fix's classification logic.
  assert.ok(Number.isFinite(got), `tau's fixed-param normalizer must still score finitely, got ${got}`);
  assert.ok(
    Math.abs(got - (-113.98012604215299)) <= 1e-6,
    `fixed-params bake path regressed: got ${got}, expected -113.98012604215299`,
  );
});
