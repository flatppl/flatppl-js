'use strict';

// ════════════════════════════════════════════════════════════════════════
// STANDING INVARIANT for the measure-lowering unification
// (flatppl-dev/measure-lowering-unification-plan.md, Phase 0).
//
// The engine walks the SAME measure IR along two paths — SAMPLE and DENSITY —
// that drift. This harness pins the property the unification must make
// structural: for a CLOSED scalar measure M, `logdensityof(M, x)` agrees with
// the empirical log-density of the SAMPLE histogram of the same M (compared as
// a RATIO between probe points spread across the support, cancelling the
// normalisation constant + bin width). Harness in test/_agreement-harness.ts;
// it is self-checked on a plain Normal + a truncate below.
//
// Each fixture is tagged by audit ID + DISPOSITION:
//   GREEN       — already correct (sweep fix or always-correct); guards against
//                 regression. assert(ok).
//   WILL-FLIP   — still divergent; the CLM unification (or an earlier targeted
//                 fix) will fix it. assert(!ok) TIED TO THE FAILURE MODE (so it
//                 is red for the RIGHT reason); flips loudly to a test FAILURE
//                 when fixed → re-tag to GREEN.
//   OUT-OF-SCOPE — a divergence CLM does NOT address (M2 selector,
//                 MvNormal-kchain); test.skip with a note, so the eventual
//                 "all green" gate is not falsely blocked.
//
// The scalar-ratio harness is BLIND to pure normalisation-constant bugs (M3
// totalmass) and cannot see correlation (H7) or density-curve shape (H10);
// those use fieldCorrelation here / dedicated checks at their phase.
// ════════════════════════════════════════════════════════════════════════

const test = require('node:test');
const assert = require('node:assert');
const { agreement, fieldCorrelation } = require('./_agreement-harness.ts');

// ---- harness self-checks (MUST pass — they validate the harness itself) ----

test('agreement self-check: closed Normal(0,1) sample ≡ density', async () => {
  const r = await agreement('m = Normal(0.0, 1.0)\n', 'm', { N: 60000, tol: 0.3 });
  assert.ok(r.ok, `self-check failed (maxErr=${r.maxErr}, ${r.reason || ''}): ` + JSON.stringify(r.probes));
});

test('agreement self-check: closed truncate(Normal) sample ≡ density (C1)', async () => {
  const r = await agreement('m = truncate(Normal(0.0, 1.0), interval(-2.0, 2.0))\n', 'm', { N: 60000, tol: 0.3 });
  assert.ok(r.ok, `truncate self-check failed (maxErr=${r.maxErr}, ${r.reason || ''}): ` + JSON.stringify(r.probes));
});

// ---- GREEN regression guards (sweep fixes / always-correct) --------------

const GREEN: Array<[string, string, string]> = [
  ['kchain hole-param over a wide prior (M1-family)', 'ch', `
a ~ Normal(0.0, 0.01)
K = functionof(Normal(mu = a, sigma = 1.0), a = a)
M = Normal(0.0, 10.0)
ch = kchain(M, K)`],
  ['M1: kchain over a RELABELLED prior (density ≡ histogram)', 'ch', `
theta ~ Normal(0.0, 1.0)
shifted = theta + 20.0
M = lawof(shifted)
K = functionof(Normal(mu = shifted, sigma = 1.0), shifted = shifted)
ch = kchain(M, K)`],
  ['kchain marginalises the prior (the correct contrast to H8)', 'ch', `
theta ~ Normal(0.0, 1.0)
K = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
prior = lawof(theta)
ch = kchain(prior, K)`],
  ['hole-kernel over a RECORD base prior (cat-arity case A)', 'ch', `
mu_p = 0.0
joint_indep = joint(t1 = Normal(mu = mu_p, sigma = 1.0), t2 = Exponential(rate = 1.0))
K = functionof(Normal(mu = t1, sigma = t2), t1 = t1, t2 = t2)
ch = kchain(joint_indep, K)`],
];

for (const [id, mname, src] of GREEN) {
  test(`[GREEN] ${id}`, async () => {
    const r = await agreement(src, mname, { N: 60000, tol: 0.4 });
    assert.ok(r.ok, `expected GREEN but got ${r.crashed ? 'CRASH:' + r.reason : 'RED maxErr=' + (r.maxErr || 0).toFixed(2)} — ` + JSON.stringify(r.probes || r.reason));
  });
}

// ---- WILL-FLIP (red today, for the right reason; flip loudly when fixed) --

test('[GREEN H8] lawof(draw with a stochastic ancestor) marginalises — density ≡ histogram', async () => {
  // pp = lawof(obs), obs~Normal(theta,1), theta~Normal(0,1). Sample ⇒ marginal
  // Normal(0,√2). The density used to score the per-atom conditional Normal(·,1)
  // — a VARIANCE mismatch glaring in the tails (maxErr≈4.15). FIXED (CLM Phase 3):
  // lowerMeasure recognises the marginalised stochastic ancestor (theta is a
  // `shared` body ref to a stochastic binding, not a retained variate) and sets
  // reduce={marginal}; applyReduce does the logsumexp − logN over theta ~ prior,
  // the same MC marginal kchain uses. Now a regression guard.
  const r = await agreement(`
theta ~ Normal(0.0, 1.0)
obs ~ Normal(mu = theta, sigma = 1.0)
pp = lawof(obs)`, 'pp', { N: 60000, tol: 0.35 });
  assert.ok(r.ok, `H8 regression — expected marginal agreement (Normal(0,√2)), got ` +
    `${r.crashed ? 'CRASH:' + r.reason : 'maxErr=' + (r.maxErr || 0).toFixed(2)}: ` +
    JSON.stringify(r.probes || r.reason));
});

test('[WILL-FLIP gen] standalone logdensityof(lawof(generative composite)) is unimplemented', async () => {
  // Single-event transport: z = f(x, uniform) at fixed pars. The MC-marginal
  // density works through the profile/bayesupdate path but NOT as a bare
  // logdensityof(lawof(z), x) — a density gap the CLM reduce={marginal} closes.
  const r = await agreement(`
sigma = 0.2
mu = 1.1
a = 0.1
b = 0.3
x ~ Normal(mu, sigma)
delta = (2.0 * draw(Uniform(interval(0, 1))) + 1.0) * a
y = (x + delta)^3 * exp(x - b)
z = y / 2.0
m = lawof(z)`, 'm', { N: 30000, tol: 0.4 });
  assert.ok(!r.ok && r.crashed === 'density',
    `generative-composite standalone density is expected to be an unimplemented GAP (density crash) until CLM; ` +
    `got ${r.ok ? 'GREEN — FLIP IT' : (r.crashed || 'RED') + ': ' + (r.reason || r.maxErr)}`);
});

test('[WILL-FLIP H7] joint(m, m) reuses the identical atoms instead of independent draws', async () => {
  // joint is the INDEPENDENT product (spec §06) ⇒ corr≈0; the bug returns the
  // memoised atom batch ⇒ corr=1. Fixed by re-seeding a reused factor.
  const corr = await fieldCorrelation(`
m = Normal(0.0, 1.0)
j = joint(a = m, b = m)`, 'j', 'a', 'b', { N: 30000 });
  assert.ok(corr > 0.9,
    `H7 is expected to still be perfectly correlated (corr>0.9) until the reused-factor re-seed lands; ` +
    `got corr=${corr.toFixed(3)} — if ≈0, FLIP IT to a regression guard`);
});

test('[WILL-FLIP H7b/B] joint(posterior, posterior) — reused weighted factor', async () => {
  // The critique's high-severity case: reusing a bayesupdate posterior as two
  // factors must agree on the IS-weighted estimate, not just corr. Today it is
  // an outright gap (crash); CLM must re-seed AND combine the weight streams.
  let crashed = false;
  try {
    await fieldCorrelation(`
theta ~ Normal(0.0, 2.0)
obs ~ iid(Normal(mu = theta, sigma = 1.0), 5)
fwd = kernelof(record(obs = obs), theta = theta)
data = [2.0, 2.1, 1.9, 2.2, 1.8]
L = likelihoodof(fwd, record(obs = data))
prior = lawof(record(theta = theta))
post = bayesupdate(L, prior)
j = joint(a = post, b = post)`, 'j', 'a', 'b', { N: 6000 });
  } catch (_) { crashed = true; }
  assert.ok(crashed,
    'joint(posterior, posterior) is expected to still be an unsupported GAP until CLM handles reused weighted factors; ' +
    'if it now succeeds, FLIP IT to an IS-weight agreement guard');
});

// ---- OUT-OF-SCOPE for CLM (documented; not gated by the "all green" goal) --

test.skip('[OUT-OF-SCOPE M2] comparison-selector mixture density pools the selector (not fixed by CLM)', () => {});
test.skip('[OUT-OF-SCOPE] MvNormal / vector-variate kchain kernel (sampling unimplemented; not fixed by CLM)', () => {});
