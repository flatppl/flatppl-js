'use strict';
// composite-iid-prior.test.ts — oracle gate for the composite-iid kernel-broadcast
// prior scorer (mcmc-density.scoreComposite, via composite-prior.ts).
//
// A draw whose measure is a kernel-broadcast over a user function returning an
// iid block — p ~ beta_row_K.(a,b), a [G,N] matrix of Betas — is scored by
// expanding the per-row (a_g,b_g) across the inner iid axis into a flat product
// of Betas. surgical-failures is the only CORPUS model with this shape, and it
// refuses (intractable pushfwd prior) before the scorer runs, so this path had
// no numeric coverage. Here a,b are FIXED (no pushfwd) ⇒ the model is tractable
// ⇒ the scorer runs, and we gate its prior log-density against scipy.
//
//   prior(p) = Σ_{g,j} logpdf_Beta(p[g][j]; a_g, b_g)
//   scipy.stats.beta.logpdf, a=[(2,5),(3,4)], p=[[.3,.5,.4],[.6,.2,.5]] → 2.306058297069055

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const { ctxFor }     = require('./density/regression-baseline.test.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const { materialiser } = require('..');

// Composite-iid prior with FIXED row params (no pushfwd) ⇒ tractable ⇒ reachable.
const MODEL = `
G = 2
N = 3
n_data = [[10, 10, 10], [10, 10, 10]]
r_data = [[7, 8, 6], [5, 6, 4]]
a = [2.0, 3.0]
b = [5.0, 4.0]
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
binomial_row_K = (n_row, p_row) -> Binomial.(n_row, p_row)
r ~ binomial_row_K.(n_data, p)
prior = lawof(record(p = p))
forward_kernel = kernelof(record(r = r), p = p)
L = likelihoodof(forward_kernel, record(r = r_data))
posterior = bayesupdate(L, prior)
`;

const ORACLE = 2.306058297069055;   // scipy Beta-product, see header
const PT = Float64Array.from([0.3, 0.5, 0.4, 0.6, 0.2, 0.5]);   // p flat, row-major [g*N+j]

test('composite-iid Beta prior: scorer matches the scipy Beta-product oracle', async () => {
  const ctx = ctxFor(MODEL, 50).ctx;
  const dv = ctx.lookupDerivation ? ctx.lookupDerivation('posterior') : ctx.derivations.posterior;
  const { priorOf } = await buildLogPi(ctx, dv);
  const lp = priorOf({ p: PT });
  assert.ok(Math.abs(lp - ORACLE) < 1e-9, `composite-iid prior ${lp} vs scipy ${ORACLE}`);
});

test('composite-iid model is reachable (no refusal) and samples a posterior', async () => {
  const m = await materialiser.materialiseMeasure('posterior', ctxFor(MODEL, 1000).ctx,
    { backend: 'mh', chains: 4, warmup: 300, draws: 300, seed: 1 });
  assert.ok(m.fields && m.fields.p, 'posterior has field p');
  const s = m.fields.p.samples || (m.fields.p.value && m.fields.p.value.data);
  assert.ok(s && s.length > 0, 'p has samples');
  for (let i = 0; i < s.length; i++) assert.ok(s[i] > 0 && s[i] < 1, 'p in (0,1)');
});
