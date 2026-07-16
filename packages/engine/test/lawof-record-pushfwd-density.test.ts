'use strict';

// #260 (d): lawof(record(f_1 = t_1, ..., f_n = t_n)) recognized as a
// pushforward DENSITY when the field-map ancestors→fields is a DIAGONAL
// bijection: each field is either a bare self-ref to a stochastic latent, or
// a straight-line known-bijection transform (bijection-registry.invertExpr,
// oracle-verified in #260 (b)/(c) — reused here, not re-derived) of EXACTLY
// ONE stochastic ancestor, every ancestor used by AT MOST ONE field, and
// #fields == #distinct ancestors (so the Jacobian is diagonal).
//
// Before this fix, `sigma2 ~ Exponential(1.0); m =
// lawof(record(sigma = sqrt(sigma2)))` threw "density: unsupported measure
// op 'sqrt'" — the field's deterministic-transform derivation (kind
// 'evaluate') fell through expandMeasureIR's structural fallback, leaking
// the raw `sqrt` op into density's dispatcher. derivations.ts's `case
// 'record'` in `_expandByName` now recognises the diagonal-bijection shape
// and synthesises the same `pushfwd` bijMeta the annotation-free `pushfwd`
// case (#260 (a)/(b)) already attaches, per field.
//
// Every oracle value below is computed independently via scipy (python MCP)
// — see the comment on each test for the closed-form change-of-variables
// used. Test 1 additionally cross-checks against the equivalent
// `pushfwd(fn(sqrt(_)), Exponential(1))` form #260 (b) already scores
// (bit-identical, not just within tolerance).
//
// HARD sampling-path constraint (#260 (d) brief): this recognition is
// density-direction only (gated on `_expandByName` being called WITH
// `bindings` — matRecord, the sampling path, never calls expandMeasure/
// expandMeasureIR at all, so it is structurally unreachable from sampling).
// The last test proves sampling the same record measure is unaffected.

const { test }   = require('node:test');
const assert     = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const TOL = 1e-9;

function compileErrors(proc: any): any[] {
  return proc.diagnostics.filter((d: any) => d.severity === 'error');
}

async function scalarOf(src: string, name: string): Promise<number> {
  const { proc, ctx } = ctxFor(src, 8);
  assert.deepEqual(compileErrors(proc), [],
    'diagnostics: ' + JSON.stringify(compileErrors(proc)));
  const m: any = await ctx.getMeasure(name);
  return m.samples[0];
}

test('#260 (d) single field: record(sigma = sqrt(sigma2)) scores the sqrt-pushforward density '
  + '(scipy oracle + bit-identical cross-check vs the equivalent pushfwd(fn(sqrt(_)), M) form)', async () => {
  const lp = await scalarOf(`
sigma2 ~ Exponential(1.0)
prior = lawof(record(sigma = sqrt(sigma2)))
lp = logdensityof(prior, record(sigma = 1.5))
`, 'lp');
  // scipy oracle (independent, python MCP):
  //   sigma2 = sigma**2 = 2.25
  //   scipy.stats.expon.logpdf(2.25, scale=1.0) + log(2*sigma)   [d(sigma^2)/d(sigma) = 2*sigma]
  //   = -2.25 + log(3.0) = -1.1513877113318902
  const oracle = -1.1513877113318902;
  assert.ok(Math.abs(lp - oracle) < TOL, `got ${lp}, oracle ${oracle}`);

  // Cross-check: the equivalent annotation-free pushfwd form (#260 (b))
  // must score BIT-IDENTICALLY (same invertExpr synthesis, same base law).
  const lpPushfwd = await scalarOf(`
pf = pushfwd(fn(sqrt(_)), Exponential(1.0))
lp = logdensityof(lawof(pf), 1.5)
`, 'lp');
  assert.equal(lp, lpPushfwd,
    'record(sigma = sqrt(sigma2)) must score identically to pushfwd(fn(sqrt(_)), Exponential(1))');
});

test('#260 (d) multi-field distinct ancestors: record(x = exp(a), y = sqrt(b)) sums independent '
  + 'pushforward densities (scipy oracle)', async () => {
  const lp = await scalarOf(`
a ~ Normal(0.0, 1.0)
b ~ Exponential(1.0)
prior = lawof(record(x = exp(a), y = sqrt(b)))
lp = logdensityof(prior, record(x = 2.0, y = 1.2))
`, 'lp');
  // scipy oracle (independent, python MCP): LogNormal(0,1) logpdf(2.0)
  // [a = log(2.0); norm.logpdf(a,0,1) - log(2.0)] + sqrt-pushforward
  // logpdf(1.2) of Exponential(1) [b = 1.2**2; expon.logpdf(b) + log(2*1.2)]
  //   = -2.416843483369819
  const oracle = -2.416843483369819;
  assert.ok(Math.abs(lp - oracle) < TOL, `got ${lp}, oracle ${oracle}`);
});

test('#260 (d) multi-op chain field: record(z = 2*exp(a)+1) (affine ∘ exp) scores via the '
  + 'chained inverse (scipy oracle)', async () => {
  const lp = await scalarOf(`
a ~ Normal(0.0, 1.0)
prior = lawof(record(z = 2.0 * exp(a) + 1.0))
lp = logdensityof(prior, record(z = 3.0))
`, 'lp');
  // scipy oracle (independent, python MCP): a = log((z-1)/2) = 0;
  // norm.logpdf(a,0,1) - log(dz/da), dz/da = 2*exp(a) = 2
  //   = -1.612085713764618
  const oracle = -1.612085713764618;
  assert.ok(Math.abs(lp - oracle) < TOL, `got ${lp}, oracle ${oracle}`);
});

test('#260 (d) REFUSE: shared ancestor record(a = x, b = sqrt(x)) throws a targeted diagnostic '
  + '(not a silent number, not the generic unsupported-op error)', async () => {
  const { proc, ctx } = ctxFor(`
x ~ Exponential(1.0)
prior = lawof(record(a = x, b = sqrt(x)))
lp = logdensityof(prior, record(a = 1.0, b = 1.0))
`, 8);
  assert.deepEqual(compileErrors(proc), []);
  await assert.rejects(() => ctx.getMeasure('lp'), /share the ancestor 'x'/);
});

test('#260 (d) REFUSE: multi-argument field record(s = x + y) (both latents) throws a targeted '
  + 'diagnostic (not a silent number, not the generic unsupported-op error)', async () => {
  const { proc, ctx } = ctxFor(`
x ~ Normal(0.0, 1.0)
y ~ Normal(0.0, 1.0)
prior = lawof(record(s = x + y))
lp = logdensityof(prior, record(s = 1.0))
`, 8);
  assert.deepEqual(compileErrors(proc), []);
  await assert.rejects(() => ctx.getMeasure('lp'), /multi-argument transform of ≥2 latents/);
});

test('#260 (d) REFUSE: off-domain transform record(w = log(a)) with a ~ Normal(0,1) (reals '
  + 'support) throws per (c)\'s reused pushfwd domain guard (not a silent sub-probability density)', async () => {
  const { proc, ctx } = ctxFor(`
a ~ Normal(0.0, 1.0)
prior = lawof(record(w = log(a)))
lp = logdensityof(prior, record(w = 1.0))
`, 8);
  assert.deepEqual(compileErrors(proc), []);
  await assert.rejects(() => ctx.getMeasure('lp'), /requires the support of ancestor 'a'/);
});

test('#260 (d) SAMPLING regression: lawof(record(sigma = sqrt(sigma2))) still samples sqrt(sigma2) '
  + 'unchanged (seed-fixed, unaffected by the new density-only recognition)', async () => {
  const { proc, ctx } = ctxFor(`
sigma2 ~ Exponential(1.0)
prior = lawof(record(sigma = sqrt(sigma2)))
`, 32);
  assert.deepEqual(compileErrors(proc), []);
  const sigma2M: any = await ctx.getMeasure('sigma2');
  const priorM: any = await ctx.getMeasure('prior');
  const sigmaSamples = priorM.fields.sigma.samples;
  assert.equal(sigmaSamples.length, 32);
  for (let i = 0; i < 32; i++) {
    const want = Math.sqrt(sigma2M.samples[i]);
    assert.ok(Math.abs(sigmaSamples[i] - want) < 1e-12,
      `atom ${i}: sigma=${sigmaSamples[i]}, sqrt(sigma2)=${want}`);
  }
});
