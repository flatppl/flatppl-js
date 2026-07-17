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

// ---------------------------------------------------------------------------
// #311: dependent-threaded hierarchical base. #260 (d) REFUSED whenever two
// fields' ancestors shared stochastic ancestry (independent-product assembly
// is silently wrong for a hierarchical base — see the #260 (d) history this
// file used to document above). This is the follow-up: a DIRECT hierarchy
// (one field's ancestor's law references ANOTHER field's ancestor, e.g.
// `b ~ Normal(a, 1)` with both `a` and `b` exposed as fields) is now SCORED —
// each ancestor's inverse image is threaded into its dependent siblings' laws
// (derivations.ts's topological-order field re-emission + `threadAs` +
// density.ts's `walkJointFieldsOrPositional` overlay extension). A common
// UN-EXPOSED stochastic ancestor (a hyperparameter, or a hidden intermediate
// in a longer chain) still REFUSES — threading needs an OBSERVED inverse
// image, which an un-exposed latent doesn't have.
//
// Regression pinned: before the independence check existed at all,
// `record(m = 1.0*a, y = b)` with `b ~ Normal(a, 1)` scored a finite WRONG
// number (-2.5817; scipy-correct -1.9629) by treating the base as an
// independent product. The dependent-threading below must reach the SAME
// scipy oracle, not that old wrong number.
// ---------------------------------------------------------------------------

test('#311 SCORE: trivial-affine hierarchical base — record(m = 1.0*a, y = b) with '
  + 'b ~ Normal(a,1) threads a\'s inverse image into b\'s law (scipy oracle; == the bare-ref '
  + 'case since the transform is the identity)', async () => {
  const lp = await scalarOf(`
a ~ Normal(0.0, 1.0)
b ~ Normal(a, 1.0)
prior = lawof(record(m = 1.0 * a, y = b))
lp = logdensityof(prior, record(m = 0.5, y = 0.5))
`, 'lp');
  // scipy oracle (independent, python MCP): a = m = 0.5 (identity transform)
  //   norm.logpdf(0.5, 0, 1) + norm.logpdf(0.5, loc=0.5, scale=1) = -1.9628770664093453
  const oracle = -1.9628770664093453;
  assert.ok(Math.abs(lp - oracle) < TOL, `got ${lp}, oracle ${oracle}`);
});

test('#311 SCORE: exp-transform hierarchical base — record(m = exp(a), y = b) with '
  + 'b ~ Normal(a,1) threads a\'s inverted (log) observed value into b\'s law (scipy oracle)', async () => {
  const lp = await scalarOf(`
a ~ Normal(0.0, 1.0)
b ~ Normal(a, 1.0)
prior = lawof(record(m = exp(a), y = b))
lp = logdensityof(prior, record(m = 1.5, y = 0.5))
`, 'lp');
  // scipy oracle (independent, python MCP): a = log(1.5) = 0.4054651081081644
  //   norm.logpdf(a, 0, 1) - a + norm.logpdf(0.5, loc=a, scale=1) = -2.3300115743565932
  const oracle = -2.3300115743565932;
  assert.ok(Math.abs(lp - oracle) < TOL, `got ${lp}, oracle ${oracle}`);
});

test('#311 SCORE: 3-field chain a -> b -> c (record(z = c, x = exp(a), y = b), fields declared '
  + 'OUT of topological order) — each ancestor threads into its dependent sibling in turn '
  + '(scipy oracle; also proves the field re-emission reorders correctly)', async () => {
  const lp = await scalarOf(`
a ~ Normal(0.0, 1.0)
b ~ Normal(a, 1.0)
c ~ Normal(b, 1.0)
prior = lawof(record(z = c, x = exp(a), y = b))
lp = logdensityof(prior, record(z = -0.1, x = 1.2, y = 0.3))
`, 'lp');
  // scipy oracle (independent, python MCP): a = log(1.2) = 0.1823215567939546
  //   norm.logpdf(a, 0, 1) - a
  //   + norm.logpdf(0.3, loc=a, scale=1)
  //   + norm.logpdf(-0.1, loc=0.3, scale=1)
  //   = -3.0426818394415576
  const oracle = -3.0426818394415576;
  assert.ok(Math.abs(lp - oracle) < TOL, `got ${lp}, oracle ${oracle}`);
});

test('#311 REFUSE: shared stochastic hyperparameter — a ~ Normal(h,1), b ~ Normal(h,1) with '
  + '`h` NOT itself a record field — still refused (threading needs h\'s OBSERVED inverse '
  + 'image, which an un-exposed latent does not have; not the same as direct hierarchy)', async () => {
  const { proc, ctx } = ctxFor(`
h ~ Normal(0.0, 1.0)
a ~ Normal(h, 1.0)
b ~ Normal(h, 1.0)
prior = lawof(record(x = exp(a), y = b))
lp = logdensityof(prior, record(x = 0.5, y = 0.5))
`, 8);
  assert.deepEqual(compileErrors(proc), []);
  await assert.rejects(() => ctx.getMeasure('lp'),
    /ancestor 'a' depends on stochastic '(h|__anon\d+)'.*not itself one of this record's fields/);
});

test('#311 REFUSE: hidden intermediate in a chain — a -> b -> c with only `a` and `c` exposed '
  + 'as fields (b is NOT a field) — still refused (threading b would need its OBSERVED value, '
  + 'which is not available; a genuine marginalization, not a substitution)', async () => {
  const { proc, ctx } = ctxFor(`
a ~ Normal(0.0, 1.0)
b ~ Normal(a, 1.0)
c ~ Normal(b, 1.0)
prior = lawof(record(x = exp(a), z = c))
lp = logdensityof(prior, record(x = 0.5, z = 0.5))
`, 8);
  assert.deepEqual(compileErrors(proc), []);
  await assert.rejects(() => ctx.getMeasure('lp'),
    /ancestor 'c' depends on stochastic 'b'.*not itself one of this record's fields/);
});

test('#260 (d) NOT hijacked: an ALL-bare-ref hierarchical record(x = a, y = b) with '
  + 'b ~ Normal(a,1) still scores correctly via the pre-existing joint env-threading path '
  + '(the transformed-field recognition does not fire)', async () => {
  const lp = await scalarOf(`
a ~ Normal(0.0, 1.0)
b ~ Normal(a, 1.0)
prior = lawof(record(x = a, y = b))
lp = logdensityof(prior, record(x = 0.5, y = 0.5))
`, 'lp');
  // scipy oracle (independent, python MCP):
  //   norm.logpdf(0.5, 0, 1) + norm.logpdf(0.5, loc=0.5, scale=1) = -1.9628770664093453
  const oracle = -1.9628770664093453;
  assert.ok(Math.abs(lp - oracle) < TOL, `got ${lp}, oracle ${oracle}`);
});

// A genuine CYCLE in the per-field ancestor dependency graph (mutually
// dependent stochastic bindings, e.g. `a ~ Normal(b,1); b ~ Normal(a,1)`) is
// guarded defensively in `_recognizeDiagonalPushforwardFields`'s topological
// sort (a back-edge throws "ancestor dependency is cyclic"), but is NOT
// reachable through this test file via valid FlatPPL source: a mutual
// stochastic cycle already crashes analyzer.ts's `absorbedPhaseOf` phase
// computation (stack overflow) well before any binding gets a derivation —
// a pre-existing, unrelated gap (phase computation's cycle guard doesn't
// cover a mutual `~` cycle, only deterministic alias/evaluate cycles per
// self-referential-derivation.test.ts) that is out of scope for #311. The
// defensive check exists so that IF this code is ever reached with a cyclic
// per-field ancestor graph (e.g. after that phase-computation gap is fixed,
// or from some other future caller), it refuses loudly instead of infinite-
// looping or silently mis-scoring.
