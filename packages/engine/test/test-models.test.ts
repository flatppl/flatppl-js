'use strict';

// =====================================================================
// test-models.test.ts — colleague-contributed Bayesian model fixtures
// =====================================================================
//
// Verifies the test-model fixtures landed by colleagues (commits
// c9baa46 + 009f372) parse, classify, and materialise where the
// engine supports the patterns involved. Each fixture targets a
// distinct PPL surface combination:
//
//   - zero-inflated-binomial.flatppl — `superpose` mixture with a
//     point-mass `Dirac` component; iid observation block. Exercises
//     the cascade-prune walker's branch-aware ref collection (the
//     `superpose` IR carries per-branch measures in a `branches`
//     field, not args/kwargs — see `ir-shared.collectSelfRefs`).
//
//   - normal-mixture.flatppl — `normalize(superpose(weighted(theta,
//     Normal(mu[1], 1)), weighted(1-theta, Normal(mu[2], 1))))` over
//     iid observations. Same `branches`-walk requirement plus
//     `mu[1]` / `mu[2]` selector indexing inside a forward kernel.
//
//   - horseshoe.flatppl — Horseshoe prior on regression coefficients
//     using `normalize(truncate(Cauchy(0, 1), interval(0, inf)))`
//     and `iid(half_cauchy, D_X)`. Exercises matIid's resolution
//     through normalize / truncate (the worker has a separate
//     `truncateSampleN` path; matIid peels normalize as a sample-
//     preserving no-op and routes truncate to truncateSampleN with
//     count = N × k).
//
//   - beta-binomial-pushfwd.flatppl — Hierarchical Beta-Binomial
//     with Pareto prior via `pushfwd(fn(0.1 * exp(_)),
//     Exponential(1.5))`. Tests pushfwd typeinfer signature (added
//     to types.ts since the inferGenericCall fallback returned
//     `deferred()` for unrecognised ops, leaving downstream
//     `iid(pareto, G)` to fail with "expects measure, got
//     deferred"). The model also chains user-fn-returning-measure
//     broadcasts (`p ~ beta_row_K.(a, b)`, `r ~ binomial_row_K.
//     (n_data, p)`) — those hit a deeper typeinfer gap that's a
//     follow-up (see TODO-flatppl-js fusion (b) / typeinfer).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function readFixture(name: string): string {
  return fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf-8');
}

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const errs = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) {
    return { errs, ctx: null, built: null };
  }
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    // Module registry: alias → (stdName, stdCompat) for cross-
    // module call dispatch. Populated by pir.lowerToModule from
    // every `standard_module(name, compat)` / `load_module(...)`
    // binding. Threaded into the worker's session env via
    // pushFixedEnv (materialiser-shared.ts) so sampler's cross-
    // module call dispatch (`_evaluateStandardModuleCall`) can
    // resolve `(call target=({ns: <alias>, name: X}) ...)` to
    // the registry's JS impl.
    moduleRegistry: lifted.loweredModule.moduleRegistry || null,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
    sampleCount: N,
    rootSeed: 42,
  };
  return { errs: [], ctx, built };
}

// =====================================================================
// 1. Zero-inflated Binomial — exercises the `branches` walk fix
// =====================================================================

test('zero-inflated-binomial: classifies + materialises', async () => {
  const { errs, ctx, built } = setupCtx(readFixture('zero-inflated-binomial.flatppl'), 50);
  assert.equal(errs.length, 0, 'no parse/typeinfer errors');
  assert.ok(built.derivations.posterior, 'posterior classifies');
  assert.equal(built.derivations.posterior.kind, 'bayesupdate');
  // Materialising posterior exercises the bayesupdate density walk,
  // which in turn needs the forward_kernel body's refs (`K`, `p`,
  // `psi`) collected via collectSelfRefs. Pre-fix, `K` lived inside
  // a `superpose` branch and got missed → "unbound self reference
  // 'K'" at eval time.
  const m = await ctx!.getMeasure('posterior');
  assert.ok(m, 'posterior materialises');
  assert.ok(typeof m.n_eff === 'number' && Number.isFinite(m.n_eff),
    `posterior n_eff is finite (got ${m.n_eff})`);
});

// =====================================================================
// 2. Normal mixture — `normalize(superpose(weighted, weighted))` +
//    `iid` + per-atom `mu[1]` / `mu[2]` indexing inside the kernel
// =====================================================================

test('normal-mixture: classifies + materialises', async () => {
  const { errs, ctx, built } = setupCtx(readFixture('normal-mixture.flatppl'), 50);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.posterior);
  assert.equal(built.derivations.posterior.kind, 'bayesupdate');
  const m = await ctx!.getMeasure('posterior');
  assert.ok(m, 'posterior materialises');
  assert.ok(typeof m.n_eff === 'number' && Number.isFinite(m.n_eff),
    `posterior n_eff is finite (got ${m.n_eff})`);
});

// =====================================================================
// 3. Horseshoe regression — matIid through normalize / truncate
// =====================================================================

test('horseshoe: classifies + iid(half_cauchy) materialises', async () => {
  const { errs, ctx, built } = setupCtx(readFixture('horseshoe.flatppl'), 100);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.posterior, 'posterior classifies');

  // Pre-fix: matIid called leafSampleIR which only peels through
  // 'alias'; half_cauchy = normalize(truncate(Cauchy(0,1),
  // interval(0,inf))) → leafSampleIR returned null and matIid threw
  // "cannot resolve leaf sample IR for half_cauchy". Post-fix:
  // matIid peels normalize + routes truncate through
  // truncateSampleN with count = N × k. Verify lambdas (the iid'd
  // half-Cauchy) materialises cleanly to [N, D_X] without NaN.
  const lambdas = await ctx!.getMeasure('lambdas');
  assert.deepEqual(lambdas.value.shape, [100, 3]);
  let anyNaN = false;
  for (let i = 0; i < lambdas.value.data.length; i++) {
    if (Number.isNaN(lambdas.value.data[i])) { anyNaN = true; break; }
  }
  assert.equal(anyNaN, false, 'no NaN in lambdas samples');
  // All lambdas samples must be positive (half-Cauchy support).
  let allPositive = true;
  for (let i = 0; i < lambdas.value.data.length; i++) {
    if (lambdas.value.data[i] < 0) { allPositive = false; break; }
  }
  assert.equal(allPositive, true, 'all lambdas samples are positive');

  // Note: posterior materialises to a measure record but the
  // bayesupdate's likelihood step currently produces NaN n_eff
  // (likely a separate engine issue in how `X * betas` matvec with
  // per-atom betas threads through density eval — a follow-up).
  // Pin the prior parts work; leave posterior n_eff unchecked.
});

// =====================================================================
// 4. Beta-Binomial with pushfwd-derived Pareto — partial coverage
// =====================================================================
//
// Tests pushfwd's typeinfer signature is correct enough for the
// simple "iid of pushfwd-derived measure" composition. Full
// hierarchical end-to-end is gated on the user-kernel-composition
// typeinfer gap (`binomial_row_K = (n, p) -> Binomial.(n, p)` whose
// return type isn't currently inferred as a measure when called via
// dotted broadcast). Tracked as a Phase F sub-follow-up.

test('beta-binomial-pushfwd (partial): pushfwd typeinfer + iid resolves', () => {
  // Strip the kernel-composition chunk to isolate the pushfwd
  // typeinfer signature. Verify the simple `iid(pareto, G)` shape
  // (which pre-fix failed "iid: arg 1 expects measure, got
  // deferred") classifies cleanly.
  const r = processSource(`
G = 2
pareto = pushfwd(fn(0.1 * exp(_)), Exponential(1.5))
a_plus_b ~ iid(pareto, G)
`);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `pushfwd typeinfer: expected no errors, got ${JSON.stringify(errs)}`);
  const built = orchestrator.buildDerivations(r.bindings);
  assert.ok(built.derivations.a_plus_b, 'iid(pareto, G) classifies');
  assert.ok(built.bindings.get('pareto')?.inferredType?.kind === 'measure',
    'pareto inferredType.kind === measure (was deferred pre-fix)');
});

// =====================================================================
// 5. Beverton-Holt stock-recruitment (steepness parametrisation)
// =====================================================================
//
// Lognormal-observation Bayesian regression for fisheries
// recruitment data. Uses pushfwd-derived priors (uniform on
// (0.2, 1) for steepness h), normalize(truncate(...)) for
// positive-support priors (half-Cauchy on σ; truncated-Normal on
// α), and a deterministic transform chaining the priors into
// `log_rhat` — a per-observation linear predictor that ends up as
// the lognormal mean.
//
// What this exercises beyond the others: a fixed-phase scalar
// constant (`max_r`) is closed-over deep inside a stochastic prior
// (`alpha ~ normalize(truncate(Normal(2 * max_r, …), …))`).
// Without the `collectRefArrays` fix that auto-pushes fixed refs
// into the worker session env, materialising `alpha` fails with
// "unbound self reference 'max_r'" — even though `max_r` is in
// `fixedValues`. (The legacy callers had to remember to push
// fixedEnv themselves; this fixture catches the gap end-to-end.)

test('beverton-holt: classifies + materialises (collectRefArrays auto-push fix)', async () => {
  const { errs, ctx, built } = setupCtx(readFixture('beverton-holt.flatppl'), 50);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.posterior);
  assert.equal(built.derivations.posterior.kind, 'bayesupdate');
  // alpha goes through normalize(truncate(Normal(2 * max_r, ...))).
  // Pre-fix: fails with "unbound self reference 'max_r'" because
  // collectRefArrays filtered out fixed refs without pushing them
  // into the worker session env.
  const alpha = await ctx!.getMeasure('alpha');
  assert.ok(alpha && alpha.samples,
    'alpha materialises (max_r is auto-pushed via setEnv merge)');
  // All alpha samples must be positive (truncate to (0, inf)).
  for (let i = 0; i < alpha.samples.length; i++) {
    assert.ok(alpha.samples[i] > 0,
      `alpha sample ${i} = ${alpha.samples[i]} must be > 0`);
  }
  // Posterior materialises end-to-end (low n_eff is expected for a
  // single MC step against informative-ish data — not an engine
  // bug; just the model's nature with a small sample count).
  const m = await ctx!.getMeasure('posterior');
  assert.ok(m, 'posterior materialises');
  assert.ok(typeof m.n_eff === 'number' && Number.isFinite(m.n_eff),
    `posterior n_eff is finite (got ${m.n_eff})`);
});

// =====================================================================
// 6. 2PL Item Response Theory (Rasch with discrimination)
// =====================================================================
//
// Educational psychometrics workhorse: per-person ability, per-item
// difficulty + discrimination, hierarchical hyperpriors on the
// difficulty mean / sd and the log-discrimination sd. The forward
// model is
//
//   P(correct[p, i]) = invlogit(discrim[i] · (ability[p] − diff[i] − diff_mean))
//
// fitted against a flat list of (person_idx, item_idx, correct) rows.
//
// Why this fixture exercises a lot of the engine surface in one shot:
//
//   - Hierarchical priors with PARAMETER-DEPENDENT sigma:
//     `diff ~ iid(Normal(0, diff_sd), n_item)` — diff_sd is itself a
//     stochastic draw. Per-atom binding of refArrays must thread
//     diff_sd[i] to all n_item inner draws sharing that atom.
//   - LogNormal scale prior on discrimination.
//   - Half-Cauchy via `normalize(truncate(Cauchy(0, 5),
//     interval(0, inf)))` — matIid's normalize + truncate peel
//     route through truncateSampleN.
//   - Gather indexing with integer index arrays:
//     `ability[person_idx]`, `diff[item_idx]`, `discrim[item_idx]`
//     — `get(arr, int_arr)` produces a length-12 vector per atom.
//   - Dotted broadcast of a distribution constructor:
//     `Bernoulli.(invlogit.(...))` — kernel-broadcast over an inline
//     elementwise computation.
//   - Record-typed kernel output: `kernelof(record(correct = …),
//     …kw)` with six inputs of mixed scalar/array shape.
//   - bayesupdate density walk over the full structure.
//
// The regression test checks each layer rather than just "no
// errors": every refactor on the dev branch (ir-walk centralisation,
// matIid leaf-preserving set, collectRefArrays auto-push,
// inferArith1 type-preserving, inferVector scalar promotion) is
// exercised by this fixture, so a future regression in any of them
// trips a specific assertion below.

test('rasch-two-parameter: hierarchical IRT classifies + materialises end-to-end', async () => {
  const { errs, ctx, built } = setupCtx(readFixture('rasch-two-parameter.flatppl'), 500);
  assert.equal(errs.length, 0, 'no parse/typeinfer errors');
  assert.ok(built.derivations.posterior, 'posterior classifies');
  assert.equal(built.derivations.posterior.kind, 'bayesupdate');

  // Half-Cauchy hyperprior — all positive, no NaN.
  const diff_sd = await ctx!.getMeasure('diff_sd');
  const diff_sd_data = diff_sd.value?.data || diff_sd.samples;
  for (let i = 0; i < diff_sd_data.length; i++) {
    assert.ok(diff_sd_data[i] > 0, `diff_sd[${i}] = ${diff_sd_data[i]} must be positive (half-Cauchy)`);
    assert.ok(!Number.isNaN(diff_sd_data[i]), `diff_sd[${i}] is NaN`);
  }

  // Hierarchical iid: diff ~ iid(Normal(0, diff_sd), n_item). Per-atom
  // diff_sd[i] should drive std(diff[i, :]). Pre-fix to matIid's
  // _resolveIidLeaf, this case fell back to the composite-inner path
  // which broke the per-atom param binding; post-fix it routes
  // through the leaf-sample worker primitive with refArrays carrying
  // diff_sd, and per-atom correlation is strong.
  const diff = await ctx!.getMeasure('diff');
  assert.deepEqual(diff.value.shape, [500, 4]);
  const dd = diff.value.data;
  let mu_sd = 0, mu_std = 0;
  const stds = new Float64Array(500);
  for (let i = 0; i < 500; i++) {
    let s = 0, ss = 0;
    for (let j = 0; j < 4; j++) { const v = dd[i*4 + j]; s += v; ss += v*v; }
    const m = s / 4;
    stds[i] = Math.sqrt(Math.max(0, ss / 4 - m*m));
    mu_sd += diff_sd_data[i];
    mu_std += stds[i];
  }
  mu_sd /= 500; mu_std /= 500;
  let cov = 0, vsd = 0, vstd = 0;
  for (let i = 0; i < 500; i++) {
    const a = diff_sd_data[i] - mu_sd, b = stds[i] - mu_std;
    cov += a*b; vsd += a*a; vstd += b*b;
  }
  const r_per_atom = cov / Math.sqrt(vsd * vstd);
  assert.ok(r_per_atom > 0.8,
    `per-atom param binding: corr(diff_sd, std(diff)) = ${r_per_atom.toFixed(3)} (must be > 0.8)`);

  // LogNormal scale: support is mathematically (0, ∞). Under
  // float-arithmetic this becomes [0, +∞]: extreme `discrim_log_sd`
  // values from the half-Cauchy(0, 5) hyperprior produce
  // `exp(Normal(0, σ))` that overflows to +∞ or underflows to 0.
  // Both are legitimate float-truncated LogNormal samples (the spec
  // and the runtime both treat them as the float endpoints of the
  // support). The invariant is "no NaN, no negative".
  const discrim = await ctx!.getMeasure('discrim');
  assert.deepEqual(discrim.value.shape, [500, 4]);
  for (let i = 0; i < discrim.value.data.length; i++) {
    const v = discrim.value.data[i];
    assert.ok(!Number.isNaN(v), `discrim[${i}] is NaN`);
    assert.ok(v >= 0, `discrim[${i}] = ${v} must be ≥ 0 (LogNormal)`);
  }

  // Gather indexing + dotted broadcast of Bernoulli: `correct` has
  // length 12 (n_obs) and per-element mean ≈ 0.5 (invlogit of
  // zero-centered prior means). Verifies that
  // `ability[person_idx]` etc. produce length-12 per-atom vectors
  // and `Bernoulli.(invlogit.(…))` produces a length-12 product
  // measure.
  const correct = await ctx!.getMeasure('correct');
  assert.deepEqual(correct.value.shape, [500, 12]);
  let total_zeros = 0;
  for (let i = 0; i < correct.value.data.length; i++) {
    const v = correct.value.data[i];
    assert.ok(v === 0 || v === 1, `correct[${i}] = ${v} must be 0 or 1`);
    if (v === 0) total_zeros++;
  }
  const zeros_frac = total_zeros / correct.value.data.length;
  assert.ok(zeros_frac > 0.4 && zeros_frac < 0.6,
    `prior predictive P(correct = 0) ≈ 0.5; got ${zeros_frac.toFixed(3)}`);

  // bayesupdate density walk produces a posterior with finite
  // logTotalmass + n_eff. The ESS will be small (12 binary obs vs a
  // 14-parameter posterior sampled from the prior — importance
  // sampling shows its limits here; that's a known property, not an
  // engine bug). What matters is that density evaluation
  // SUCCEEDS — pre-collectRefArrays-fix this would have failed with
  // unbound-ref errors deep inside the bayesupdate density walk
  // over the kernel-broadcast body.
  const m = await ctx!.getMeasure('posterior');
  assert.ok(m, 'posterior materialises');
  assert.ok(typeof m.n_eff === 'number' && Number.isFinite(m.n_eff),
    `posterior n_eff is finite (got ${m.n_eff})`);
  assert.ok(Number.isFinite(m.logTotalmass),
    `posterior logTotalmass is finite (got ${m.logTotalmass})`);
  // logWeights must NOT be all -∞ or NaN (would indicate density-eval
  // collapsed everywhere).
  let finite_count = 0;
  if (m.logWeights) {
    for (const w of m.logWeights) if (Number.isFinite(w)) finite_count++;
  } else {
    finite_count = 500; // null logWeights ⇒ uniform.
  }
  assert.ok(finite_count > 0, `posterior has at least one finite-weight atom (got ${finite_count}/500)`);

  // Composite-measure invariant (empirical.ts §"composite-measure
  // invariant"): the IS weights live at the OUTER level alone;
  // sub-fields are SoA storage with null logWeights / no n_eff.
  // Pre-fix (empirical.recordMeasure transparent): the prior's
  // half-Cauchy normalize weights survived on
  // `posterior.fields.diff_sd.logWeights`, stale relative to the
  // bayesupdate output. Post-fix: every field's metadata is
  // cleared on composite construction.
  for (const fname of ['ability', 'diff', 'discrim', 'diff_mean', 'diff_sd', 'discrim_log_sd']) {
    const f = m.fields[fname];
    assert.equal(f.logWeights, null,
      `posterior.fields.${fname}.logWeights cleared on composite construction`);
    assert.equal(f.n_eff, undefined,
      `posterior.fields.${fname}.n_eff cleared on composite construction`);
  }
});

// =====================================================================
// 7. Hadron-physics resonance fixture — exercises standard-module
//    loading + cross-module call dispatch end-to-end
// =====================================================================
//
// A particle-physics resonance amplitude composed from
// `particle-physics.resonance_breitwigner` (complex Breit-Wigner with
// mass-dependent width + Blatt-Weisskopf form factors) and a Chebyshev-
// series background from `polynomials.chebyshev`. Three pieces of
// infrastructure get exercised in one fixture:
//
//   - Module-aware lowering (lower.ts): `hepphys.resonance_breitwigner`
//     and `polynomials.chebyshev` lower to `(%ref mod X)` per spec §11,
//     NOT to `get_field`.
//   - Alias resolution (alias-resolution.ts): the `breit_wigner =
//     hepphys.resonance_breitwigner` and `chebyshev = polynomials
//     .chebyshev` aliases canonicalise to module-namespaced refs
//     across the whole module — call sites inside `resonance(s)`,
//     `bw_scaled(x)`, `cheb_density(x)` all carry resolved targets.
//   - Sampler cross-module dispatch (sampler.ts `_evaluateStandard
//     ModuleCall`): runtime evaluation of `breit_wigner(...)` /
//     `chebyshev(...)` reaches the standard-modules.ts registry's
//     JS impl via env.__moduleRegistry.
//   - Higher-order ops over std-module callables (sampler.ts
//     `_synthStdModuleFn`): the `chebyshev.([0, 1, 2], …)` dotted
//     broadcast lowers to `broadcast(<module-aliased-ref>, …)`; the
//     broadcast's _resolveFn synthesises a per-arity wrapper that
//     calls the registry impl.

test('hadron-physics: standard-module loading + cross-module call dispatch', () => {
  const src = readFixture('hadron-physics-resonance.flatppl');
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `no parse/typeinfer errors: ${JSON.stringify(errs)}`);

  // The two module-typed bindings classify as `module` and their
  // moduleRegistry entries point at the right (stdName, stdCompat).
  assert.equal(r.bindings.get('hepphys')?.type, 'module');
  assert.equal(r.bindings.get('polynomials')?.type, 'module');
  const reg = r.loweredModule.moduleRegistry;
  assert.equal(reg.hepphys.stdName, 'particle-physics');
  assert.equal(reg.hepphys.stdCompat, '0.1');
  assert.equal(reg.polynomials.stdName, 'polynomials');
  assert.equal(reg.polynomials.stdCompat, '0.1');

  // The alias `breit_wigner` canonicalises to a module-namespaced
  // ref; its inferredType is the std-module's function signature.
  const bw = r.loweredModule.bindings.get('breit_wigner');
  assert.equal(bw.isAlias, true);
  assert.equal(bw.rhs.kind, 'ref');
  assert.equal(bw.rhs.ns, 'hepphys');
  assert.equal(bw.rhs.name, 'resonance_breitwigner');
  assert.equal(bw.inferredType.kind, 'function');

  // After alias-resolution, `bw_scaled`'s functionof body has its
  // inner call's target already canonicalised. Walking the IR
  // confirms the alias never leaks into the body.
  const bwScaled = r.loweredModule.bindings.get('bw_scaled');
  assert.equal(bwScaled.rhs.op, 'functionof');
  const innerCall = bwScaled.rhs.body.args[1];  // 10 * <breit_wigner_call>
  assert.equal(innerCall.kind, 'call');
  assert.equal(innerCall.target.ns, 'hepphys');
  assert.equal(innerCall.target.name, 'resonance_breitwigner');
});

test('hadron-physics: D1 / D2 / mixture materialise end-to-end', async () => {
  // The full pipeline:
  //   D1 = normalize(weighted(full_intensity, Lebesgue(interval(a, b))))
  //   D2 = normalize(weighted(cheb_density, Lebesgue(interval(a, b))))
  //   mixture = superpose(weighted(f1, D1), weighted(f2, D2))
  //
  // Three engine extensions land together to make this work:
  //   - classifyLebesgueInterval recognises `Lebesgue(support =
  //     interval(a, b))` as a Uniform-sample derivation. The
  //     unnormalised totalmass (= b − a) is dropped — the outer
  //     `normalize` discards it anyway.
  //   - classifyWeighted recognises `weighted(<function>, <base>)`
  //     per spec §06 ("f is a non-negative weight, a constant or a
  //     function of the variate"): substitutes the function's
  //     parameter with `(%ref self <baseName>)`, producing a
  //     weightIR the existing materialiser path evaluates per atom.
  //   - `_perAtomFallback`'s packing recognises per-atom complex
  //     `{re, im}` returns and emits a shape-rich complex Value
  //     (shape=[N], dtype='complex'). Without this, the batched
  //     evaluator returned NaN for IRs that include complex-valued
  //     intermediates (full_intensity composes complex Breit-Wigner
  //     amplitude × complex coupling + real background before
  //     `abs2` collapses to real).
  const { errs, ctx, built } = setupCtx(readFixture('hadron-physics-resonance.flatppl'), 200);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.D1, 'D1 classifies');
  assert.ok(built.derivations.D2, 'D2 classifies');
  assert.ok(built.derivations.mixture, 'mixture classifies');
  for (const name of ['D1', 'D2', 'mixture']) {
    const m = await ctx!.getMeasure(name);
    const data = (m.value && m.value.data) || m.samples;
    assert.equal(data.length, 200, `${name} has N samples`);
    let mn = Infinity, mx = -Infinity;
    for (const v of data) {
      assert.ok(!Number.isNaN(v), `${name} sample is NaN`);
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    // a = 0.5, b = 2.5 — every sample sits in the spec support.
    assert.ok(mn >= 0.5 - 1e-9, `${name} min ${mn} ≥ a=0.5`);
    assert.ok(mx <= 2.5 + 1e-9, `${name} max ${mx} ≤ b=2.5`);
    assert.ok(Number.isFinite(m.n_eff), `${name} n_eff finite (got ${m.n_eff})`);
  }
});

test('hadron-physics: bw_scaled / cheb_density / full_intensity evaluate to spec values', () => {
  const src = readFixture('hadron-physics-resonance.flatppl');
  const r = processSource(src);
  const built = orchestrator.buildDerivations(r.bindings);
  const sampler = require('../sampler.ts');
  const sigMod = require('../signatures.ts');
  const env: any = { __moduleRegistry: r.loweredModule.moduleRegistry };
  for (const [k, v] of built.fixedValues) env[k] = v;

  // 1. bw_scaled(x) = 10 · BW(x, m=1.1, width=0.2, ℓ=0, m_a=m_b=0)
  //    At ℓ=0, m_a=m_b=0: BW(σ) = 1/(m² − σ − iΓm) (spec §09 closing
  //    note). At σ = m² = 1.21 the real denominator vanishes; |BW|
  //    peaks at 1/(Γ·m) = 1/(0.2·1.1) = 4.545; the scaled value is 10·.
  const sig = sigMod.signatureOf('bw_scaled', built.bindings);
  const peakBody = sigMod.substituteLocals(sig.body, { _x_: 1.21 });
  const peakBW = sampler.evaluateExpr(peakBody, env);
  // pure imaginary at resonance — real denominator is exactly 0
  assert.ok(Math.abs(peakBW.re) < 1e-10, `BW(m²) real ≈ 0; got ${peakBW.re}`);
  assert.ok(Math.abs(peakBW.im - 10 / (0.2 * 1.1)) < 1e-6,
    `BW(m²) imag = 10/(Γ·m) = 45.4545; got ${peakBW.im}`);

  // Off-resonance: |BW| drops away monotonically (in this simple
  // ℓ=0 case).
  const offBody = sigMod.substituteLocals(sig.body, { _x_: 3.0 });
  const offBW = sampler.evaluateExpr(offBody, env);
  const peakMag = Math.hypot(peakBW.re, peakBW.im);
  const offMag = Math.hypot(offBW.re, offBW.im);
  assert.ok(offMag < peakMag,
    `off-resonance magnitude must be smaller; |BW(3.0)|=${offMag}, |BW(m²)|=${peakMag}`);

  // 2. cheb_density(x) — uses `chebyshev.([0, 1, 2], affine_x)` which
  //    routes the broadcast through `_synthStdModuleFn`. Coefficients
  //    are b0=1.1, b1=0.2, b2=0.2 on the spec basis a=0.5, b=2.5.
  //    affine_x(2.5) = 1 → T₀=T₁=T₂=1 → density = 1.1+0.2+0.2 = 1.5.
  //    affine_x(0.5) = -1 → T₀=1,T₁=-1,T₂=1 → density = 1.1-0.2+0.2 = 1.1.
  const chebSig = sigMod.signatureOf('cheb_density', built.bindings);
  const cAt05 = sampler.evaluateExpr(
    sigMod.substituteLocals(chebSig.body, { _x_: 0.5 }), env);
  assert.ok(Math.abs(cAt05 - 1.1) < 1e-9, `cheb_density(0.5) ≈ 1.1; got ${cAt05}`);
  const cAt25 = sampler.evaluateExpr(
    sigMod.substituteLocals(chebSig.body, { _x_: 2.5 }), env);
  assert.ok(Math.abs(cAt25 - 1.5) < 1e-9, `cheb_density(2.5) ≈ 1.5; got ${cAt25}`);

  // 3. full_intensity(x) — composes resonance + coherent background
  //    + complex coupling. Peaks at x = m = √(m²) = 1.5 (resonance
  //    mass) since x² = σ enters BW(σ); off-peak < peak.
  const fiSig = sigMod.signatureOf('full_intensity', built.bindings);
  const fiPeak = sampler.evaluateExpr(
    sigMod.substituteLocals(fiSig.body, { _x_: 1.5 }), env);
  const fiOff = sampler.evaluateExpr(
    sigMod.substituteLocals(fiSig.body, { _x_: 3.0 }), env);
  assert.ok(fiPeak > fiOff,
    `full_intensity peaks at resonance: fiPeak=${fiPeak}, fiOff=${fiOff}`);
});

test('beta-binomial-pushfwd: full fixture exposes user-kernel-composition gap', () => {
  // Pin the failure mode of the full fixture so a regression in
  // either direction (fix lands → catch flips green; gap reopens
  // → assertion fails) is detected. The deeper gap: a user fn
  // returning `Binomial.(n_row, p_row)` (kernel-broadcast) doesn't
  // typeinfer as a measure when the user fn is called via dotted
  // broadcast (`binomial_row_K.(n_data, p)`) — the outer broadcast
  // sees `array of real` instead of `array of measure`.
  const src = readFixture('beta-binomial-pushfwd.flatppl');
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  // Pre-fix: would fail at L27 (iid: expects measure, got deferred).
  // Post-fix: pushfwd typeinfer now produces measure(real); first
  // failure moves to L39 (draw: expects measure, got array of
  // real) — the user-kernel-composition gap.
  assert.ok(errs.length > 0, 'still has remaining typeinfer errors');
  assert.match(errs[0].message,
    /draw|measure|user.*kernel|broadcast|expected/i,
    `first error is the user-kernel-composition gap: ${errs[0]?.message}`);
});
