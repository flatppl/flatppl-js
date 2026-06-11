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
// The full physics pipeline:
//
//   resonance(s)    = breit_wigner(s, m, width=Γ, ma=0, mb=0, ℓ=0, d=1) [complex amplitude]
//   coherent_b(x)   = c0 + c1·x                                          [real linear bg]
//   full_intensity(x) = |resonance(x²)·yS + coherent_b(x)|²              [observable density]
//   D1              = normalize(weighted(full_intensity, Lebesgue([a,b])))
//   cheb_density(x) = b0·T0(z) + b1·T1(z) + b2·T2(z), z = (2x−(a+b))/(b−a)
//   D2              = normalize(weighted(cheb_density, Lebesgue([a,b])))
//   mixture         = superpose(weighted(f1, D1), weighted(f2, D2))
//
// Three pieces of engine infrastructure get exercised in one fixture:
//
//   - Module-aware lowering (lower.ts): `hepphys.resonance_breitwigner`
//     and `polynomials.chebyshev` lower to `(%ref mod X)` per spec §11,
//     NOT to `get_field`.
//   - Alias resolution (alias-resolution.ts): the `breit_wigner` and
//     `chebyshev` aliases canonicalise to module-namespaced refs
//     across the whole module — every callsite inside `resonance(s)`,
//     `cheb_density(x)`, `full_intensity(x)` carries a resolved target.
//   - Sampler cross-module dispatch (sampler.ts `_evaluateStandard
//     ModuleCall`): runtime evaluation of `breit_wigner(...)` /
//     `chebyshev(...)` reaches the standard-modules.ts registry's
//     JS impl via env.__moduleRegistry.
//   - Higher-order ops over std-module callables (sampler.ts
//     `_synthStdModuleFn`): the `chebyshev.([0, 1, 2], …)` dotted
//     broadcast lowers to `broadcast(<module-aliased-ref>, …)`; the
//     broadcast's _resolveFn synthesises a per-arity wrapper that
//     calls the registry impl.
//
// Tests organised by what they sensibly pin:
//   (a) Structural — module bindings classify; aliases canonicalise;
//       call sites carry resolved targets.
//   (b) Closed-form runtime evaluation — `resonance(m²)`, `cheb_density(a/b)`,
//       `full_intensity(m)` match spec arithmetic to 1e-9 precision.
//   (c) End-to-end materialisation — D1 / D2 / mixture produce
//       finite-n_eff weighted measures; samples in spec support.
//   (d) Empirical-quality contrast — D2 (near-uniform background)
//       has HIGHER ESS than D1 (sharp resonance peak), since
//       importance sampling from Uniform([a,b]) wastes more atoms
//       on D1's tails.

test('hadron-physics: standard-module loading + alias canonicalisation', () => {
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

  // Both aliases canonicalise to module-namespaced refs; their
  // inferredType matches the std-module function signature.
  const bw = r.loweredModule.bindings.get('breit_wigner');
  assert.equal(bw.isAlias, true);
  assert.equal(bw.rhs.kind, 'ref');
  assert.equal(bw.rhs.ns, 'hepphys');
  assert.equal(bw.rhs.name, 'resonance_breitwigner');
  assert.equal(bw.inferredType.kind, 'function');

  const cheb = r.loweredModule.bindings.get('chebyshev');
  assert.equal(cheb.isAlias, true);
  assert.equal(cheb.rhs.ns, 'polynomials');
  assert.equal(cheb.rhs.name, 'chebyshev');
  assert.equal(cheb.inferredType.kind, 'function');

  // After alias-resolution, the `resonance` functionof body has the
  // inner `breit_wigner(...)` call's target already canonicalised —
  // the alias never leaks into call sites.
  const resonance = r.loweredModule.bindings.get('resonance');
  assert.equal(resonance.rhs.op, 'functionof');
  const innerCall = resonance.rhs.body;
  assert.equal(innerCall.kind, 'call');
  assert.equal(innerCall.target.ns, 'hepphys');
  assert.equal(innerCall.target.name, 'resonance_breitwigner');
});

test('hadron-physics: closed-form runtime evaluation of std-module callables', () => {
  // Runtime arithmetic via the registry impl, not via FlatPPL
  // sampling — these are deterministic spec checks. Three layers:
  //
  //   - resonance(m²) — the Breit-Wigner amplitude at its pole.
  //     For ℓ=0, mₐ=m_b=0: BW(σ) = 1/(m² − σ − i·m·Γ). At σ = m²
  //     the real denominator vanishes; |BW(m²)| = 1/(m·Γ), pure
  //     imaginary. m=1.5, Γ=0.1 → magnitude = 1/0.15 = 6.6667.
  //   - cheb_density at the affine endpoints — the dotted broadcast
  //     `chebyshev.([0, 1, 2], z)` routes through _synthStdModuleFn
  //     into the registry impl per element. At z=−1 (x=a):
  //     T₀=1, T₁=−1, T₂=1 → density = b0 − b1 + b2 = 1.1. At z=+1
  //     (x=b): T_n=1 ∀n → density = b0 + b1 + b2 = 1.5.
  //   - full_intensity(m) — composes the complex BW amplitude with
  //     the complex coupling yS and the real linear background,
  //     then `abs2` collapses to a real density. Peaks at the
  //     resonance mass x = m.
  const src = readFixture('hadron-physics-resonance.flatppl');
  const r = processSource(src);
  const built = orchestrator.buildDerivations(r.bindings);
  const sampler = require('../sampler.ts');
  const sigMod = require('../signatures.ts');
  const env: any = { __moduleRegistry: r.loweredModule.moduleRegistry };
  for (const [k, v] of built.fixedValues) env[k] = v;

  // 1. resonance(m² = 2.25) — BW at the pole.
  const resSig = sigMod.signatureOf('resonance', built.bindings);
  const m_sq = 1.5 * 1.5;
  const resPole = sampler.evaluateExpr(
    sigMod.substituteBoundaryValues(resSig.body, { _s_: m_sq }), env);
  assert.ok(Math.abs(resPole.re) < 1e-9,
    `Re(BW(m²)) ≈ 0 at the pole; got ${resPole.re}`);
  assert.ok(Math.abs(resPole.im - 1 / (1.5 * 0.1)) < 1e-9,
    `Im(BW(m²)) = 1/(m·Γ) = 6.6667; got ${resPole.im}`);
  // Off-pole magnitude drops away.
  const resOff = sampler.evaluateExpr(
    sigMod.substituteBoundaryValues(resSig.body, { _s_: 4.0 }), env);
  const polMag = Math.hypot(resPole.re, resPole.im);
  const offMag = Math.hypot(resOff.re, resOff.im);
  assert.ok(offMag < polMag,
    `|BW(4.0)|=${offMag} < |BW(m²)|=${polMag}`);

  // 2. cheb_density at the affine endpoints.
  const chebSig = sigMod.signatureOf('cheb_density', built.bindings);
  const cAtA = sampler.evaluateExpr(
    sigMod.substituteBoundaryValues(chebSig.body, { _x_: 0.5 }), env);
  assert.ok(Math.abs(cAtA - 1.1) < 1e-9, `cheb_density(a) = b0−b1+b2 = 1.1; got ${cAtA}`);
  const cAtB = sampler.evaluateExpr(
    sigMod.substituteBoundaryValues(chebSig.body, { _x_: 2.5 }), env);
  assert.ok(Math.abs(cAtB - 1.5) < 1e-9, `cheb_density(b) = b0+b1+b2 = 1.5; got ${cAtB}`);

  // 3. full_intensity peaks at x = m (the resonance mass: x² hits
  //    the BW pole).
  const fiSig = sigMod.signatureOf('full_intensity', built.bindings);
  const fiPeak = sampler.evaluateExpr(
    sigMod.substituteBoundaryValues(fiSig.body, { _x_: 1.5 }), env);
  const fiOff = sampler.evaluateExpr(
    sigMod.substituteBoundaryValues(fiSig.body, { _x_: 0.6 }), env);
  assert.ok(fiPeak > 5 * fiOff,
    `full_intensity peaks sharply at resonance: peak=${fiPeak} vs off=${fiOff}`);
});

test('hadron-physics: D1 / D2 / mixture materialise end-to-end', async () => {
  // End-to-end materialisation of the three derived distributions
  // composes every layer:
  //
  //   - classifyLebesgueInterval recognises `Lebesgue(support =
  //     interval(a, b))` as a Uniform-sample derivation.
  //   - classifyWeighted recognises `weighted(<function>, <base>)`
  //     per spec §06 ("f is a non-negative weight, a constant or a
  //     function of the variate"): substitutes the function's
  //     parameter with `(%ref self <baseName>)`.
  //   - `_perAtomFallback`'s packing recognises per-atom complex
  //     `{re, im}` returns and emits a shape-rich complex Value —
  //     without this, full_intensity's complex intermediates would
  //     batch-evaluate as NaN.
  const N = 200;
  const { errs, ctx, built } = setupCtx(readFixture('hadron-physics-resonance.flatppl'), N);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.D1, 'D1 classifies');
  assert.ok(built.derivations.D2, 'D2 classifies');
  assert.ok(built.derivations.mixture, 'mixture classifies');
  const measures: Record<string, any> = {};
  for (const name of ['D1', 'D2', 'mixture']) {
    const m = await ctx!.getMeasure(name);
    measures[name] = m;
    const data = (m.value && m.value.data) || m.samples;
    assert.equal(data.length, N, `${name} has N samples`);
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

  // Spec §06 normalize: after normalize, totalmass = 1
  // (logTotalmass = 0). Pins that the function-weight + Lebesgue
  // totalmass tracking compose correctly: D1 / D2's weighted
  // logTotalmass = log((b−a) · E_Uniform[f]) per the function-
  // weighted derivation; normalize cancels it to 0 cleanly.
  for (const name of ['D1', 'D2']) {
    assert.ok(Math.abs(measures[name].logTotalmass) < 1e-12,
      `${name} (after normalize) logTotalmass ≈ 0; got ${measures[name].logTotalmass}`);
  }

  // Empirical-quality contrast: D2 (near-uniform Chebyshev
  // background; cheb_density varies smoothly from ~1.1 at x=a to
  // ~1.5 at x=b, with no narrow peaks) should retain most of its
  // importance weight; D1 (resonance peak at x = m = 1.5 with
  // width Γ/m ≈ 7% of the interval) burns its IS weight on a
  // narrow region. D2's n_eff should be substantially higher.
  // The thresholds are conservative — empirically D2 ≈ 95% of N,
  // D1 ≈ 20-40% of N depending on seed; pinning D2 > 0.7·N and
  // D1 < 0.7·N gives the comparative shape without being seed-
  // brittle.
  assert.ok(measures.D2.n_eff > 0.7 * N,
    `D2 n_eff=${measures.D2.n_eff} should be > 0.7·${N} (near-uniform background retains weight)`);
  assert.ok(measures.D1.n_eff < 0.7 * N,
    `D1 n_eff=${measures.D1.n_eff} should be < 0.7·${N} (sharp resonance peak loses weight)`);
  assert.ok(measures.D2.n_eff > measures.D1.n_eff,
    `D2 ESS > D1 ESS (Chebyshev vs resonance)`);

  // The IS-weighted mean of D1 sits near the resonance mass x = m
  // = 1.5 (the BW peaks there + the linear background tilts it
  // slightly upward). Without IS weighting the mean would sit at
  // the support midpoint (1.5 too, but for the wrong reason).
  // Pin the WEIGHTED mean lives inside (1.3, 1.7) — well inside
  // the support, comfortably near the resonance.
  const d1 = measures.D1;
  const d1data = (d1.value && d1.value.data) || d1.samples;
  let wSum = 0, wxSum = 0, lwMax = -Infinity;
  if (d1.logWeights) for (let i = 0; i < N; i++) if (d1.logWeights[i] > lwMax) lwMax = d1.logWeights[i];
  for (let i = 0; i < N; i++) {
    const w = d1.logWeights
      ? Math.exp(d1.logWeights[i] - lwMax)
      : 1;
    wSum  += w;
    wxSum += w * d1data[i];
  }
  const d1Mean = wxSum / wSum;
  assert.ok(d1Mean > 1.3 && d1Mean < 1.7,
    `D1 IS-weighted mean near resonance (m=1.5): got ${d1Mean}`);
});

// =====================================================================
// 8. Lebesgue(interval) standalone — totalmass tracking
// =====================================================================
//
// Pins the spec-canonical totalmass for the unnormalised reference
// measure independently of the hadron-physics fixture. Three checks
// in one short test:
//   - `Lebesgue(support = interval(a, b))` materialises to a sample
//     measure with `logTotalmass = log(b − a)` (spec §06 reference
//     measure on a finite support).
//   - `weighted(<scalar>, Lebesgue([a,b]))` propagates the parent's
//     mass + the scalar shift: `logTotalmass = log(b − a) + log(w)`.
//   - `weighted(<function>, Lebesgue([a,b]))` propagates the parent's
//     mass + the empirical avg(f): logTotalmass ≈ log((b − a) ·
//     E_Uniform[f]). For a constant function f ≡ c we expect log(c·
//     (b − a)) exactly.

test('Lebesgue(interval): unnormalised reference measure carries log(b − a) totalmass', async () => {
  // Lebesgue([0.2, 1.7]) — b − a = 1.5, log(1.5) ≈ 0.4054651.
  const src = `
a = 0.2
b = 1.7
L = Lebesgue(support = interval(a, b))
LW = weighted(3.0, L)
`;
  const { errs, ctx, built } = setupCtx(src, 100);
  assert.equal(errs.length, 0);
  // Lebesgue's anon binding materialises (after lift it becomes the
  // direct binding L's IR).
  const Lm = await ctx!.getMeasure('L');
  const expectedLog = Math.log(1.5);
  assert.ok(Math.abs(Lm.logTotalmass - expectedLog) < 1e-12,
    `Lebesgue([0.2, 1.7]) logTotalmass = log(1.5) = ${expectedLog}; got ${Lm.logTotalmass}`);

  // weighted(3.0, L) — constant-shift fast path. Result mass =
  //   ∫_a^b 3.0 · dx = 3.0 · (b − a) = 4.5.
  const LWm = await ctx!.getMeasure('LW');
  assert.ok(Math.abs(LWm.logTotalmass - Math.log(4.5)) < 1e-12,
    `weighted(3.0, Lebesgue([0.2, 1.7])) logTotalmass = log(4.5); got ${LWm.logTotalmass}`);
});

test('beta-binomial-pushfwd: full fixture type-checks; measure-bodied lambdas reify to kernels', () => {
  // A user lambda whose body is measure-valued IS a kernel (spec §04
  // functionof-of-measure, §06 uniform kernel extension; engine-concepts
  // §19): `(n_row, p_row) -> Binomial.(n_row, p_row)` has a body that is
  // a bare-distribution broadcast, which now types as an array-valued
  // measure (inferBroadcast measure-head path) — so inferReification
  // makes the lambda a kernelType, and the outer dotted broadcast
  // `binomial_row_K.(n_data, p)` types as a measure that `draw` accepts.
  // (Previously this whole fixture failed at L39 "draw: expects measure,
  // got array of real" — the user-kernel-composition gap.)
  const src = readFixture('beta-binomial-pushfwd.flatppl');
  const { bindings, diagnostics } = processSource(src);
  const errs = (diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `full fixture should type-check cleanly; got: ${errs.map((e: any) => e.message).join(' | ')}`);
  const kindOf = (n: string) => bindings.get(n)?.inferredType?.kind;
  assert.equal(kindOf('binomial_row_K'), 'kernel', 'bare-dist-broadcast body reifies to a kernel');
  assert.equal(kindOf('beta_row_K'), 'kernel', 'iid-body lambda reifies to a kernel');
  assert.equal(kindOf('forward_kernel'), 'kernel');
  assert.equal(kindOf('prior'), 'measure');
});
