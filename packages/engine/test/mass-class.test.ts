'use strict';

// =====================================================================
// mass-class.test.ts — static total-mass class inference (spec §11)
// =====================================================================
//
// Pins `typeinfer.fillMasses` — the `%mass` class on measure types and
// the spec-§06 `normalize`-of-infinite/null static diagnostic
// (engine-concepts §17.3 normalization domain). The expected classes
// are the CROSS-ENGINE oracle: they mirror flatppl-rust's
// `flatppl-infer` golden tests (`crates/infer/tests/golden.rs`
// mass_classes_compose / weighted_fixed_scalar_mass_rules /
// joint_mass_products / normalize_of_known_infinite_mass_is_a_static_error),
// so the two engines agree at the `%meta` mass layer.
//
// Mass classes: 'normalized' | 'finite' | 'locallyfinite' | 'null' |
// 'unknown' (absent ⇒ 'deferred').

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const pirSexpr = require('../pir-sexpr.ts');

// Mass class of binding `name` after inference.
function massOf(src: string, name: string): any {
  const r = processSource(src);
  const b = r.loweredModule.bindings.get(name);
  return b && b.inferredType && b.inferredType.kind === 'measure'
    ? b.inferredType.mass : undefined;
}

// Error-severity diagnostic messages.
function errorsOf(src: string): string[] {
  const r = processSource(src);
  return (r.diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

// Annotated FlatPIR (%meta on) for a source module.
function sexprOf(src: string): string {
  const r = processSource(src);
  return pirSexpr.toSexpr(r.loweredModule, { meta: true });
}

// =====================================================================
// 1. Distributions are probability measures → normalized
// =====================================================================

test('mass: every distribution constructor is normalized', () => {
  // Scalar distributions + Dirac + the MULTIVARIATE / PROCESS constructors
  // (MvNormal / Dirichlet / Multinomial / Wishart-family / LKJ-family /
  // PoissonProcess / BinnedPoissonProcess) — all probability measures, so
  // `fillMasses` (DISTRIBUTIONS → normalized) tags them once typeinfer types
  // them as measures (the multivariates gained measure-typed signatures —
  // they were `deferred` before, the JS↔Rust %meta type-coverage gap).
  for (const [src, name] of [
    ['m = Normal(mu = 0.0, sigma = 1.0)', 'm'],
    ['m = Beta(alpha = 1.0, beta = 1.0)', 'm'],
    ['m = Poisson(rate = 3.0)', 'm'],
    ['m = Bernoulli(p = 0.5)', 'm'],
    ['m = Exponential(rate = 1.0)', 'm'],
    ['m = Dirac(value = 2.0)', 'm'],
    ['m = MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]])', 'm'],
    ['m = Dirichlet(alpha = [1.0, 1.0, 1.0])', 'm'],
    ['m = Multinomial(n = 5, p = [0.2, 0.3, 0.5])', 'm'],
    ['m = Wishart(nu = 3.0, scale = [[1.0, 0.0], [0.0, 1.0]])', 'm'],
    ['m = InverseWishart(nu = 3.0, scale = [[1.0, 0.0], [0.0, 1.0]])', 'm'],
    ['m = LKJ(n = 2, eta = 1.0)', 'm'],
    ['m = LKJCholesky(n = 2, eta = 1.0)', 'm'],
    ['m = BinnedPoissonProcess(rates = [1.0, 2.0, 3.0])', 'm'],
    ['m = PoissonProcess(intensity = weighted(5.0, Normal(0.0, 1.0)))', 'm'],
  ] as [string, string][]) {
    assert.equal(massOf(src, name), 'normalized', src);
  }
});

test('mass: multivariate/process dists carry an array measure-domain (exact vector length)', () => {
  // The measure type has an array domain (the per-atom shape). The three
  // statically-sized VECTOR dists carry the exact length (mirrors rust
  // ops.rs param_dim); matrix dists + the genuinely-ragged PoissonProcess
  // stay %dynamic. (The precise extent also rides via the valueset.)
  const cases: [string, Array<number | string>][] = [
    ['m = Dirichlet(alpha = [1.0, 1.0, 1.0])', [3]],
    ['m = MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]])', [2]],
    ['m = Multinomial(n = 5, p = [0.2, 0.3, 0.5])', [3]],
    ['m = Wishart(nu = 3.0, scale = [[1.0, 0.0], [0.0, 1.0]])', ['%dynamic', '%dynamic']],
    ['m = PoissonProcess(intensity = weighted(5.0, Normal(0.0, 1.0)))', ['%dynamic']],
  ];
  for (const [src, shape] of cases) {
    const r = processSource(src);
    const t = r.loweredModule.bindings.get('m').inferredType;
    assert.equal(t.kind, 'measure', src);
    assert.equal(t.domain.kind, 'array', src + ' domain');
    assert.deepEqual(t.domain.shape, shape, src + ' shape');
  }
});

test('mass: bayesupdate over a multivariate prior is a measure', () => {
  // Previously deferred (the prior was deferred ⇒ bayesupdate fell through);
  // with the prior measure-typed, bayesupdate(L, prior) returns the prior's
  // measure type (spec §06: unnormalized posterior over the prior's domain).
  const r = processSource(`
L = likelihoodof(fn(Normal(_, 1.0)), 0.5)
post = bayesupdate(L, Dirichlet(alpha = [1.0, 1.0, 1.0]))
`);
  const t = r.loweredModule.bindings.get('post').inferredType;
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'array');
});

// =====================================================================
// 2. Reference measures — finite on a bounded support, locally finite
//    on an unbounded one (golden: mass_classes_compose)
// =====================================================================

test('mass: Lebesgue/Counting boundedness', () => {
  assert.equal(massOf('m = Lebesgue(support = reals)', 'm'), 'locallyfinite');
  assert.equal(massOf('m = Lebesgue(support = posreals)', 'm'), 'locallyfinite');
  assert.equal(massOf('m = Lebesgue(support = unitinterval)', 'm'), 'finite');
  assert.equal(massOf('m = Lebesgue(support = interval(0.0, 1.0))', 'm'), 'finite');
  // Bounded only when BOTH bounds are finite.
  assert.equal(massOf('m = Lebesgue(support = interval(0.0, inf))', 'm'), 'locallyfinite');
  assert.equal(massOf('m = Lebesgue(support = stdsimplex(3))', 'm'), 'finite');
  assert.equal(massOf('m = Counting(support = integers)', 'm'), 'locallyfinite');
  // Parametric bound → boundedness unknown → mass unknown (conservative).
  assert.equal(massOf('hi = elementof(posreals)\nm = Lebesgue(support = interval(0.0, hi))', 'm'), 'unknown');
});

// =====================================================================
// 3. weighted / truncate / normalize / bayesupdate compose
//    (golden: mass_classes_compose + weighted_fixed_scalar_mass_rules)
// =====================================================================

test('mass: weighted by a fixed scalar rescales the class', () => {
  // %normalized demotes to %finite (the constant is no longer one);
  // %locallyfinite survives.
  assert.equal(massOf('m = weighted(2.5, Normal(mu = 0.0, sigma = 1.0))', 'm'), 'finite');
  assert.equal(massOf('m = weighted(2.5, Lebesgue(support = reals))', 'm'), 'locallyfinite');
});

test('mass: a function / parametric weight is unknown', () => {
  // A function weight (Bernstein shape) → unknown — so normalize over it
  // raises NO false error.
  assert.equal(
    massOf('c0 = elementof(reals)\nf = fn(_ * 2.0)\nm = weighted(f, Lebesgue(support = interval(0.0, 1.0)))', 'm'),
    'unknown');
  assert.equal(
    massOf('n = elementof(posreals)\nm = weighted(n, Normal(mu = 0.0, sigma = 1.0))', 'm'),
    'unknown');
});

test('mass: truncate demotes to finite; normalize restores normalized', () => {
  // truncate of a probability measure is finite (renormalization isn't
  // optional); normalize of a finite measure is a probability measure.
  assert.equal(
    massOf('m = truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, inf))', 'm'), 'finite');
  assert.equal(
    massOf('m = normalize(truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, inf)))', 'm'),
    'normalized');
  // truncate of a locally-finite measure to a BOUNDED set is finite.
  assert.equal(
    massOf('m = truncate(Lebesgue(support = reals), interval(0.0, 1.0))', 'm'), 'finite');
  // …to an unbounded set, unknown.
  assert.equal(
    massOf('m = truncate(Lebesgue(support = reals), interval(0.0, inf))', 'm'), 'unknown');
});

// NOTE: Rust's golden `mass_classes_compose` also pins bayesupdate →
// %unknown, but JS typeinfer types `bayesupdate(L, prior)` as `deferred`
// (it isn't yet a measure-producing signature here), so there is no
// `%mass` slot to fill. Covered by the typeinfer-coverage follow-up in
// TODO-flatppl-js.md, not by the mass pass.

// =====================================================================
// 4. iid, superpose, joint, pushfwd composition
// =====================================================================

test('mass: iid is a homomorphism on the class', () => {
  assert.equal(massOf('m = iid(Normal(mu = 0.0, sigma = 1.0), 5)', 'm'), 'normalized');
  assert.equal(massOf('m = iid(Lebesgue(support = reals), 5)', 'm'), 'locallyfinite');
});

test('mass: superpose adds (sum of probability measures is finite, not normalized)', () => {
  assert.equal(
    massOf('m = superpose(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 5.0, sigma = 1.0))', 'm'),
    'finite');
  // A superpose with a locally-finite component is locally finite.
  assert.equal(
    massOf('m = superpose(Normal(mu = 0.0, sigma = 1.0), Lebesgue(support = reals))', 'm'),
    'locallyfinite');
});

test('mass: a superpose whose weights provably sum to one is normalized', () => {
  // The decidable exception to "generally not normalized" (§06), checked
  // before the additive fold because it recovers a class the folded
  // component masses have already lost. Full proof coverage — both
  // readings, the f64-vs-decimal disagreements, and the identity
  // soundness controls — lives in draw-mass-gate.test.ts.
  const N0 = 'Normal(mu = 0.0, sigma = 1.0)', N1 = 'Normal(mu = 5.0, sigma = 1.0)';
  assert.equal(massOf(`m = superpose(weighted(0.3, ${N0}), weighted(0.7, ${N1}))`, 'm'),
    'normalized');
  assert.equal(massOf(`m = superpose(weighted(0.3, ${N0}), weighted(0.8, ${N1}))`, 'm'),
    'finite');
});

test('mass: a broadcast is an independent product of its cell measure', () => {
  // `D.(args…)` is one cell measure per data cell (§04 Broadcasting), so
  // the cell's class carries to the product (Rust `broadcast_mass`).
  assert.equal(massOf('mus = [1.0, 2.0]\nm = Normal.(mus, 1.0)', 'm'), 'normalized');
});

test('mass: a shape with no rule is deferred, not unknown (§11)', () => {
  // §11 separates "not yet inferred" from "unknown total mass", and the
  // draw gate consumes the difference: it rejects `unknown` and passes
  // `deferred`. A measure-`relabel` over a NAMED base has no rule in this
  // pass yet — it stood in for the chain ops here once those gained their
  // §06 mass rules. Written inline the relabel would fold into a keyword
  // `joint` and classify `normalized`, so the base stays a binding.
  const src = 'j = joint(Normal(mu = 0.0, sigma = 1.0), '
    + 'Beta(alpha = 1.0, beta = 1.0))\n'
    + 'm = relabel(j, ["a", "b"])';
  assert.equal(massOf(src, 'm'), 'deferred');
  // bayesupdate's evidence integral IS a verdict of unknown: a rule ran.
  assert.equal(massOf('mu = elementof(reals)\n'
    + 'K = kernelof(Normal(mu = mu, sigma = 1.0), mu = mu)\n'
    + 'L = likelihoodof(K, 0.5)\n'
    + 'prior = lawof(draw(Normal(mu = 0.0, sigma = 1.0)))\n'
    + 'm = bayesupdate(L, prior)', 'm'), 'unknown');
});

// =====================================================================
// Dependent composition: kchain / jointchain (spec §06)
// =====================================================================
//
// §06 "Dependent composition" defines the bind as
//   kchain:     ν(B)     = ∫ κ(a, B) dμ(a)
//   jointchain: ν(A × B) = ∫_A κ(a, B) dμ(a)
//
// Take the whole space. With every kernel Markov (κ(a, whole) = 1 for all a),
// kchain gives ν(whole) = ∫ 1 dμ = 1 when μ is a probability measure, and
// jointchain gives ν(whole) = ∫ κ(a, whole) dμ(a) = μ(whole) for ANY μ. So
// the two differ on a non-normalized base: `jointchain` carries the base's
// class, `kchain` does not pursue it (§06 calls the bind's marginalization
// integral "generally intractable"). That asymmetry is the point of these
// tests, and it is forced by the two integrals above, not copied from the
// other engine.

// A Markov step kernel, and a NON-Markov one (its output is a restriction,
// so κ(a, whole) < 1) — a legitimate reification per §04 "Reifying
// measure-valued expressions to kernels".
const MARKOV_K = 'mu = elementof(reals)\n'
  + 'K = kernelof(Normal(mu = mu, sigma = 1.0), mu = mu)\n';
const SUB_K = 'mu = elementof(reals)\n'
  + 'KT = kernelof(truncate(Normal(mu = mu, sigma = 1.0), '
  + 'interval(0.0, 1.0)), mu = mu)\n';
// A step kernel whose own output class is not yet inferred.
const DEFERRED_K = 'mu = elementof(reals)\n'
  + 'jj = joint(Normal(mu = mu, sigma = 1.0), Beta(alpha = 1.0, beta = 1.0))\n'
  + 'KD = kernelof(relabel(jj, ["a", "b"]), mu = mu)\n';
const NORM_BASE = 'Normal(mu = 0.0, sigma = 1.0)';
const FINITE_BASE = 'b = truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, 1.0))\n';

test('mass: a kchain of a probability measure and Markov kernels is normalized', () => {
  // ν(whole) = ∫ κ(a, whole) dμ(a) = ∫ 1 dμ = 1.
  assert.equal(massOf(MARKOV_K + `m = kchain(${NORM_BASE}, K)`, 'm'), 'normalized');
});

test('mass: a kchain does not pursue a non-normalized base', () => {
  // In exact maths a finite base through a Markov kernel stays finite, but
  // §06 calls the bind's marginalization integral "generally intractable",
  // so nothing weaker than the all-normalized case is claimed. `unknown` is
  // the safe direction: it is not a proof of normalization, so the draw gate
  // still refuses it.
  assert.equal(massOf(MARKOV_K + FINITE_BASE + 'm = kchain(b, K)', 'm'), 'unknown');
});

test('mass: a kchain through a NON-Markov kernel is unknown', () => {
  assert.equal(massOf(SUB_K + `m = kchain(${NORM_BASE}, KT)`, 'm'), 'unknown');
});

test('mass: a jointchain carries the BASE class through Markov kernels', () => {
  // ν(whole) = ∫ κ(a, whole) dμ(a) = μ(whole): no marginalization, so the
  // base's class survives exactly. This is where jointchain and kchain part.
  assert.equal(massOf(MARKOV_K + `m = jointchain(${NORM_BASE}, K)`, 'm'),
    'normalized');
  assert.equal(massOf(MARKOV_K + FINITE_BASE + 'm = jointchain(b, K)', 'm'),
    'finite');
  assert.equal(
    massOf(MARKOV_K + 'm = jointchain(Lebesgue(support = reals), K)', 'm'),
    'locallyfinite');
});

test('mass: a jointchain through a NON-Markov kernel is unknown', () => {
  assert.equal(massOf(SUB_K + `m = jointchain(${NORM_BASE}, KT)`, 'm'), 'unknown');
});

test('mass: the keyword form of jointchain classifies like the positional one', () => {
  // `jointchain(name = M, …)` carries its components in `fields`, so the rule
  // has to read that slot as well as `args` (§06 "Keyword form").
  assert.equal(
    massOf(MARKOV_K + `m = jointchain(base = ${NORM_BASE}, step = K)`, 'm'),
    'normalized');
});

test('mass: a chain component with no class keeps the chain deferred (§11)', () => {
  // A gap in a step kernel's own mass rules must stay "not yet inferred"
  // rather than collapsing to `unknown`, which the draw gate would reject on
  // a possibly well-formed model.
  assert.equal(massOf(DEFERRED_K + `m = kchain(${NORM_BASE}, KD)`, 'm'),
    'deferred');
  assert.equal(massOf(DEFERRED_K + `m = jointchain(${NORM_BASE}, KD)`, 'm'),
    'deferred');
});

test('mass: restrict is a sub-measure, so it is never normalized (§06)', () => {
  // §06 "Measure restriction" calls `restrict(M, x)` "the non-normalized
  // conditional measure of `M` given `x`". No rule in this pass names
  // `restrict`, and none is needed: the analyzer lowers it to
  // disintegrate + likelihoodof + `bayesupdate`, whose evidence integral is
  // `unknown` — a verdict, not a gap, so a draw from it is refused. A
  // `restrict` node never reaches the mass pass, so a rule keyed on the op
  // would be dead code.
  const src = 'prior = joint(mu = Normal(mu = 0.0, sigma = 1.0), '
    + 's = Exponential(lambda = 1.0))\n'
    + 'nu = restrict(prior, s = 0.8)';
  assert.equal(massOf(src, 'nu'), 'unknown');
  assert.equal(processSource(src).loweredModule.bindings.get('nu').rhs.op,
    'bayesupdate');
});

test('mass: joint is an independent product (golden: joint_mass_products)', () => {
  // normalized × normalized = normalized.
  assert.equal(
    massOf('j1 = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Beta(alpha = 1.0, beta = 1.0))', 'j1'),
    'normalized');
  // normalized × locallyfinite = locallyfinite.
  assert.equal(
    massOf('j2 = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Lebesgue(support = reals))', 'j2'),
    'locallyfinite');
});

test('mass: pushfwd is mass-preserving', () => {
  // LogNormal = pushfwd(exp, Normal) — still a probability measure.
  assert.equal(
    massOf('m = pushfwd(fn(exp(_)), Normal(mu = 0.0, sigma = 1.0))', 'm'), 'normalized');
});

// =====================================================================
// 5. The static diagnostic — normalize of a known-infinite / null mass
//    (golden: normalize_of_known_infinite_mass_is_a_static_error)
// =====================================================================

test('mass diagnostic: normalize of a locally-finite measure is a static error', () => {
  const errs = errorsOf('m = normalize(Lebesgue(support = reals))');
  assert.ok(errs.some((e) => /infinite total mass/.test(e)),
    'expected an infinite-mass error, got: ' + JSON.stringify(errs));
});

test('mass diagnostic: the error sees through iid / weighted', () => {
  assert.ok(errorsOf('m = normalize(iid(Lebesgue(support = reals), 5))')
    .some((e) => /infinite total mass/.test(e)));
  assert.ok(errorsOf('m = normalize(weighted(2.0, Lebesgue(support = reals)))')
    .some((e) => /infinite total mass/.test(e)));
});

test('mass diagnostic: no false positive on valid normalizations', () => {
  // The canonical mixture, the half-normal, a normalized leaf, and a
  // function-weighted (unknown-mass) shape must all pass clean.
  for (const src of [
    'a ~ Normal(mu = 0.0, sigma = 1.0)\nb ~ Normal(mu = 5.0, sigma = 1.0)\n'
      + 'M1 = lawof(a)\nM2 = lawof(b)\n'
      + 'mix = normalize(superpose(weighted(0.7, M1), weighted(0.3, M2)))',
    'm = normalize(truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, inf)))',
    'm = normalize(Normal(mu = 0.0, sigma = 1.0))',
    'lo = elementof(reals)\nhi = elementof(reals)\n'
      + 'm = normalize(weighted(fn(_ * 2.0), Lebesgue(support = interval(lo, hi))))',
  ]) {
    assert.deepEqual(errorsOf(src), [], src);
  }
});

// =====================================================================
// 6. FlatPIR `%mass` emission (spec §11), opt-in via toSexpr({meta:true})
// =====================================================================

test('mass: toSexpr emits the (%mass …) slot on measure types', () => {
  const norm = sexprOf('m = Normal(mu = 0.0, sigma = 1.0)');
  assert.ok(/\(%measure \(%domain \(%scalar real\)\) \(%mass %normalized\)\)/.test(norm),
    'normalized measure emits (%mass %normalized):\n' + norm);

  const leb = sexprOf('m = Lebesgue(support = reals)');
  assert.ok(/\(%mass %locallyfinite\)/.test(leb),
    'Lebesgue(reals) emits (%mass %locallyfinite):\n' + leb);

  // A deferred / un-inferred mass omits the slot (absent ⇒ %deferred).
  const valueOnly = sexprOf('x = 2.0');
  assert.ok(!/%mass/.test(valueOnly), 'a value binding carries no %mass:\n' + valueOnly);
});

test('mass: a kernel emits its output measure class', () => {
  // kernelof(draw, …) reifies a Markov kernel — output measure normalized.
  const out = sexprOf('mu = elementof(reals)\nx ~ Normal(mu = mu, sigma = 1.0)\n'
    + 'K = kernelof(x, mu = mu)');
  // The kernel binding K renders as (%kernel (%inputs …) (%mass %normalized)).
  assert.ok(/\(%kernel \(%inputs[^)]*\) \(%mass %normalized\)\)/.test(out),
    'kernel emits its output-measure mass:\n' + out);
});
