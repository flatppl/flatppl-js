'use strict';

// weighted(w, M) with a FUNCTION-of-variate weight — spec §06 "Density of
// composed measures": logdensityof(weighted(w, M), x) = log w(x) +
// logdensityof(M, x), w a constant OR a function of the variate. Buffy
// #307: the engine dropped w entirely when it was a function (named-ref OR
// inline lambda) — the density path (numerator, mat-density's normalize-Z
// materialisation, and the §12-style symbolic totalmass fold) all treated
// `weighted(fn, M)` as if the weight were absent, so `normalize(weighted(fn,
// M))` came out flat/uniform regardless of fn. Root causes (two entangled
// fronts) fixed here:
//
// FRONT 1 — the function weight itself was dropped:
//   - derivations.ts `_classifyWeightedByFunction` already substituted a
//     function-of-variate weight's parameter with a self-ref to the BASE
//     measure's own name (both the named-fn-ref and inline-lambda forms end
//     up at this substitution) — but `expandMeasureIR`'s 'weighted' case
//     re-embedded that self-ref VERBATIM, and `_expandByName`'s 'sample' case
//     for a bounded Lebesgue (lowered to a synthetic Uniform sample + a
//     logTotalmass=log(b−a) annotation) dropped that annotation when
//     inlining the distIR for density-walking. density.ts's walkWeighted
//     only recognised an INLINE `functionof` weight (never a self-ref), so
//     it fell through to the atom-scalar path and resolved the self-ref via
//     getMeasure — re-materialising an UNRELATED sample of the base instead
//     of reading the point actually being scored.
//   - lift.ts's `containsHoleOrPlaceholder` bailed on lifting
//     `weighted(<inline lambda>, M)` to its own anon binding because the
//     lambda's OWN placeholder looked like an escaping hole — blocking
//     classification of the inline form entirely ("no derivation").
//   - normalize-mass.ts's `totalMassExpr` treated ANY `weighted(w, M)` as a
//     mixture branch with a scalar mass factor `w · mass(M)`, which breaks
//     once `w` is a real `functionof` node (mass(weighted(fn,M)) = ∫f dM,
//     not a pointwise value).
//   Fixed: `expandMeasureIR`'s 'weighted' case re-wraps a weight IR that
//   still self-refs its base into a `functionof` (density.ts's EXISTING
//   inline-lambda branch then binds it correctly); `_expandByName`'s
//   'sample' case restores the dropped logTotalmass as a `logweighted`
//   shift (parity with mat-density's matWeighted, which already reads it);
//   `containsHoleOrPlaceholder` treats a `fn`/`functionof`/`kernelof`
//   CallExpr as a closed, opaque reifier (its own holes never escape it);
//   `totalMassExpr` bails (null) on a `functionof`-shaped weight rather than
//   misusing it as a scalar factor.
//
// FRONT 2 — the complex resonance weight NaN'd once FRONT 1 stopped hiding
// it: density.ts's `applyAtomScalar` assumed a weight's batched evaluation
// result was either a plain number/boolean or a bare indexable
// Float64Array/Array. A weight touching COMPLEX arithmetic (`abs2(<complex
// expr>)`, the particle-physics resonance amplitude here) evaluates through
// the shape-tagged Value path instead — `{shape, data: Float64Array}` — so
// `result[i]` silently read `undefined`, `+undefined` was NaN, and
// `addLogW` collapsed every atom to −∞ (later masked as a flat/zero
// density by the SAME entangled front-1 bugs above). Fixed: unwrap a
// shape-tagged Value result (broadcasting a single-element Value across N
// atoms when the weight had no per-atom-varying ref — e.g. every occurrence
// of the variate was substituted with the SAME consumed point).
//
// ORACLES — every density assertion below is checked against an
// INDEPENDENTLY DERIVED value (closed-form or scipy numeric quadrature via
// the python MCP), never against this engine's own prior output. See each
// test for its derivation. maths > spec > code.

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');
const fs = require('fs');
const path = require('path');

function buildCtx(src: string, N: number, seed = 3) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: seed, rootSeed: seed, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

// `x` labels for generated binding names — FlatPPL identifiers can't start
// with a digit or contain '.', so "0.25" → "x0p25".
function xLabel(x: number): string {
  return 'x' + String(x).replace('.', 'p').replace('-', 'm');
}

// Build a `logdensityof(<measureName>, x)` source suffix with one named
// binding per point, so a SINGLE buildCtx scores every point — the
// Z-bearing measure they all reference materialises ONCE (memoised by
// ctx.getMeasure) and every point reuses that SAME MC estimate, instead of
// re-paying the N-atom Z materialisation cost once per point.
function ldBindings(measureName: string, xs: number[]): string {
  return xs.map((x) => `ld_${xLabel(x)} = logdensityof(${measureName}, ${x})\n`).join('');
}

// Score every point of `ldBindings(measureName, xs)` from `ctx`, returning
// { x → logdensity }.
async function scoreAll(ctx: any, xs: number[]): Promise<Record<string, number>> {
  const out: Record<string, number> = {};
  for (const x of xs) {
    const m = await ctx.getMeasure('ld_' + xLabel(x));
    out[String(x)] = m.samples[0];
  }
  return out;
}

// ---------------------------------------------------------------------
// FRONT 1 — triangular density: normalize(weighted(x -> x, Lebesgue(0,1)))
//
// ORACLE (closed form): weighted(x, Lebesgue) has density f(x) = x w.r.t.
// Lebesgue on [0,1]; Z = ∫_0^1 x dx = 1/2, so the normalized density is
// 2x. pdf(0.25)=0.5, pdf(0.5)=1.0, pdf(0.75)=1.5 (task brief's own oracle;
// independently: d/dx of the CDF x²  gives 2x directly).
// ---------------------------------------------------------------------

const TRIANGULAR_N = 800000;   // MC atoms for the runtime Z estimate (§06
                                 // normalize massFrom path — Lebesgue is
                                 // sampled as Uniform for this purpose, so
                                 // Z = ∫w dLebesgue is a Monte-Carlo mean,
                                 // not a closed form, once w is a genuine
                                 // function of the variate).
const TRIANGULAR_TOL = 5e-3;    // relative; empirically ~1e-3 at this N
                                 // (1/√N scaling) — 5e-3 leaves margin
                                 // against test-to-test RNG variance
                                 // without being loose enough to hide a
                                 // real regression (a dropped-weight bug
                                 // would be flat, i.e. off by ~2-3x, not
                                 // a few permille).

test('FRONT 1 (named): normalize(weighted(w, Lebesgue(0,1))) scores the triangular 2x density', async () => {
  const xs = [0.25, 0.5, 0.75];
  const src = `
w = x -> x
D = normalize(weighted(w, Lebesgue(support = interval(0, 1))))
` + ldBindings('D', xs);
  const ctx = buildCtx(src, TRIANGULAR_N, 101);
  const oracle: Record<string, number> = { '0.25': 0.5, '0.5': 1.0, '0.75': 1.5 };
  const scores = await scoreAll(ctx, xs);
  for (const xs2 of Object.keys(oracle)) {
    const density = Math.exp(scores[xs2]);
    const rel = Math.abs(density - oracle[xs2]) / oracle[xs2];
    assert.ok(rel < TRIANGULAR_TOL,
      `named form x=${xs2}: density=${density} oracle=${oracle[xs2]} rel=${rel}`);
  }
});

test('FRONT 1 (inline): normalize(weighted(x -> x, Lebesgue(0,1))) scores the SAME triangular 2x density', async () => {
  const xs = [0.25, 0.5, 0.75];
  const src = `
D = normalize(weighted(x -> x, Lebesgue(support = interval(0, 1))))
` + ldBindings('D', xs);
  const ctx = buildCtx(src, TRIANGULAR_N, 101);
  const oracle: Record<string, number> = { '0.25': 0.5, '0.5': 1.0, '0.75': 1.5 };
  const scores = await scoreAll(ctx, xs);
  for (const xs2 of Object.keys(oracle)) {
    const density = Math.exp(scores[xs2]);
    const rel = Math.abs(density - oracle[xs2]) / oracle[xs2];
    assert.ok(rel < TRIANGULAR_TOL,
      `inline form x=${xs2}: density=${density} oracle=${oracle[xs2]} rel=${rel}`);
  }
});

// ---------------------------------------------------------------------
// FRONT 1 — scalar-weight control: weighted(2.0, Lebesgue(0,1)) must stay
// exactly uniform after normalize (no regression from the function-weight
// fix — a plain constant weight takes the closedFormLogTotalmass path,
// untouched by any of this change).
// ---------------------------------------------------------------------

test('FRONT 1 control: normalize(weighted(2.0, Lebesgue(0,1))) is still exactly uniform', async () => {
  const xs = [0.25, 0.5, 0.75];
  const src = `
D = normalize(weighted(2.0, Lebesgue(support = interval(0, 1))))
` + ldBindings('D', xs);
  const ctx = buildCtx(src, 100, 7);
  const scores = await scoreAll(ctx, xs);
  for (const x of xs) {
    const density = Math.exp(scores[String(x)]);
    assert.ok(Math.abs(density - 1.0) < 1e-9,
      `scalar-weight control x=${x}: density=${density} (expected exactly 1.0)`);
  }
});

// ---------------------------------------------------------------------
// FRONT 1 — Chebyshev-series density (the fixture's D2 component,
// interval(0.5, 2.5) — b-a=2, so this ALSO exercises the Lebesgue-as-
// Uniform logTotalmass-parity fix (masked at b-a=1 by the triangular test
// above): a leftover factor-of-(b-a) mismatch between the numerator and Z
// would show up here as pdf values off by exactly 2x, not a few permille.
//
// ORACLE: scipy.integrate.quad of the SAME cheb_density(x) = Σ bᵢ Tᵢ(t),
// t=(2x-3)/2, over [0.5,2.5] (mcp__python__python_eval; independently
// reproduces the task brief's own given oracle to machine precision:
// Z=2.0666666666666673, matching exactly).
// ---------------------------------------------------------------------

test('FRONT 1: Chebyshev-series weighted density matches the scipy quadrature oracle', async () => {
  const oracle: Record<string, number> = {
    '0.6': 0.5051612903225805,
    '1': 0.4354838709677418,
    '1.5': 0.4354838709677419,
    '2': 0.5322580645161289,
    '2.4': 0.6793548387096772,
  };
  const xs = Object.keys(oracle).map(Number);
  const src = `
polynomials = standard_module("polynomials", "0.1")
chebyshev = polynomials.chebyshev
b0 = 1.1
b1 = 0.2
b2 = 0.2
a = 0.5
b = 2.5
cheb_density = x -> sum([b0, b1, b2] .* chebyshev.([0, 1, 2], (2 * x - (a + b)) / (b - a)))
D2 = normalize(weighted(cheb_density, Lebesgue(support = interval(a, b))))
` + ldBindings('D2', xs);
  const ctx = buildCtx(src, 1000000, 202);
  // scipy.integrate.quad(cheb_density, 0.5, 2.5) → Z=2.0666666666666673
  // (err ~2.3e-14); pdf = cheb_density(x)/Z. Verified against the task
  // brief's independently-stated oracle to full precision.
  const TOL = 1e-3;
  const scores = await scoreAll(ctx, xs);
  for (const xs2 of Object.keys(oracle)) {
    const density = Math.exp(scores[xs2]);
    const rel = Math.abs(density - oracle[xs2]) / oracle[xs2];
    assert.ok(rel < TOL,
      `Chebyshev D2 x=${xs2}: density=${density} oracle=${oracle[xs2]} rel=${rel}`);
  }
});

// ---------------------------------------------------------------------
// FRONT 2 — the complex-valued Breit-Wigner resonance weight (D1).
//
// full_intensity(x) = abs2(breit_wigner(x², m=1.5, width=0.1, l=0, d=1,
// ma=mb=0) * cis(3π/4) + (1.1 + 0.02x)), D1 = normalize(weighted(
// full_intensity, Lebesgue(support=interval(0.5,2.5)))).
//
// ORACLE (mcp__python__python_eval / scipy): replicated the
// resonance_breitwigner body (standard-modules.ts ~228-279) exactly for
// ma=mb=0, l=0 — the mass-dependent width Γ(σ) collapses to the constant Γ
// in this limit (the module's own closing note; independently re-derived:
// the breakup-momentum ratio p(σ)/p₀ = √σ/m and the l=0 Blatt-Weisskopf
// ratio is 1, so Γ(σ)·(m/√σ)·(p/p₀) = Γ exactly), giving the plain
// relativistic BW  BW(σ) = 1/(m² − σ − i·m·Γ). Verified the Python
// full_intensity(x) reproduction against the engine's own (pre-normalize)
// per-point evaluation at 5 grid points to full double precision before
// trusting it as the independent oracle. Z = ∫_0.5^2.5 full_intensity(x) dx
// via scipy.integrate.quad (Z≈7.448253470611266, quad err≈2.2e-8).
// ---------------------------------------------------------------------

const FIXTURE_PATH = path.join(__dirname, 'fixtures', 'hadron-physics-resonance.flatppl');
const FIXTURE_SRC = fs.readFileSync(FIXTURE_PATH, 'utf8');
const BW_N = 1500000;
const BW_TOL = 1e-2;   // relative; looser than the Chebyshev test's 1e-3 —
                        // full_intensity's sharp resonance peak (width
                        // Γ=0.1 against a [0.5,2.5] integration range) has
                        // much higher pointwise variance than the smooth
                        // Chebyshev density, so the MC estimate of Z
                        // converges more slowly for the same N (observed
                        // ~2e-3 at N=1e6, ~4e-3 at N=2e5 — empirically
                        // 1/√N as expected). 1e-2 stays a solid order of
                        // magnitude above the observed noise floor.

test('FRONT 2: D1 (Breit-Wigner resonance component) scores finite and matches the scipy oracle', async () => {
  // Oracle (scipy quad, see header): full_intensity(x)/Z.
  const oracle: Record<string, number> = {
    '0.6': 0.08354404747260157,
    '1': 0.06528637716448958,
    '1.5': 4.7081644593506535,
    '2': 0.3302653463592122,
    '2.4': 0.24715738703945622,
  };
  const xs = Object.keys(oracle).map(Number);
  const src = FIXTURE_SRC + ldBindings('D1', xs);
  const ctx = buildCtx(src, BW_N, 303);
  const scores = await scoreAll(ctx, xs);
  for (const xs2 of Object.keys(oracle)) {
    const ld = scores[xs2];
    assert.ok(Number.isFinite(ld), `D1 x=${xs2}: logdensity is not finite (${ld})`);
    const density = Math.exp(ld);
    const rel = Math.abs(density - oracle[xs2]) / oracle[xs2];
    assert.ok(rel < BW_TOL,
      `D1 x=${xs2}: density=${density} oracle=${oracle[xs2]} rel=${rel}`);
  }
});

test('FRONT 2: the mixture peaks near x=1.5 (the resonance mass) and matches f1*D1 + f2*D2', async () => {
  // Oracle: f1*D1_pdf(x) + f2*D2_pdf(x) with f1=0.4, f2=0.6, both D1/D2
  // pdfs from the SAME independent scipy quadrature as the tests above.
  // scipy grid scan over [0.5,2.5] (4001 points) puts the mixture's peak
  // at x≈1.5075 (value≈2.189) — near the resonance mass m=1.5, confirming
  // the fixture's whole point (it used to plot flat).
  const oracle: Record<string, number> = {
    '0.6': 0.3365143931825889,
    '1': 0.28740487344644094,
    '1.5': 2.1445561063209064,
    '2': 0.4514609772533622,
    '2.4': 0.5064758580415888,
  };
  const xs = Object.keys(oracle).map(Number);
  const src = FIXTURE_SRC + ldBindings('mixture', xs);
  const ctx = buildCtx(src, BW_N, 303);
  const scores = await scoreAll(ctx, xs);
  const values: Record<string, number> = {};
  for (const xs2 of Object.keys(oracle)) {
    const ld = scores[xs2];
    assert.ok(Number.isFinite(ld), `mixture x=${xs2}: logdensity is not finite (${ld})`);
    const density = Math.exp(ld);
    values[xs2] = density;
    const rel = Math.abs(density - oracle[xs2]) / oracle[xs2];
    assert.ok(rel < BW_TOL,
      `mixture x=${xs2}: density=${density} oracle=${oracle[xs2]} rel=${rel}`);
  }
  // The whole point of the card: the mixture PEAKS near x=1.5, not flat.
  const peakX = Object.keys(values).reduce((best, k) =>
    values[k] > values[best] ? k : best, Object.keys(values)[0]);
  assert.equal(peakX, '1.5', `mixture should peak at x=1.5, peaked at x=${peakX} instead`);
});

// ---------------------------------------------------------------------
// Guard: the fixture scores finite (no NaN) across a grid — the original
// symptom ("plots flat / NaN everywhere").
// ---------------------------------------------------------------------

test('guard: hadron-physics-resonance.flatppl scores finite (no NaN) across the [0.5,2.5] grid', async () => {
  const xs = [0.55, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.45];
  const src = FIXTURE_SRC
    + ldBindings('D1', xs).replace(/ld_/g, 'ld_d1_')
    + ldBindings('D2', xs).replace(/ld_/g, 'ld_d2_')
    + ldBindings('mixture', xs).replace(/ld_/g, 'ld_mix_');
  const ctx = buildCtx(src, 50000, 404);
  for (const x of xs) {
    for (const prefix of ['ld_d1_', 'ld_d2_', 'ld_mix_']) {
      const m = await ctx.getMeasure(prefix + xLabel(x));
      const ld = m.samples[0];
      assert.ok(Number.isFinite(ld),
        `${prefix} at x=${x}: logdensity is not finite (${ld}) — regression to the #307 NaN/flat bug`);
    }
  }
});
