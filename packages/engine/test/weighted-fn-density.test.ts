'use strict';

// weighted(w, M) with a FUNCTION-of-variate weight — spec §06 "Density of
// composed measures": logdensityof(weighted(w, M), x) = log w(x) +
// logdensityof(M, x), w a constant OR a function of the variate. Buffy
// #307 FRONT 1: the engine dropped w entirely when it was a function
// (named-ref OR inline lambda) — the density path (numerator, mat-density's
// normalize-Z materialisation, and the §12-style symbolic totalmass fold)
// all treated `weighted(fn, M)` as if the weight were absent, so
// `normalize(weighted(fn, M))` came out flat/uniform regardless of fn.
//
// Root causes:
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
//
// Fixed: `expandMeasureIR`'s 'weighted' case re-wraps a weight IR that
// still self-refs its base into a `functionof` (density.ts's EXISTING
// inline-lambda branch then binds it correctly); `_expandByName`'s
// 'sample' case restores the dropped logTotalmass as a `logweighted`
// shift (parity with mat-density's matWeighted, which already reads it);
// `liftMeasure`'s hole/placeholder check treats an INLINE closed reifier
// (`fn`/`functionof`/`kernelof`) in a `weighted`/`logweighted` weight slot
// as opaque (its own holes never escape it) WITHOUT loosening the check
// anywhere else (jointchain/kchain kernel args, record/joint kwarg fields,
// broadcast array literals, and weighted's own BASE slot keep the original
// conservative behaviour — see lift.ts's `isClosedReifierCall`);
// `totalMassExpr` bails (null) on a `functionof`-shaped weight rather than
// misusing it as a scalar factor.
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
