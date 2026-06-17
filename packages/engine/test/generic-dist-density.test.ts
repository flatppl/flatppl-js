'use strict';

// Generic-density (HS3 generic_dist, RooFit RooGenericPdf) scoring — spec §12.
//
// A generic_dist is a NORMALIZED density over its observable's bounded domain:
// RooFit normalizes RooGenericPdf over the observable's fit range. It lowers to
//   normalize(truncate(weighted(x -> w(x), Lebesgue(reals)), interval(lo, hi)))
// so its density on S = [lo, hi] is  w(x) / ∫_S w(x) dx.
//
// Scoring this previously THREW: the #37 normalize-truncate normalizer only
// recognised scalar reference-measure bases (Normal closed-form / its narrow
// quadrature path). A `weighted(<weight-fn>, Lebesgue(reals))` base is a generic
// unnormalized density — its support-restricted normalizer Z = ∫_S w(x) dx is
// computed by composite-midpoint quadrature (QUAD_POINTS = 8192) of the weight
// function at the fixed point θ (mat-density.weightedBaseLogMass), reusing the
// numericProductLogZ midpoint machinery. The per-point density walk gained a
// Lebesgue handler (density ≡ 1 → 0 logpdf) and a functionof-weight branch for
// weighted (density.walkLebesgue / walkWeighted).
//
// INDEPENDENT ORACLE — derived via scipy (numpy midpoint sum, 8192 nodes),
// NOT any engine output. For w(x; α) = 1 + 0.1·|x| + sin(√|x·α + 0.1|) on
// S = [−20, 20]:
//   Z(α=5)        = 86.2772714369     logZ(5) = 4.4575661965
//   ld(x=1,  α=5) = log w(1, 5)  − logZ(5) = −3.8301202295
//   ld(x=−7.3,α=5)= log w(−7.3,5)− logZ(5) = −4.0637350031
// (The engine's 8192-pt midpoint Z agrees with scipy's adaptive quadrature to
// ~1e-6, so the per-point score matches the oracle well within 1e-4.)

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function buildCtx(src: string, N: number) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

const WEIGHT = '1.0 + 0.1 * abs(x) + sin(sqrt(abs(x * alpha + 0.1)))';
const MODEL = (likelihood: string) => `
alpha = elementof(interval(0.1, 10.0))
genpdf = normalize(truncate(weighted(x -> ${WEIGHT}, Lebesgue(reals)), interval(-20.0, 20.0)))
${likelihood}
ld = logdensityof(L, record(alpha = 5.0))
`;

test('generic_dist: per-point density at x=1, α=5 matches the scipy oracle', async () => {
  const ORACLE = -3.8301202295;   // scipy-derived (see header); NOT an engine output
  const ctx = buildCtx(MODEL('L = likelihoodof(genpdf, 1.0)'), 1);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - ORACLE) < 1e-4,
    `score ${m.samples[0]} ≈ oracle ${ORACLE} (Δ ${Math.abs(m.samples[0] - ORACLE)})`);
});

test('generic_dist: per-point density at x=-7.3, α=5 matches the scipy oracle', async () => {
  const ORACLE = -4.0637350031;   // scipy-derived (see header); NOT an engine output
  const ctx = buildCtx(MODEL('L = likelihoodof(genpdf, -7.3)'), 1);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - ORACLE) < 1e-4,
    `score ${m.samples[0]} ≈ oracle ${ORACLE} (Δ ${Math.abs(m.samples[0] - ORACLE)})`);
});

test('generic_dist: iid likelihood is the sum of per-entry normalized scores', async () => {
  // Two-entry dataset: the unbinned likelihood is the SUM of the two per-point
  // normalized densities (each scored above). Independent: oracle sum.
  const ORACLE = -3.8301202295 + -4.0637350031;
  const ctx = buildCtx(MODEL('L = likelihoodof(iid(genpdf, 2), [1.0, -7.3])'), 1);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - ORACLE) < 1e-4,
    `Σ score ${m.samples[0]} ≈ oracle ${ORACLE} (Δ ${Math.abs(m.samples[0] - ORACLE)})`);
});
