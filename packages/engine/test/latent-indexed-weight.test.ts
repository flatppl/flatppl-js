'use strict';

// Indexing a STOCHASTIC array literal in a value position — a `weighted`
// weight or a distribution parameter.
//
//   psi ~ Beta(alpha = 2.0, beta = 3.0)
//   wp  = [psi, 1.0 - psi]
//   M   = superpose(weighted(wp[1], …), weighted(wp[2], …))
//
// This is the natural Beta-weighted two-component mixture. It produced NO
// derivation at all: `wp`, `M` and the query were all silently absent from the
// graph, and materialising the query threw `no derivation for 'ld'`.
//
// WHY. Spec §04 "Variates and measures" — "arithmetic on variates" — makes
// `[psi, 1.0 - psi]` a DETERMINISTIC node whose value is an array built from
// the variate `psi`. A fixed-phase array literal gets exactly that treatment:
// the fixed-value pre-eval resolves it, and `wp[1]` reads a real number (the
// two `psi = 0.3` tests below pass on the pre-fix engine and match the closed
// form). A stochastic one had no value path. `vector` is deliberately kept out
// of EVALUABLE_OPS — an array of stochastic refs does not fit the worker's
// scalar-per-atom contract — so classification fell to the array-literal
// branch, which accepted only two shapes: all-numeric-literal elements
// (`kind:'array'`) or all-self-ref elements (`kind:'tuple'`). `[psi, 1.0 -
// psi]` is neither (element 2 is a `sub` call), so classifyDerivation returned
// null and the cascade-prune took `M` and the query down with it.
//
// The all-self-ref spelling (`q = 1.0 - psi; wp = [psi, q]`) reached
// `kind:'tuple'` and then died differently — `feedInputs: measure for "wp" has
// neither .value nor .samples`. `kind:'tuple'` is a positional joint measure
// built for PLOTTING, assembled by `_materialiseFactorsIndependent`; a value
// consumer asking for `wp[1]` finds a measure with no per-atom samples.
//
// FIX. `foldStochasticVectorGets` folds `V[k]` to V's k-th element IR when V
// is a stochastic-phase array literal, before classification. The weight slot
// then holds `ref psi` / `sub(1.0, ref psi)` — the same IR the working
// direct-scalar spelling produces — so no new evaluator path is involved and
// the scalar-per-atom contract holds. `wp` keeps its own `tuple` derivation,
// so plotting the array is unaffected.
//
// ORACLES. The marginal density of the latent-psi mixture needs an integral,
// so it is NOT the check here. Two exact/independent anchors instead:
//   - EXACT, CLOSED FORM. Fix psi to a constant and score the CONDITIONAL
//     mixture: p(x) = psi·φ(x) + (1-psi)·φ(x-5), computed in-test from
//     Math.exp / Math.log, independent of the engine.
//   - MONTE CARLO. With psi latent the mixture must SAMPLE with component
//     proportions matching E[psi] = alpha/(alpha+beta) = 0.4.
// The Dirichlet spelling is pinned separately: it already worked, and must not
// move.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function buildCtx(src: string, N: number, seed = 7) {
  const proc = processSource(src);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const built = orchestrator.buildDerivations(proc.bindings);
  const berrs = (built.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(berrs.length, 0, berrs.map((e: any) => e.message).join(' | '));
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed });
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

function derivationsOf(src: string) {
  const proc = processSource(src);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  return orchestrator.buildDerivations(proc.bindings).derivations;
}

async function scoreOf(src: string, binding: string, N = 4096, seed = 7) {
  const ctx = buildCtx(src, N, seed);
  return (await ctx.getMeasure(binding)).samples[0];
}

async function samplesOf(src: string, binding: string, N = 8192, seed = 7) {
  const ctx = buildCtx(src, N, seed);
  return (await ctx.getMeasure(binding)).samples;
}

// Standard normal density — the in-test closed form.
const phi = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

// ── The two-component mixture, spelled through an indexed array.
const MIX_INDEXED = (psiDecl: string) => `
${psiDecl}
wp = [psi, 1.0 - psi]
M = superpose(weighted(wp[1], Normal(mu = 0.0, sigma = 1.0)), weighted(wp[2], Normal(mu = 5.0, sigma = 1.0)))
`;

// The same measure with the weights as separate scalar bindings — the spelling
// that already worked, kept as an independent control.
const MIX_SCALAR = (psiDecl: string) => `
${psiDecl}
q = 1.0 - psi
M = superpose(weighted(psi, Normal(mu = 0.0, sigma = 1.0)), weighted(q, Normal(mu = 5.0, sigma = 1.0)))
`;

const BETA = 'psi ~ Beta(alpha = 2.0, beta = 3.0)';

// =====================================================================
// Derivation — the regression proper
// =====================================================================

test('an indexed LATENT array weight derives its measure', () => {
  const d = derivationsOf(MIX_INDEXED(BETA) + '\nld = logdensityof(M, 1.0)\n');
  assert.equal(d['M'] && d['M'].kind, 'superpose',
    'the indexed latent weight was pruned: derivations are ' + Object.keys(d).join(', '));
  assert.equal(d['ld'] && d['ld'].kind, 'logdensityof',
    'the query was pruned: derivations are ' + Object.keys(d).join(', '));
});

// The all-refs spelling — this one always CLASSIFIED (kind:'tuple') and then
// failed at materialisation with `feedInputs: measure for "wp" has neither
// .value nor .samples`. So it needs a scoring assertion, not a derivation one.
const ALL_REFS = (psiDecl: string) => `
${psiDecl}
q = 1.0 - psi
wp = [psi, q]
M = superpose(weighted(wp[1], Normal(mu = 0.0, sigma = 1.0)), weighted(wp[2], Normal(mu = 5.0, sigma = 1.0)))
`;

test('an all-refs indexed LATENT array weight samples instead of throwing', async () => {
  // psi must stay LATENT here: a fixed psi resolves through the fixed-value
  // pre-eval and never touches the tuple path this test is about.
  const s = await samplesOf(ALL_REFS(BETA), 'M');
  let below = 0;
  for (let i = 0; i < s.length; i++) if (s[i] < 2.5) below++;
  assert.ok(Math.abs(below / s.length - 0.4) < 0.03,
    'all-refs share ' + (below / s.length) + ' ≠ E[psi] = 0.4 within 0.03');
});

test('an indexed LATENT array in a distribution parameter derives', () => {
  const d = derivationsOf(`
${BETA}
wp = [psi, 1.0 - psi]
M = Normal(mu = wp[1], sigma = 1.0)
`);
  assert.equal(d['M'] && d['M'].kind, 'sample',
    'indexed latent parameter pruned: ' + Object.keys(d).join(', '));
});

test('the array binding keeps its own tuple derivation', () => {
  // The fold rewrites CONSUMERS of `wp[k]`, not `wp` itself, so the viewer
  // still plots the array through kind:'tuple'.
  //
  // KNOWN GAP, deliberately not widened here: this holds for the all-refs
  // spelling only. A MIXED array literal (`[psi, 1.0 - psi]`) still gets no
  // derivation of its own, so `wp` itself is unplottable there even though
  // every `wp[k]` consumer now works. Closing that needs lift to hoist
  // non-ref elements, which would also re-route working fixed-phase arrays.
  const d = derivationsOf(ALL_REFS(BETA));
  assert.equal(d['wp'] && d['wp'].kind, 'tuple',
    'wp should stay a tuple measure, got ' + JSON.stringify(d['wp']));
});

// =====================================================================
// EXACT — the conditional mixture against the closed form
// =====================================================================

test('a FIXED indexed weight scores the closed-form mixture density', async () => {
  // psi·φ(x) + (1-psi)·φ(x-5), exact: no marginalisation is involved.
  for (const [psi, x] of [[0.3, 1.0], [0.75, 4.0], [0.1, 2.5]] as [number, number][]) {
    const want = Math.log(psi * phi(x) + (1 - psi) * phi(x - 5));
    const got = await scoreOf(
      MIX_INDEXED('psi = ' + psi) + '\nld = logdensityof(M, ' + x + ')\n', 'ld');
    assert.ok(Math.abs(got - want) < 1e-9,
      'psi=' + psi + ' x=' + x + ': log density ' + got + ' ≠ ' + want);
  }
});

test('the scalar spelling independently scores the same closed form', async () => {
  // The control. Each spelling is checked against the closed form SEPARATELY;
  // agreement between the two alone would prove nothing.
  for (const [psi, x] of [[0.3, 1.0], [0.75, 4.0], [0.1, 2.5]] as [number, number][]) {
    const want = Math.log(psi * phi(x) + (1 - psi) * phi(x - 5));
    const got = await scoreOf(
      MIX_SCALAR('psi = ' + psi) + '\nld = logdensityof(M, ' + x + ')\n', 'ld');
    assert.ok(Math.abs(got - want) < 1e-9,
      'scalar psi=' + psi + ' x=' + x + ': log density ' + got + ' ≠ ' + want);
  }
});

test('a FIXED indexed array in a distribution parameter scores Normal(psi, 1)', async () => {
  const got = await scoreOf(`
psi = 0.3
wp = [psi, 1.0 - psi]
M = Normal(mu = wp[1], sigma = 1.0)
ld = logdensityof(M, 1.0)
`, 'ld');
  assert.ok(Math.abs(got - Math.log(phi(1.0 - 0.3))) < 1e-9,
    'indexed mu: ' + got + ' ≠ ' + Math.log(phi(0.7)));
});

test('the SECOND element scores too — the fold is 1-indexed (spec §03)', async () => {
  // wp[2] must be 1.0 - psi = 0.7, not psi. A 0-indexed fold would score
  // Normal(0.3, 1) here and the two numbers differ by ~0.3 in log density.
  const got = await scoreOf(`
psi = 0.3
wp = [psi, 1.0 - psi]
M = Normal(mu = wp[2], sigma = 1.0)
ld = logdensityof(M, 1.0)
`, 'ld');
  assert.ok(Math.abs(got - Math.log(phi(1.0 - 0.7))) < 1e-9,
    'wp[2] as mu: ' + got + ' ≠ ' + Math.log(phi(0.3)));
});

// =====================================================================
// MONTE CARLO — the latent mixture's component proportions
// =====================================================================

test('the LATENT indexed mixture samples in proportion to E[psi]', async () => {
  // psi ~ Beta(2, 3) ⇒ E[psi] = 2/5 = 0.4. The components are Normal(0,1) and
  // Normal(5,1), which overlap negligibly at the 2.5 midpoint, so the fraction
  // of draws below 2.5 estimates E[psi]. N = 8192 gives a standard error of
  // ~0.006 on the multinomial part; the tolerance below is 5σ of that.
  const s = await samplesOf(MIX_INDEXED(BETA), 'M');
  let below = 0;
  for (let i = 0; i < s.length; i++) if (s[i] < 2.5) below++;
  const frac = below / s.length;
  assert.ok(Math.abs(frac - 0.4) < 0.03,
    'component-1 share ' + frac + ' ≠ E[psi] = 0.4 within 0.03');
});

test('the LATENT scalar spelling samples the same proportion', async () => {
  const s = await samplesOf(MIX_SCALAR(BETA), 'M');
  let below = 0;
  for (let i = 0; i < s.length; i++) if (s[i] < 2.5) below++;
  assert.ok(Math.abs(below / s.length - 0.4) < 0.03,
    'scalar spelling share ' + (below / s.length) + ' ≠ 0.4 within 0.03');
});

// =====================================================================
// The Dirichlet spelling must not move
// =====================================================================

test('the Dirichlet-weighted mixture is unchanged', async () => {
  // This spelling always worked — Dirichlet is a single vector-valued draw, so
  // `wp` is a leaf sample rather than an array literal, and the fold never
  // sees it. Pinned against the pre-change value on this branch.
  const got = await scoreOf(`
wp ~ Dirichlet(alpha = [2.0, 3.0])
M = superpose(weighted(wp[1], Normal(mu = 0.0, sigma = 1.0)), weighted(wp[2], Normal(mu = 5.0, sigma = 1.0)))
ld = logdensityof(M, 1.0)
`, 'ld', 512);
  assert.equal(got, -2.558252810720372,
    'the Dirichlet spelling moved: ' + got);
});
