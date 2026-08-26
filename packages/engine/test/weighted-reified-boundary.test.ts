'use strict';

// `weighted(w, M)` where the weight routes through a REIFICATION with declared
// boundary inputs — spec §06 "Density reweighting": "`weighted(weight, base)`
// … where $f$ is a non-negative weight (a constant or a function of the variate
// $x$ of $M$)", and spec §04 "Reification to functions and kernels": "`functionof`
// traces the ancestor subgraph of its argument back to all leaves". A reified
// function IS a function, so `weighted(functionof(expr, x = x), M)` is legal and
// must score the same measure as the equivalent lambda.
//
// It did not. `functionof(y, x = x)` stores its body as a bare `ref y` (the
// intermediate binding), which the self-contain pass in buildDerivations later
// inlines to reach the boundary. Classification ran BEFORE that pass, so
// `_classifyWeightedByFunction` bound the parameter against the pre-inlining
// body, matched nothing, and left the weight referring to `y` — whose own
// derivation the cascade-prune then dropped for referencing the parametric
// boundary `x`, taking the weighted binding and every measure above it down.
// Nothing failed loudly; the model simply had no derivation.
//
// ORACLES. Two classes, kept apart deliberately:
//   - EXACT. The 1-D normalizer is deterministic quadrature, so
//     `normalize(weighted(f, Lebesgue(interval)))` matches the closed-form
//     density to ~1e-8 and is independent of the sample count.
//   - APPROXIMATE. The N-D box normalizer estimates Z from the box's own
//     atoms. Measured on this branch, its error on a bilinear weight is
//     ~0.5%, is seed-independent, and does NOT shrink with the sample count
//     (Z over [0,2]x[0,3] reads 9.0408 at N=4096 and 9.0457 at N=65536
//     against a closed form of 9). That bias is PRE-EXISTING and unrelated to
//     reification: a plain-lambda weight returns the identical number. The
//     tolerances below are 1%, well inside the factor-level error a DROPPED
//     weight would produce.
//
// The load-bearing assertion is the last one: the reified spelling and the
// lambda spelling must agree BIT-FOR-BIT, since they denote the same measure.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function buildCtx(src: string, N: number, seed = 7) {
  const proc = processSource(src);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const built = orchestrator.buildDerivations(proc.bindings);
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

// ── 1-D. w(x) = x², base Lebesgue([0,1]).
//    Z = ∫₀¹ x² dx = 1/3, so the normalized density is x²/(1/3) = 3x².
//    At x = 0.5: 3(0.25) = 0.75.   At x = 0.8: 3(0.64) = 1.92.
const REIFIED_1D = `
x = elementof(interval(0.0, 1.0))
graph_out = x^2
w = functionof(graph_out, x = x)
M = normalize(weighted(w, Lebesgue(support = interval(0.0, 1.0))))
`;

const LAMBDA_1D = `
M = normalize(weighted(t -> t^2, Lebesgue(support = interval(0.0, 1.0))))
`;

// ── 2-D. w(x,y) = xy, base Lebesgue([0,1]²).
//    Z = ∫₀¹∫₀¹ xy dx dy = (1/2)(1/2) = 1/4, so the density is xy/(1/4) = 4xy.
//    At (0.5, 0.5): 4(0.25) = 1, so log density 0.
const REIFIED_2D = `
m = elementof(interval(0.0, 1.0))
c = elementof(interval(0.0, 1.0))
intensity = m * c
intensity_fn = functionof(intensity, m = m, c = c)
M = normalize(weighted(intensity_fn, Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 1.0)))))
`;

const LAMBDA_2D = `
M = normalize(weighted((a, b) -> a * b, Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 1.0)))))
`;

// =====================================================================
// Derivation — the regression proper
// =====================================================================

test('a 1-D reified-boundary weight derives its measure', () => {
  const d = derivationsOf(REIFIED_1D);
  assert.equal(d['M'] && d['M'].kind, 'normalize',
    'the reified weight was cascade-pruned: derivations are ' + Object.keys(d).join(', '));
});

test('an N-D reified-boundary weight derives its measure', () => {
  const d = derivationsOf(REIFIED_2D);
  assert.equal(d['M'] && d['M'].kind, 'normalize',
    'the reified 2-D weight was cascade-pruned: derivations are ' + Object.keys(d).join(', '));
});

test('a RENAMED boundary input derives too', () => {
  // §04: "g = functionof(e, p = a, q = d)" — the declared name need not match
  // the boundary binding's name. The substitution must key on the boundary
  // binding, which is what the lowered `params` carries.
  const d = derivationsOf(`
a = elementof(interval(0.0, 1.0))
y = a^2
w = functionof(y, p = a)
M = normalize(weighted(w, Lebesgue(support = interval(0.0, 1.0))))
`);
  assert.equal(d['M'] && d['M'].kind, 'normalize',
    'renamed-boundary reified weight pruned: ' + Object.keys(d).join(', '));
});

test('a named function that CALLS a reified function derives', () => {
  // The amplitude-analysis spelling: the weight is an ordinary two-parameter
  // function whose body applies the reified callable. Lifting hoists that
  // application to an anonymous binding, so the function's own formals end up
  // behind a ref the weight consumer cannot substitute through.
  const d = derivationsOf(`
m = elementof(interval(0.0, 1.0))
c = elementof(interval(0.0, 1.0))
intensity = m * c
intensity_fn = functionof(intensity, m = m, c = c)
pw(mass, cost) = intensity_fn(mass, cost)
M = normalize(weighted(pw, Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 1.0)))))
`);
  assert.equal(d['M'] && d['M'].kind, 'normalize',
    'reified-call weight pruned: ' + Object.keys(d).join(', '));
});

// =====================================================================
// 1-D density against the closed form (EXACT — deterministic quadrature)
// =====================================================================

test('1-D reified weight scores the closed-form density 3x²', async () => {
  for (const [pt, want] of [[0.5, 0.75], [0.8, 1.92]] as [number, number][]) {
    const got = await scoreOf(
      REIFIED_1D + '\nld = logdensityof(M, ' + pt.toFixed(1) + ')\n', 'ld');
    assert.ok(Math.abs(got - Math.log(want)) < 1e-7,
      'reified at x=' + pt + ': log density ' + got + ' ≠ log ' + want
        + ' = ' + Math.log(want));
  }
});

test('the lambda spelling independently scores 3x² too', async () => {
  // The control. Both spellings are checked against the closed form
  // SEPARATELY — agreement between them alone would prove nothing.
  for (const [pt, want] of [[0.5, 0.75], [0.8, 1.92]] as [number, number][]) {
    const got = await scoreOf(
      LAMBDA_1D + '\nld = logdensityof(M, ' + pt.toFixed(1) + ')\n', 'ld');
    assert.ok(Math.abs(got - Math.log(want)) < 1e-7,
      'lambda at x=' + pt + ': log density ' + got + ' ≠ log ' + want);
  }
});

test('1-D quadrature is exact — the score does not move with the sample count', async () => {
  const vals: number[] = [];
  for (const N of [512, 4096, 32768]) {
    vals.push(await scoreOf(REIFIED_1D + '\nld = logdensityof(M, 0.8)\n', 'ld', N));
  }
  for (const v of vals) {
    assert.ok(Math.abs(v - vals[0]) < 1e-12,
      'the 1-D normalizer should be deterministic quadrature, got ' + vals.join(' / '));
  }
});

// =====================================================================
// 2-D density and mass (APPROXIMATE — see the header)
// =====================================================================

test('2-D reified weight scores the closed-form density 4xy', async () => {
  const got = await scoreOf(REIFIED_2D + '\nld = logdensityof(M, [0.5, 0.5])\n', 'ld');
  // 4·(0.5)·(0.5) = 1, so log density 0.
  assert.ok(Math.abs(Math.exp(got) - 1) < 0.01,
    '2-D reified at (0.5,0.5): density ' + Math.exp(got) + ' ≠ 1 within 1%');
});

test('totalmass of a reified weight over [0,2]x[0,3] is 9', async () => {
  // Z = ∫₀²∫₀³ xy dy dx = (2²/2)·(3²/2) = 2·4.5 = 9.
  const got = await scoreOf(`
m = elementof(interval(0.0, 2.0))
c = elementof(interval(0.0, 3.0))
intensity = m * c
fn = functionof(intensity, m = m, c = c)
Z = totalmass(weighted(fn, Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))))
`, 'Z');
  assert.ok(Math.abs(got - 9) / 9 < 0.01,
    'totalmass ' + got + ' ≠ 9 within 1%');
});

// =====================================================================
// The invariant: the two spellings denote ONE measure
// =====================================================================

test('the reified and lambda spellings agree bit-for-bit', async () => {
  // Same function, same base, so the same measure — any difference means one
  // of the two paths is scoring something else.
  const r1 = await scoreOf(REIFIED_1D + '\nld = logdensityof(M, 0.8)\n', 'ld');
  const l1 = await scoreOf(LAMBDA_1D + '\nld = logdensityof(M, 0.8)\n', 'ld');
  assert.equal(r1, l1, '1-D: reified ' + r1 + ' vs lambda ' + l1);

  const r2 = await scoreOf(REIFIED_2D + '\nld = logdensityof(M, [0.5, 0.5])\n', 'ld');
  const l2 = await scoreOf(LAMBDA_2D + '\nld = logdensityof(M, [0.5, 0.5])\n', 'ld');
  assert.equal(r2, l2, '2-D: reified ' + r2 + ' vs lambda ' + l2);
});

test('an ordinary lambda body sharing an intermediate is left alone', async () => {
  // Guard on the narrowness of the self-contain extension: an intermediate
  // that does NOT carry the function's formals stays a ref, so a lambda
  // closing over a resolvable constant keeps scoring as before.
  // scale = 3, w(t) = 3t² ⇒ Z = 3·(1/3) = 1 and the density is still 3t².
  const got = await scoreOf(`
scale = 3.0
M = normalize(weighted(t -> scale * t^2, Lebesgue(support = interval(0.0, 1.0))))
ld = logdensityof(M, 0.8)
`, 'ld');
  assert.ok(Math.abs(got - Math.log(1.92)) < 1e-7,
    'shared-intermediate lambda: ' + got + ' ≠ log 1.92');
});
