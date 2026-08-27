'use strict';

// =====================================================================
// generative-boundary-surface-name.test.ts
// =====================================================================
//
// `_executeGenerativeComposite` keys its layout maps by the kernel's
// FORMAL parameter name but hands `bcKwargs` to `_substituteKernelParams`,
// which looks a boundary up by the SURFACE kwarg name. The two coincide
// only under the off-spec `x = x` shorthand. Spec §11 "Reified callables"
// admits two spellings that make them DIFFER:
//
//   - a placeholder within the output: `kernelof(<body>, x = _x_)`
//   - a renaming cut to an ancestor node: `kernelof(<body>, x = xin)`
//
// Under either one a formal-keyed `bcKwargs` left every boundary ref
// unsubstituted, and the batch evaluation failed with "unbound %local
// reference" / "unbound self reference". The same defect class was fixed
// for `_executeJointComposite` earlier; this pins the generative twin.
//
// Oracle is closed-form. The body is
//
//   y = (x + 0.1 * (2u + 1))^3 * exp(x - 0.3),   u ~ Uniform(0, 1)
//
// so with a = x + 0.1 and b = 0.2 the per-cell support is
// [a^3, (a+b)^3] * exp(x - 0.3) and
//
//   E[y] = exp(x - 0.3) * ((a+b)^4 - a^4) / (4b).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');
const materialiser = require('../materialiser.ts');
const { createWorkerHandler } = require('../worker.ts');

const XS = [0.1, 0.2, 0.3];

// Boundary spelled as a placeholder (§04 "Placeholders and holes").
const PLACEHOLDER_SRC = `
delta_u = draw(Uniform(interval(0, 1)))
transport = kernelof(
  (_x_ + (2 * delta_u + 1) * 0.1)^3 * exp(_x_ - 0.3),
  x = _x_)
xs = [${XS.join(', ')}]
ys ~ transport.(x = xs)
`;

// Same measure, boundary spelled as a RENAMING cut to a module node.
const CUT_SRC = `
delta_u = draw(Uniform(interval(0, 1)))
xin = elementof(reals)
transport = kernelof(
  (xin + (2 * delta_u + 1) * 0.1)^3 * exp(xin - 0.3),
  x = xin)
xs = [${XS.join(', ')}]
ys ~ transport.(x = xs)
`;

// Placeholder boundary reached POSITIONALLY (`K.(xs)`), which fills
// `d.argIRs` instead of `d.kwargIRs`.
const POSITIONAL_SRC = `
delta_u = draw(Uniform(interval(0, 1)))
transport = kernelof(
  (_x_ + (2 * delta_u + 1) * 0.1)^3 * exp(_x_ - 0.3),
  x = _x_)
xs = [${XS.join(', ')}]
ys ~ transport.(xs)
`;

function drawYs(src: string, N: number, seed: number) {
  const lifted = processSource(src);
  const errors = (lifted.diagnostics || [])
    .filter((dg: any) => dg.severity === 'error').map((dg: any) => dg.message);
  assert.deepEqual(errors, [], 'source analyses without errors');
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (nm: string) => {
      if (cache.has(nm)) return cache.get(nm);
      const p = materialiser.materialiseMeasure(nm, ctx);
      cache.set(nm, p);
      return p;
    },
    sendWorker: (msg: any) => Promise.resolve(worker.handle(msg)),
    sampleCount: N, rootKey: [seed, 0],
  };
  return ctx.getMeasure('ys');
}

function cellSupport(x: number) {
  const a = x + 0.1, b = 0.2, s = Math.exp(x - 0.3);
  return { lo: a * a * a * s, hi: (a + b) * (a + b) * (a + b) * s };
}

function cellMean(x: number) {
  const a = x + 0.1, b = 0.2;
  return Math.exp(x - 0.3) * (Math.pow(a + b, 4) - Math.pow(a, 4)) / (4 * b);
}

test('generative composite: placeholder boundary substitutes and matches the closed form', async () => {
  const N = 40000;
  const m = await drawYs(PLACEHOLDER_SRC, N, 11);
  const K = XS.length;
  assert.deepEqual(m.value.shape, [N, K]);
  const d = m.value.data;
  for (let j = 0; j < K; j++) {
    const { lo, hi } = cellSupport(XS[j]);
    let sum = 0;
    for (let i = 0; i < N; i++) {
      const v = d[i * K + j];
      assert.ok(v >= lo - 1e-12 && v <= hi + 1e-12,
        `cell ${j} draw ${v} outside the analytic support [${lo}, ${hi}]`);
      sum += v;
    }
    const got = sum / N, want = cellMean(XS[j]);
    assert.ok(Math.abs(got - want) < 0.002,
      `cell ${j} mean ${got} vs closed form ${want}`);
  }
});

test('generative composite: renaming cut draws identically to the placeholder spelling', async () => {
  const N = 2000;
  const [a, b] = await Promise.all([
    drawYs(PLACEHOLDER_SRC, N, 11), drawYs(CUT_SRC, N, 11),
  ]);
  assert.deepEqual(a.value.shape, b.value.shape);
  let differing = 0;
  for (let i = 0; i < a.value.data.length; i++) {
    if (a.value.data[i] !== b.value.data[i]) differing++;
  }
  assert.equal(differing, 0,
    'the two spec-legal boundary spellings name the same measure');
});

test('generative composite: positional boundary arg also reaches the body', async () => {
  const N = 2000;
  const [a, b] = await Promise.all([
    drawYs(PLACEHOLDER_SRC, N, 11), drawYs(POSITIONAL_SRC, N, 11),
  ]);
  assert.deepEqual(a.value.shape, b.value.shape);
  let differing = 0;
  for (let i = 0; i < a.value.data.length; i++) {
    if (a.value.data[i] !== b.value.data[i]) differing++;
  }
  assert.equal(differing, 0,
    'a positional broadcast arg binds the same boundary as the kwarg form');
});
