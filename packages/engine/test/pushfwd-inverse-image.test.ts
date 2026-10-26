'use strict';

// A registry bijection (bijection-registry.ts, spec §06 case 1) scores
// pushfwd(f, M) at y through its inverse. The density outside the image of f
// is 0, so log density -Infinity. Three inverses were wrong:
//   - an odd power k maps the reals onto the reals, but y^(1/k) is NaN for
//     y < 0, so `pushfwd(fn(_^3), Normal(0, 1))` scored -Infinity at every
//     y < 0 and carried half its mass;
//   - k = 1/2, 1/4 (and `sqrt`) map [0, inf] into [0, inf], but the inverse
//     y^(1/k) is finite for y < 0, so a point outside the image scored finite;
//   - atan maps onto (-pi/2, pi/2), but tan is finite outside it.
//
// ORACLE: scipy.stats.norm.logpdf and numpy, run by hand, not the engine:
//   Y = X^3, X ~ Normal(0, 1):  log f(y) = log phi(cbrt y) - log(3 cbrt(y)^2)
//     y = +-1: -2.5175508218727822    y = +-8: -5.403845182992673
//   Y = X^5:                    log f(-3) = -4.183189063530941
//   Y = 2 X^3 + 1:              log f(-1) = log f_{X^3}(-1) - log 2 = -3.2106980024327276
//   Y = sqrt(X), X ~ Exponential(1): f(y) = 2 y exp(-y^2) on y >= 0
//     log f(2) = -2.613705638880109, f(y) = 0 for y < 0
//   Y = X^(1/4), X ~ Exponential(1): log f(1.2) = -1.2^4 + log(4 * 1.2^3) = -0.1403409684982455
//   Y = atan(X), X ~ Normal(0, 1): log f(y) = log phi(tan y) + log(1 + tan^2 y)
//     y = +-1: -0.9004450028400242, f(y) = 0 for |y| > pi/2
// scipy.integrate.quad gives total mass 1 for the cube, sqrt and atan densities.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { processSource } = require('../index.ts');

async function scores(measure: string, pts: number[]): Promise<number[]> {
  const src = `M = ${measure}\nlp = broadcast(logdensityof, M, [${pts.join(', ')}])\n`;
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('lp');
  return Array.from((m.value?.data ?? m.samples) as ArrayLike<number>);
}

async function score(measure: string, y: number): Promise<number> {
  const { ctx } = ctxFor(`M = ${measure}\nlp = logdensityof(M, ${y})\n`, 1);
  const m = await ctx.getMeasure('lp');
  return Array.from((m.value?.data ?? m.samples) as ArrayLike<number>)[0];
}

const close = (got: number, want: number, what: string) =>
  assert.ok(Math.abs(got - want) < 1e-12, `${what}: got ${got}, want ${want}`);

const CASES: [string, number, number][] = [
  ['pushfwd(fn(_^3), Normal(0, 1))', 1, -2.5175508218727822],
  ['pushfwd(fn(_^3), Normal(0, 1))', -1, -2.5175508218727822],
  ['pushfwd(fn(_^3), Normal(0, 1))', -8, -5.403845182992673],
  ['pushfwd(fn(pow(_, 3)), Normal(0, 1))', -1, -2.5175508218727822],
  ['pushfwd(fn(_^5), Normal(0, 1))', -3, -4.183189063530941],
  ['pushfwd(fn(2 * _^3 + 1), Normal(0, 1))', -1, -3.2106980024327276],
  ['pushfwd(fn(sqrt(_)), Exponential(1))', 2, -2.613705638880109],
  ['pushfwd(fn(_^0.5), Exponential(1))', 2, -2.613705638880109],
  ['pushfwd(fn(sqrt(_)), Exponential(1))', -1, -Infinity],
  ['pushfwd(fn(_^0.5), Exponential(1))', -1, -Infinity],
  ['pushfwd(fn(_^2), Exponential(1))', -4, -Infinity],
  ['pushfwd(fn(_^0.25), Exponential(1))', 1.2, -0.1403409684982455],
  ['pushfwd(fn(_^0.25), Exponential(1))', -1.2, -Infinity],
  ['pushfwd(fn(atan(_)), Normal(0, 1))', 1, -0.9004450028400242],
  ['pushfwd(fn(atan(_)), Normal(0, 1))', -1, -0.9004450028400242],
  ['pushfwd(fn(atan(_)), Normal(0, 1))', 2, -Infinity],
  ['pushfwd(fn(atan(_)), Normal(0, 1))', -2, -Infinity],
  ['pushfwd(fn(atan(_) + 1), Normal(0, 1))', 3, -Infinity],
];

for (const [measure, y, want] of CASES) {
  test(`${measure} at y = ${y} scores the closed form`, async () => {
    const got = await score(measure, y);
    if (want === -Infinity) assert.equal(got, -Infinity, `${measure} at ${y}`);
    else close(got, want, `${measure} at ${y}`);
  });
}

test('an odd power scores each batched point on its own branch', async () => {
  const got = await scores('pushfwd(fn(_^3), Normal(0, 1))', [1, -1, 8, -8]);
  const want = [-2.5175508218727822, -2.5175508218727822, -5.403845182992673, -5.403845182992673];
  got.forEach((g, i) => close(g, want[i], `point ${i}`));
});

// Substituting y = t^3 removes the |y|^(-2/3) pole at 0:
// mass = integral of f(t^3) 3 t^2 dt, by the midpoint rule.
// A dropped negative branch gives 0 below zero.
test('the cube of a standard normal keeps mass 1, half on each side', async () => {
  const h = 0.01;
  const ts: number[] = [];
  for (let i = 0; i < 1400; i++) ts.push(-7 + h / 2 + i * h);
  const ld = await scores('pushfwd(fn(_^3), Normal(0, 1))', ts.map((t) => t ** 3));
  let neg = 0, pos = 0;
  ts.forEach((t, i) => {
    const w = Math.exp(ld[i]) * 3 * t * t * h;
    if (t < 0) neg += w; else pos += w;
  });
  assert.ok(Math.abs(neg - 0.5) < 1e-6, `mass on y < 0: ${neg}`);
  assert.ok(Math.abs(pos - 0.5) < 1e-6, `mass on y > 0: ${pos}`);
});

test('the square root of an Exponential(1) keeps mass 1 on y >= 0', async () => {
  const h = 0.001;
  const ys: number[] = [-2, -1, -0.5];
  for (let i = 0; i < 8000; i++) ys.push(h / 2 + i * h);
  const ld = await scores('pushfwd(fn(sqrt(_)), Exponential(1))', ys);
  let mass = 0;
  ys.forEach((y, i) => { if (y < 0) assert.equal(ld[i], -Infinity, `y = ${y}`); else mass += Math.exp(ld[i]) * h; });
  assert.ok(Math.abs(mass - 1) < 1e-5, `mass: ${mass}`);
});

test('atan of a standard normal keeps mass 1 on (-pi/2, pi/2) and none outside', async () => {
  const h = 0.0005;
  const ys: number[] = [-3, -2, -1.6, 1.6, 2, 3];
  const n = Math.round(Math.PI / h);
  for (let i = 0; i < n; i++) ys.push(-Math.PI / 2 + (i + 0.5) * (Math.PI / n));
  const ld = await scores('pushfwd(fn(atan(_)), Normal(0, 1))', ys);
  let mass = 0;
  ys.forEach((y, i) => {
    if (Math.abs(y) > Math.PI / 2) assert.equal(ld[i], -Infinity, `y = ${y}`);
    else mass += Math.exp(ld[i]) * (Math.PI / n);
  });
  assert.ok(Math.abs(mass - 1) < 1e-6, `mass: ${mass}`);
});

// §06: `pow` needs a base support within nonnegreals. Only an odd INTEGER
// literal exponent is exempt, as a bijection of the reals; a real literal
// (`3.0`), even when integral, and a non-literal exponent are not.
for (const f of ['_^3.0', 'pow(_, 3.0)', '_^(1/3)', '_^(-1.0)']) {
  test(`pushfwd(fn(${f}), Normal(0, 1)) refuses with a located diagnostic`, () => {
    const errs = processSource(`M = pushfwd(fn(${f}), Normal(0, 1))\nlp = logdensityof(M, -1)\n`)
      .diagnostics.filter((d: any) => d.severity === 'error');
    assert.equal(errs.length, 1, JSON.stringify(errs.map((d: any) => d.message)));
    assert.match(errs[0].message, /uses 'pow', which requires the base measure's support to be non-negative/);
    assert.ok(errs[0].loc && errs[0].loc.start, 'diagnostic is located');
  });
}

for (const f of ['_^3', 'pow(_, 3)', '_^(-1)']) {
  test(`pushfwd(fn(${f}), Normal(0, 1)) has no domain diagnostic`, () => {
    const errs = processSource(`M = pushfwd(fn(${f}), Normal(0, 1))\nlp = logdensityof(M, -1)\n`)
      .diagnostics.filter((d: any) => d.severity === 'error');
    assert.deepEqual(errs.map((d: any) => d.message), []);
  });
}

// `_signed_pow` and `_image_gate` are engine-internal. The inverse is built
// in buildDerivations, after lowering, so it lives only in the expanded
// measure the materialiser hands to the worker. FlatPIR prints the lowered
// module's binding RHS, and source cannot name the ops.
test('the internal inverse ops never reach FlatPIR or source', () => {
  const { orchestrator, pirSexpr } = require('../index.ts');
  const proc = processSource('M = pushfwd(fn(_^3), Normal(0, 1))\nN = pushfwd(fn(atan(_)), Normal(0, 1))\n');
  const built = orchestrator.buildDerivations(proc.linkedBindings, { moduleRegistry: proc.linkedModuleRegistry });
  const ctx = { derivations: built.derivations, bindings: built.bindings };
  assert.match(JSON.stringify(orchestrator.expandMeasure('M', ctx)), /"_signed_pow"/);
  assert.match(JSON.stringify(orchestrator.expandMeasure('N', ctx)), /"_image_gate"/);
  for (const meta of [false, true]) {
    assert.doesNotMatch(pirSexpr.toSexpr(proc.loweredModule, { meta }), /_signed_pow|_image_gate/);
  }
  for (const src of ['a = _signed_pow(-8.0, 0.5)\n', 'b = _image_gate(1.0, 0.0, 2.0)\n']) {
    const errs = processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
    assert.match(errs[0].message, /Undefined variable '_(signed_pow|image_gate)'/);
  }
});

// -0 is 0, inside the image: sqrt(Exp(1)) has density 2y exp(-y^2) = 0 at y = 0.
test('y = -0 scores the y = 0 density, not a finite value off the image', async () => {
  const at = async (measure: string, y: string) => {
    const { ctx } = ctxFor(`M = ${measure}\nlp = logdensityof(M, ${y})\n`, 1);
    const m = await ctx.getMeasure('lp');
    return Array.from((m.value?.data ?? m.samples) as ArrayLike<number>)[0];
  };
  assert.equal(await at('pushfwd(fn(sqrt(_)), Exponential(1))', '-0.0'), -Infinity);
  assert.equal(await at('pushfwd(fn(sqrt(_)), Exponential(1))', '0.0'), -Infinity);
  assert.equal(await at('pushfwd(fn(_^3), Normal(0, 1))', '-0.0'), await at('pushfwd(fn(_^3), Normal(0, 1))', '0.0'));
  assert.equal(await at('pushfwd(fn(_^0.5), Exponential(1))', '-0.0'), -Infinity);
});

test('the elementwise forms of the internal inverse ops match their scalar forms', () => {
  const vo = require('../value-ops.ts');
  const { scalar } = require('../value.ts');
  const ys = [-8, -0.5, 0, 0.5, 27];
  const v = { shape: [5], data: Float64Array.from(ys) };
  assert.deepEqual(Array.from(vo.signedPowElem(v, scalar(1 / 3)).data), ys.map((y) => vo._signedPowScalar(y, 1 / 3)));
  const gated: number[] = Array.from(vo.imageGateElem(v, scalar(0), scalar(1)).data as Float64Array);
  assert.deepEqual(gated.map((g: number) => (Number.isNaN(g) ? 'NaN' : g)), ['NaN', 'NaN', 0, 0.5, 'NaN']);
});
