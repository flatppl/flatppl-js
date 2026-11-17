'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { processSource } = require('../index.ts');
const sampler = require('../sampler.ts');

const source = 'sp = standard_module("splines", "0.1")\n';
const close = (actual: number, expected: number, tolerance = 2e-12) =>
  assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);

function evaluate(expression: string): any {
  const result = processSource(source + 'r = ' + expression);
  assert.deepEqual(result.diagnostics.filter((d: any) => d.severity === 'error'), []);
  return sampler.evaluateExpr(result.loweredModule.bindings.get('r').rhs,
    { __moduleRegistry: result.loweredModule.moduleRegistry });
}

// The irregular-knot fixtures are independently computed by SciPy CubicSpline
// (natural boundary conditions) and PchipInterpolator, not by this engine.
test('irregular interpolation matches independent cubic and PCHIP values', () => {
  const knots = '[-2,-.5,.3,2,4]', values = '[1,2,-.5,3,2]';
  const points = [-2, -1.2, -.5, 0, .3, 1.4, 3, 4];
  const expected = {
    cubic: [1, 2.455944259967245, 2, .30844691586712225, -.5, 1.2152631219909078, 3.5144193413713, 2],
    pchip: [1, 1.8983703703703703, 2, .291015625, -.5, 1.999796458375737, 2.875, 2],
  };
  for (const method of ['cubic', 'pchip'] as const) {
    points.forEach((x, i) => close(evaluate(`sp.interp_${method}(${x},${knots},${values},"error")`), expected[method][i]));
  }
  close(evaluate('sp.interp_linear(1.4,[-2,-.5,.3,2,4],[1,2,-.5,3,2],"error")'), 30 / 17);
});

test('interpolation endpoints, exterior policies and two-knot reduction are analytic', () => {
  for (const method of ['linear', 'cubic', 'pchip']) {
    close(evaluate(`sp.interp_${method}(-1,[0,2],[1,5])`), 1);
    close(evaluate(`sp.interp_${method}(record(x = 3, knots = [0,2], values = [1,5]))`), 5);
    for (const [x, expected] of [[-1, 1], [0, 1], [.5, 2], [2, 5], [3, 5]]) {
      close(evaluate(`sp.interp_${method}(${x},[0,2],[1,5],"constant")`), expected);
    }
    close(evaluate(`sp.interp_${method}(-1,[0,2],[1,5],"linear")`), -1);
    close(evaluate(`sp.interp_${method}(3,[0,2],[1,5],"linear")`), 7);
    assert.throws(() => evaluate(`sp.interp_${method}(-1,[0,2],[1,5],"error")`), /outside/);
  }
  // Natural second derivatives for (0,0),(1,1),(2,0) are (0,-3,0),
  // giving endpoint slopes +/-3/2; PCHIP instead gives +/-2.
  close(evaluate('sp.interp_cubic(-1,[0,1,2],[0,1,0],"linear")'), -1.5);
  close(evaluate('sp.interp_cubic(3,[0,1,2],[0,1,0],"linear")'), -1.5);
  close(evaluate('sp.interp_pchip(-1,[0,1,2],[0,1,0],"linear")'), -2);
  close(evaluate('sp.interp_pchip(3,[0,1,2],[0,1,0],"linear")'), -2);
});

test('module aliases, named rows and outer broadcasting retain complete knot vectors', async () => {
  const { ctx } = makeMatCtx(source + `
f = sp.interp_cubic
knots = [0,1,2]
values = [0,1,0]
y(x) = f(values = values, x = x, extrapolation = "error", knots = knots)
a = y.([0,.5,1,1.5,2])
b = f(record(values = values, extrapolation = "error", x = .5, knots = knots))
`, { sampleCount: 1 });
  const result = await ctx.getMeasure('a');
  [0, .6875, 1, .6875, 0].forEach((v, i) => close(result.samples[i], v));
  close((await ctx.getMeasure('b')).samples[0], .6875);
});

test('sampled interpolation values use each atom and gradients match closed forms', async () => {
  const { ctx } = makeMatCtx(source + `
x ~ Uniform(interval(0,2))
a ~ Normal(0,1)
ordinates ~ iid(Normal(0,1),3)
y = sp.interp_cubic(x,[0,1,2],[0,a,0],"error")
mode = "error"
f = sp.interp_cubic
named = f(extrapolation = mode, values = [0,a,0], x = x, knots = [0,1,2])
splatted = f(record(x = x, knots = [0,1,2], values = ordinates, extrapolation = mode))
literal_mode = f(record(x = x, knots = [0,1,2], values = ordinates, extrapolation = "error"))
`, { sampleCount: 24, rootSeed: 132 });
  const x = (await ctx.getMeasure('x')).samples, a = (await ctx.getMeasure('a')).samples;
  const y = (await ctx.getMeasure('y')).samples;
  const named = (await ctx.getMeasure('named')).samples;
  const splatted = (await ctx.getMeasure('splatted')).samples;
  const literalMode = (await ctx.getMeasure('literal_mode')).samples;
  const ordinates = (await ctx.getMeasure('ordinates')).value.data;
  for (let i = 0; i < x.length; i++) {
    const z = x[i] < 1 ? x[i] : 2 - x[i];
    close(y[i], a[i] * (1.5 * z - .5 * z ** 3));
    close(named[i], y[i]);
    const v = ordinates.slice(i * 3, i * 3 + 3), t = x[i] < 1 ? x[i] : x[i] - 1;
    const j = x[i] < 1 ? 0 : 1, second = 1.5 * (v[2] - 2 * v[1] + v[0]);
    const cubic = j === 0 ? t ** 3 - t : (1 - t) ** 3 - (1 - t);
    close(splatted[i], (1 - t) * v[j] + t * v[j + 1] + cubic * second / 6);
    close(literalMode[i], splatted[i]);
  }
  // JS's local optimizer differentiates by finite differences. The natural
  // interpolant on [0,1] is 3x/2-x^3/2, independent of the spline recurrence.
  const h = 1e-5, value = (x: number, a = 1) => evaluate(`sp.interp_cubic(${x},[0,1,2],[0,${a},0],"error")`);
  close((value(.4 + h) - value(.4 - h)) / (2 * h), 1.26, 2e-9);
  close((value(.4, 1 + h) - value(.4, 1 - h)) / (2 * h), .568, 2e-9);
  const movingKnot = (k: number) => evaluate(`sp.interp_linear(.4,[0,${k}],[0,1],"error")`);
  close((movingKnot(2 + h) - movingKnot(2 - h)) / (2 * h), -.1, 2e-9);
});

test('PCHIP preserves plateaus and monotonicity on irregular knots', () => {
  // The one-sided slope would be -3 and is limited to zero. The interior
  // harmonic slope is 9/5, hence Hermite interpolation at 1/2 gives 11/40.
  close(evaluate('sp.interp_pchip(.5,[0,1,2],[0,1,10],"error")'), 11 / 40);
  let previous = -Infinity;
  for (let i = 0; i <= 40; i++) {
    const x = i / 10, y = evaluate(`sp.interp_pchip(${x},[0,.2,1,4],[0,1,1,3],"error")`);
    assert.ok(y >= previous - 1e-14 && y >= 0 && y <= 3);
    if (x >= .2 && x <= 1) close(y, 1);
    previous = y;
  }
});

test('interpolation rejects repeated abscissae and mismatched ordinates', () => {
  for (const args of ['0,[0,0],[1,2],"error"', '0,[0,1],[1],"error"',
    '0,[1,0],[1,2],"error"', '0,[0,inf],[1,2],"error"', '0,[0,1],[1,2],"unknown"']) {
    assert.throws(() => evaluate(`sp.interp_linear(${args})`), /splines:/);
  }
});
