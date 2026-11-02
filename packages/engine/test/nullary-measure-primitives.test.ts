'use strict';

// The nullary form of the measure-kernel evaluation primitives (spec §07
// §sec:measure-eval-prims): a measure is a nullary kernel, so
// `builtin_logdensityof(M, x)`, `builtin_sample(rngstate, M, n, …)` and the
// transports omit `kernel_input`.
//
// Oracles are scipy.stats values (scipy 1.x), computed outside the engine:
//   norm(1, 2).logpdf(0.5)                          = -1.643335713764618
//   norm(1, 2).logpdf(1.0)                          = -1.612085713764618
//   expon(scale=0.5).cdf(0.5)                       =  0.6321205588285577
//   norm(1, 2).ppf(norm.cdf(0.5))                   =  2.0
//   poisson(3).logpmf(2)                            = -1.4959226032237258
//   multivariate_normal([0, 1], [[2, .5], [.5, 1]])
//     .logpdf([0.3, 0.7])                           = -2.2205421032341994
//   crystalball(beta=1.5, m=3, loc=0, scale=1)
//     .logpdf(0.5)                                  = -1.1047611536741122
//     .logpdf(-2.5)                                 = -3.3211564779986054

const { test } = require('node:test');
const assert = require('node:assert/strict');

const engine = require('../index.ts');
const orchestrator = require('../orchestrator.ts');

function evaluate(src: string): Map<string, any> {
  const r = engine.processSource(`flatppl_compat = "0.1"\n${src}`);
  const errors = r.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
  assert.deepEqual(errors, [], src);
  return orchestrator.buildDerivations(r.bindings, r.loweredModule).fixedValues;
}

function errorsOf(src: string): string[] {
  const r = engine.processSource(`flatppl_compat = "0.1"\n${src}`);
  return r.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

function numbers(v: any): number[] {
  return Array.isArray(v) ? v : Array.from(v.data);
}

function close(actual: number, expected: number, what: string) {
  assert.ok(Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)),
    `${what}: ${actual} vs ${expected}`);
}

test('a closed built-in measure scores like its kernel form', () => {
  const cases: [string, number][] = [
    ['lp = builtin_logdensityof(Normal(1.0, 2.0), 0.5)', -1.643335713764618],
    ['lp = builtin_logdensityof(Normal(mu = 1.0, sigma = 2.0), 0.5)', -1.643335713764618],
    ['lp = builtin_logdensityof(N, 0.5)\nM = Normal(1.0, 2.0)\nN = M', -1.643335713764618],
    ['lp = builtin_logdensityof(Poisson(3.0), 2)', -1.4959226032237258],
    ['lp = builtin_logdensityof(MvNormal([0.0, 1.0], rowstack([[2.0, 0.5], [0.5, 1.0]])), [0.3, 0.7])',
      -2.2205421032341994],
    ['hep = standard_module("hepphys", "0.1")\nlp = builtin_logdensityof(hep.CrystalBall(0.0, 1.0, 1.5, 3.0), 0.5)',
      -1.1047611536741122],
    ['hep = standard_module("hepphys", "0.1")\nlp = builtin_logdensityof(hep.CrystalBall(0.0, 1.0, 1.5, 3.0), -2.5)',
      -3.3211564779986054],
    ['lp = builtin_touniform(Exponential(2.0), 0.5)', 0.6321205588285577],
    ['lp = builtin_fromnormal(Normal(1.0, 2.0), 0.5)', 2.0],
  ];
  for (const [src, expected] of cases) close(evaluate(src).get('lp'), expected, src);
});

test('a broadcast holds the measure constant', () => {
  const lp = numbers(evaluate('lp = builtin_logdensityof.(Normal(1.0, 2.0), [0.5, 1.0])').get('lp'));
  close(lp[0], -1.643335713764618, 'cell 0');
  close(lp[1], -1.612085713764618, 'cell 1');
  const u = numbers(evaluate('u = builtin_touniform.(Exponential(2.0), [0.5, 0.5])').get('u'));
  close(u[1], 0.6321205588285577, 'transport cell');
});

test('a nullary sample draws what the kernel form draws', () => {
  const values = evaluate(`
state = rnginit([0, 1, 2, 3])
M = Exponential(2.0)
a, sa = builtin_sample(state, M, 5)
b, sb = builtin_sample(state, Exponential, record(rate = 2.0), 5)
c, sc = builtin_sample(sa, Normal(1.0, 2.0))
d, sd = builtin_sample(sb, Normal, record(mu = 1.0, sigma = 2.0))
`);
  assert.deepEqual(numbers(values.get('a')), numbers(values.get('b')));
  assert.equal(values.get('c'), values.get('d'));
  assert.deepEqual(values.get('sc'), values.get('sd'));
});

test('a measure-input kernel and a composite measure stay refused', () => {
  assert.match(
    errorsOf('lp = builtin_logdensityof(PoissonProcess(Normal(0.0, 1.0)), [0.1, 0.2])').join('\n'),
    /takes a measure input/);
  assert.match(
    errorsOf('lp = builtin_logdensityof(iid(Normal(0.0, 1.0), 2), [0.1, 0.2])').join('\n'),
    /or a closed built-in measure such as Normal\(0, 1\)/);
});
