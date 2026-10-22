'use strict';

// A bare `normalize(M)` that reaches the density walker has to know Z = totalmass(M):
// density(normalize(M), x) = density(M, x) - log Z. The walker used to resolve Z for
// a truncation, a weighting and a mixture only, and scored every other measure with
// Z = 1 — `normalize(joint(weighted(2, Normal(0, 1)), weighted(3, Normal(1, 2))))` came
// out log 6 too high, and so did a pushfwd, locscale, relabel or record around a
// weighted base. A Z it cannot resolve is now refused, naming the measure and its
// source position.
//
// ORACLE: closed form. Every weighted base below is a probability measure scaled by
// a constant, so the normalised density is the probability measure's own density —
// log N(x; m, s) = -log(s sqrt(2 pi)) - ((x - m) / s)^2 / 2 — and a raw (unnormalised)
// score exceeds it by exactly log(weight).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const density = require('../density.ts');
const { parseSetIR } = require('../ir-shared.ts');

const logN = (x: number, m = 0, s = 1) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;

function near(got: number, want: number, tol = 1e-12) {
  assert.ok(Math.abs(got - want) < tol, `got ${got}, want ${want}`);
}

const lit = (value: any) => ({ kind: 'lit', value });
const call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });
const normal = (m: number, s: number) => call('Normal', lit(m), lit(s));
const weighted = (w: number, m: any) => call('weighted', lit(w), m);
const opts = { parseSet: (set: any) => parseSetIR(set, null), resolveMeasureRef: () => null };
const score = (ir: any, point: any) => density.logDensity(ir, point, {}, opts);

async function batched(measure: string, pts: string) {
  const { ctx } = ctxFor(`M = ${measure}\nlp = broadcast(logdensityof, M, ${pts})\n`, 1);
  const m = await ctx.getMeasure('lp');
  return Array.from(m.value ? m.value.data : m.samples) as number[];
}

test('a joint of weighted parts divides by the product of the weights', async () => {
  const want = [logN(0.5) + logN(1, 1, 2), logN(1) + logN(-1, 1, 2)];
  const joint = await batched('normalize(joint(weighted(2.0, Normal(0, 1)), weighted(3.0, Normal(1, 2))))',
    '[[0.5, 1.0], [1.0, -1.0]]');
  joint.forEach((got, i) => near(got, want[i]));
  const nested = await batched('normalize(weighted(2.0, joint(Normal(0, 1), Normal(1, 2))))',
    '[[0.5, 1.0], [1.0, -1.0]]');
  nested.forEach((got, i) => near(got, want[i]));
});

test('a record joint divides by the product of the weights', async () => {
  const got = await batched('normalize(joint(a = weighted(2.0, Normal(0, 1)), b = Normal(1, 2)))',
    '[record(a = 0.5, b = 1.0), record(a = 1.0, b = -1.0)]');
  near(got[0], logN(0.5) + logN(1, 1, 2));
  near(got[1], logN(1) + logN(-1, 1, 2));
});

test('an iid plate of a weighted measure divides by weight^n', async () => {
  // iid(Exponential(1), 2) has density exp(-x1 - x2).
  const got = await batched('normalize(weighted(2.0, iid(Exponential(1.0), 2)))',
    '[[0.5, 1.0], [1.0, 2.0]]');
  near(got[0], -1.5);
  near(got[1], -3);
});

test('a weighted Dirac normalises to mass 1 at its atom and -Infinity elsewhere', async () => {
  const got = await batched('normalize(weighted(2.0, Dirac(1.0)))', '[1.0, 2.0]');
  near(got[0], 0);
  assert.equal(got[1], -Infinity);
});

test('a weight that is a function of the variate is still normalised by its integral', async () => {
  // fn(exp(-_)) over [0, 2] integrates to 1 - exp(-2): the density at x is exp(-x) / (1 - exp(-2)).
  const got = await batched('normalize(weighted(fn(exp(-_)), Lebesgue(interval(0, 2))))', '[1.0, 1.5]');
  // The integral is an adaptive quadrature (relative tolerance 1e-8), not a closed form.
  near(got[0], -1 - Math.log(1 - Math.exp(-2)), 1e-7);
  near(got[1], -1.5 - Math.log(1 - Math.exp(-2)), 1e-7);
});

test('pushfwd, locscale and relabel keep the mass of their base', () => {
  const exp = { ...call('pushfwd', { kind: 'ref', ns: 'self', name: 'f' }, weighted(2, normal(0, 1))),
    bijection: { fInv: { body: call('log', { kind: 'ref', ns: '%local', name: 'y' }), paramName: 'y' },
      logVolume: { kind: 'fn', body: { kind: 'ref', ns: '%local', name: 'x' }, paramName: 'x' } } };
  // LogNormal(0, 1): log N(log y) - log y.
  near(score(call('normalize', exp), 1.5), logN(Math.log(1.5)) - Math.log(1.5));
  near(score(exp, 1.5), Math.log(2) + logN(Math.log(1.5)) - Math.log(1.5));
  near(score(call('normalize', call('locscale', weighted(2, normal(0, 1)), lit(1), lit(3))), 2.0),
    logN(2, 1, 3));
  near(score(call('normalize', call('relabel', weighted(2, normal(0, 1)), call('vector', lit('x')))), 0.5),
    logN(0.5));
});

test('draw, lawof and a nested normalize keep the mass of their base', () => {
  near(score(call('normalize', call('draw', weighted(2, normal(0, 1)))), 0.5), logN(0.5));
  near(score(call('normalize', call('lawof', weighted(2, normal(0, 1)))), 0.5), logN(0.5));
  near(score(call('normalize', call('normalize', weighted(2, normal(0, 1)))), 0.5), logN(0.5));
});

test('a record of a weighted part divides by its weight, with the parts named or positional', () => {
  const field = (name: string, value: any) => ({ name, value });
  const fields = { kind: 'call', op: 'joint', args: [],
    fields: [field('a', weighted(2, normal(0, 1))), field('b', normal(1, 2))] };
  near(score(call('normalize', fields), { a: 0.5, b: 1.0 }), logN(0.5) + logN(1, 1, 2));
  near(score(call('normalize', call('joint', weighted(2, normal(0, 1)), normal(1, 2))), [0.5, 1.0]),
    logN(0.5) + logN(1, 1, 2));
});

test('a mixture of mixtures resolves its mass through the nested weights', () => {
  // 0.25 N(0, 1) + 0.75 N(1, 2) carries mass 1 inside a select; the outer weight is 4.
  const inner = { kind: 'call', op: 'select', logweights: null, branches: [
    call('logweighted', lit(Math.log(0.25)), normal(0, 1)), call('logweighted', lit(Math.log(0.75)), normal(1, 2))] };
  const mix = (x: number) => Math.log(0.25 * Math.exp(logN(x)) + 0.75 * Math.exp(logN(x, 1, 2)));
  near(score(call('normalize', weighted(4, inner)), 0.5), mix(0.5));
});

function refusal(ir: any, point: any, message: RegExp) {
  assert.throws(() => score(ir, point), (err: any) => {
    assert.match(err.message, message);
    assert.equal(err.modelRefusal, true);
    return true;
  });
}

test('a measure whose mass is not known is refused, naming it and its position', () => {
  const at = (ir: any) => ({ ...ir, loc: { start: { line: 4, col: 6 }, end: { line: 4, col: 30 } } });
  // §09: ContinuedPoisson is not normalised, so normalize needs its quadrature mass.
  refusal(at(call('normalize', call('ContinuedPoisson', lit(3)))), 1, /normalize\(ContinuedPoisson\) at 5:7 — its total mass Z is not known/);
  refusal(call('normalize', call('joint', weighted(2, normal(0, 1)), call('ContinuedPoisson', lit(3)))),
    [0.5, 1], /normalize\(joint\)/);
  refusal(call('normalize', call('iid', call('ContinuedPoisson', lit(3)), lit(2))), [1, 2], /normalize\(iid\)/);
  refusal(call('normalize', { kind: 'call', target: { ns: 'm', name: 'Mystery' }, args: [] }), 1, /normalize\(Mystery\)/);
  refusal(call('normalize', call('Lebesgue', call('interval', lit(0), lit(2)))), 1, /normalize\(Lebesgue\)/);
  refusal(call('normalize', call('weighted', call('functionof', lit(1)), normal(0, 1))), 0.5, /normalize\(weighted\)/);
  refusal(call('normalize', weighted(2, { kind: 'ref', ns: 'self', name: 'q' })), 0.5, /normalize\(weighted\)/);
});

test('a probability leaf is its own normalisation', () => {
  near(score(call('normalize', normal(1, 2)), 0.5), logN(0.5, 1, 2));
});

test('a truncation whose bounds follow a per-atom parameter is normalised atom by atom', () => {
  // Normal(theta, 1) restricted to [0, 5] has mass Phi(5 - theta) - Phi(-theta).
  const phi = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2));
  const theta = { kind: 'ref', ns: 'self', name: 'theta' };
  const ir = call('normalize', call('truncate', call('Normal', theta, lit(1)),
    call('interval', lit(0), lit(5))));
  const thetas = [0.0, 1.5];
  const got = density.logDensityN(ir, 1.0, { theta: thetas }, 2, opts);
  thetas.forEach((t, i) => near(got[i], logN(1, t, 1) - Math.log(phi(5 - t) - phi(-t)), 1e-9));
});

// Abramowitz-Stegun 7.1.26 is too coarse for 1e-9; use the series for |x| < 4 and the
// continued fraction beyond.
function erf(x: number): number {
  const ax = Math.abs(x);
  let value: number;
  if (ax < 4) {
    let term = ax; let sum = ax;
    for (let n = 1; n < 200; n++) { term *= -ax * ax / n; sum += term / (2 * n + 1); }
    value = 2 / Math.sqrt(Math.PI) * sum;
  } else {
    let f = 0;
    for (let k = 60; k >= 1; k--) f = k / 2 / (ax + f);
    value = 1 - Math.exp(-ax * ax) / Math.sqrt(Math.PI) / (ax + f);
  }
  return x < 0 ? -value : value;
}

test('a truncation whose bound comes from the environment is normalised at that value', () => {
  const bound = { kind: 'ref', ns: 'self', name: 'b' };
  const ir = call('normalize', call('truncate', normal(0, 1), call('interval', lit(0), bound)));
  const phi = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2));
  near(density.logDensity(ir, 1.0, { b: 2 }, opts), logN(1) - Math.log(phi(2) - 0.5), 1e-9);
});

test('normalizers of a measure held constant over many points are resolved once', () => {
  const hoisted = density.hoistNormalizers(call('normalize', weighted(2, normal(0, 1))), opts, {}, ['x']);
  assert.equal(hoisted.op, 'logweighted');
  assert.equal(hoisted.fromNormalize, true);
  near(hoisted.args[0].value, -Math.log(2));
  near(score(hoisted, 0.5), logN(0.5));
  // Inside a joint and under a weight, the nested normalizer is resolved too; the joint's
  // own normalizer needs Z(joint) = 1 * 2, and the outer weight 3 cancels.
  const nested = call('normalize', weighted(3, call('joint', call('normalize', weighted(2, normal(0, 1))), normal(1, 2))));
  const flat = density.hoistNormalizers(nested, opts, {}, ['x']);
  assert.equal(flat.op, 'logweighted');
  near(score(flat, [0.5, 1.0]), logN(0.5) + logN(1, 1, 2));
  const inner = flat.args[1].args[1].args[0];
  assert.equal(inner.op, 'logweighted');
});

test('a normalizer whose mass is not known is left for the walker to refuse', () => {
  const unknown = call('normalize', call('ContinuedPoisson', lit(3)));
  assert.equal(density.hoistNormalizers(unknown, opts, {}, ['x']), unknown);
  const wrapped = weighted(2, unknown);
  assert.equal(density.hoistNormalizers(wrapped, opts, {}, ['x']), wrapped);
});

test('a normalizer that follows the broadcast parameter keeps its per-point mass', () => {
  const rate = { kind: 'ref', ns: '%local', name: 'rate' };
  const dependent = call('normalize', weighted(2, call('Normal', rate, lit(1))));
  assert.equal(density.hoistNormalizers(dependent, opts, {}, ['rate']), dependent);
  // The same measure with the parameter bound elsewhere is constant and is resolved.
  assert.equal(density.hoistNormalizers(dependent, opts, {}, ['other']).op, 'logweighted');
});

test('a measure operand that cannot be resolved is left as written', () => {
  const unresolved = weighted(2, { kind: 'ref', ns: 'self', name: 'q' });
  assert.equal(density.hoistNormalizers(unresolved, opts, {}, ['x']), unresolved);
});
