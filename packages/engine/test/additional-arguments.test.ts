'use strict';

// Spec §04 "Additional arguments" (flatppl-design #216): an operation that supplies the
// leading arguments of a callable passes the call's remaining arguments to the callable's
// remaining inputs, positionally or by name, whole. A sole additional record or table
// passes whole if one positional input remains, otherwise its fields or columns bind the
// remaining inputs by name. No input may be left unbound or bound twice.
//
// ORACLES are closed forms. weighted(w, Normal(0, 1), ...) has the log density
// log w(x, ...) + log N(x) at x; logweighted adds the weight; the pushforward of
// Normal(0, 1) through y = p exp(x) has log N(log(y / p)) - log y.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const logN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;

async function value(src: string, name: string): Promise<number[]> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure(name);
  return Array.from(m.value ? m.value.data : m.samples);
}

function errors(src: string): string[] {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

function near(got: number[], want: number[], tol = 1e-12) {
  assert.equal(got.length, want.length);
  want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) < tol, `got ${got[i]}, want ${w}`));
}

const W = 'w(x, p, q) = p * x * x + q\n';
const AT = 0.5;
const WANT = Math.log(3 * AT * AT + 1) + logN(AT);

test('weighted: two additional arguments bind positionally, by name, or from a record', async () => {
  for (const extras of ['3.0, 1.0', 'p = 3.0, q = 1.0', 'q = 1.0, p = 3.0', 'record(p = 3.0, q = 1.0)']) {
    near(await value(`${W}m = weighted(w, Normal(0, 1), ${extras})\nlp = logdensityof(m, ${AT})\n`, 'lp'), [WANT]);
  }
});

test('weighted: a sole record passes whole when one positional input remains', async () => {
  // w(x, p) = x * p.a with p = record(a = 2): the weight at 0.5 is 1.
  near(await value('w(x, p) = x * p.a\nm = weighted(w, Normal(0, 1), record(a = 2.0))\nlp = logdensityof(m, 0.5)\n', 'lp'),
    [logN(0.5)]);
});

test('weighted: a sole table splats its columns by name', async () => {
  const src = 'w(x, p, q) = p[1] * x * x + q[1]\nt = table(p = [3.0], q = [1.0])\n'
    + 'm = weighted(w, Normal(0, 1), t)\nlp = logdensityof(m, 0.5)\n';
  near(await value(src, 'lp'), [WANT]);
});

test('logweighted: the callback gets the same additional arguments', async () => {
  const lw = 'lw(x, p, q) = p * x + q\n';
  const want = 2.0 * AT + 1.0 + logN(AT);
  for (const extras of ['2.0, 1.0', 'q = 1.0, p = 2.0', 'record(p = 2.0, q = 1.0)']) {
    near(await value(`${lw}m = logweighted(lw, Normal(0, 1), ${extras})\nlp = logdensityof(m, ${AT})\n`, 'lp'), [want]);
  }
});

test('an unbound, unknown or doubly bound input is a static error', () => {
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), p = 3.0)\n`).some(m => /input "q" is not bound/.test(m)));
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), p = 3.0, q = 1.0, r = 2.0)\n`).some(m => /no remaining input named "r"/.test(m)));
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), x = 3.0, p = 1.0, q = 1.0)\n`).some(m => /no remaining input named "x"/.test(m)));
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), 3.0)\n`).some(m => /callback receives 2 whole inputs, but declares 3/.test(m)));
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), 3.0, q = 1.0)\n`).length > 0);
});

test('the leading inputs of weighted stay positional', () => {
  assert.ok(errors(`${W}m = weighted(weight = w, base = Normal(0, 1), p = 3.0, q = 1.0)\n`)
    .some(m => /never by keyword argument/.test(m)));
});

test('pushfwd: additional arguments reach the function, its inverse and its log-volume', async () => {
  const bij = 'f(x, p) = p * exp(x)\nfi(y, p) = log(y / p)\nlv(x, p) = x + log(p)\nb = bijection(f, fi, lv)\n';
  const want = logN(Math.log(3 / 2)) - Math.log(3);
  for (const extras of ['2.0', 'p = 2.0']) {
    const src = `${bij}m = pushfwd(b, Normal(0, 1), ${extras})\nlp = logdensityof(m, 3.0)\n`;
    assert.deepEqual(errors(src), []);
    near(await value(src, 'lp'), [want]);
  }
  near(await value('g(x, p) = p * exp(x)\nm = pushfwd(g, Normal(0, 1), p = 2.0)\nlp = logdensityof(m, 3.0)\n', 'lp'), [want]);
});

test('reduce: additional arguments follow the two inputs and stay fixed in every call', async () => {
  const f = 'f(a, b, p, q) = a + b * p + q\n';
  // ((1 + 2*2 + 1) + 3*2 + 1) = 13
  for (const extras of ['2.0, 1.0', 'p = 2.0, q = 1.0', 'record(p = 2.0, q = 1.0)']) {
    assert.deepEqual(await value(`${f}r = reduce(f, [1.0, 2.0, 3.0], ${extras})\n`, 'r'), [13]);
  }
  assert.deepEqual(await value('r = reduce(add, [10, 20] .* [2, 3])\n', 'r'), [80]);
  assert.deepEqual(await value('r = reduce(add, [7.0])\n', 'r'), [7]);
  assert.ok(errors(`${f}r = reduce(f, [1.0, 2.0], p = 2.0)\n`).some(m => /input "q" is not bound/.test(m)));
});

// ---------------------------------------------------------------------------
// scan and filter: the additional arguments are sequences traversed together
// ---------------------------------------------------------------------------

const STEP = 'step(acc, x, w) = acc + x * w\n';

test('scan: sequences are traversed together, positionally or by name', async () => {
  // acc: 0 + 10*2 = 20, then 20 + 20*3 = 80.
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, [10, 20], [2, 3])\n`, 'r'), [20, 80]);
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, w = [2, 3], x = [10, 20])\n`, 'r'), [20, 80]);
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, record(x = [10, 20], w = [2, 3]))\n`, 'r'), [20, 80]);
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, table(x = [10, 20], w = [2, 3]))\n`, 'r'), [20, 80]);
  // reduce over the elementwise products is the final accumulator.
  assert.deepEqual(await value('r = reduce(add, [10, 20] .* [2, 3])\n', 'r'), [80]);
});

test('scan: a length-one sequence repeats, an empty one with length-one inputs gives no steps', async () => {
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, [10, 20], [2])\n`, 'r'), [20, 60]);
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, x = [10, 20], w = [2])\n`, 'r'), [20, 60]);
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, [10], [2])\n`, 'r'), [20]);
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, fill(0, 0), [2])\n`, 'r'), []);
  // A one-row table supplies the same record at every step.
  assert.deepEqual(await value('g(acc, x, row) = acc + x * row.w\nr = scan(g, 0, [10, 20], table(w = [2]))\n', 'r'), [20, 60]);
});

test('scan: one sequence keeps its meaning, unequal lengths and a missing sequence are errors', async () => {
  assert.deepEqual(await value('s1(acc, x) = acc + x\nr = scan(s1, 0, [1, 2, 3])\n', 'r'), [1, 3, 6]);
  assert.ok(errors(`${STEP}r = scan(step, 0, [10, 20, 30], [2, 3])\n`).some(m => /unequal lengths 3, 2/.test(m)));
  assert.ok(errors(`${STEP}r = scan(step, 0)\n`).some(m => /at least three arguments/.test(m)));
  assert.ok(errors(`${STEP}r = scan(step, 0, x = [1, 2], v = [1, 2])\n`).some(m => /no remaining input named "v"/.test(m)));
  assert.ok(errors(`${STEP}r = scan(step, 0, [1, 2], w = [1, 2])\n`).some(m => /all positional or all named/.test(m)));
});

const WITHIN = 'within(x, low, high) = low <= x <= high\n';

test('filter: additional inputs are traversed with the data, length-one inputs repeat', async () => {
  // 1 in [0, 10], 4 below its low bound 5, 9 in [8, 10].
  assert.deepEqual(await value(`${WITHIN}r = filter(within, [1, 4, 9], low = [0, 5, 8], high = [10])\n`, 'r'), [1, 9]);
  assert.deepEqual(await value(`${WITHIN}r = filter(within, [1, 4, 9], [0, 5, 8], [10])\n`, 'r'), [1, 9]);
  assert.deepEqual(await value('keep(x, p) = x > p\nr = filter(keep, [1.0, 4.0, 9.0], [3.0])\n', 'r'), [4, 9]);
  assert.deepEqual(await value('keep(x, p) = x > p\nr = filter(keep, [1.0, 4.0, 9.0], [1.0, 5.0, 8.0])\n', 'r'), [9]);
  assert.deepEqual(await value(`${WITHIN}r = filter(within, [1, 4, 9], table(low = [0, 5, 8], high = [10, 10, 10]))\n`, 'r'), [1, 9]);
});

test('filter: a table is filtered by rows and empty data stays empty', async () => {
  const src = 'above(row, p) = row.value > p\nt = table(value = [1, 2, 3], other = [10, 20, 30])\n'
    + 'kept = filter(above, t, [1])\nr = kept.other\n';
  assert.deepEqual(await value(src, 'r'), [20, 30]);
  assert.deepEqual(await value('keep(x, p) = x > p\nr = filter(keep, fill(1.0, 0), [3.0])\n', 'r'), []);
});

test('filter: an additional input of the wrong length is an error', async () => {
  assert.ok(errors('keep(x, p) = x > p\nr = filter(keep, [1.0, 4.0, 9.0], [3.0, 5.0])\n').some(m => /unequal lengths 3, 2/.test(m)));
});

// ---------------------------------------------------------------------------
// markovchain and kscan
// ---------------------------------------------------------------------------

test('markovchain: the spec spellings of the additional arguments give one trajectory law', async () => {
  // §05: positional, named, and a sole record are the same call.
  const step = 'step(state, drift, scale) = Normal(state + drift, scale)\ndrift = 0.1\nscale = 0.5\n';
  const path = [0.3, 0.1, 0.9];
  // Normal(state + drift, scale) at each step, from init 0.
  let prev = 0, want = 0;
  for (const x of path) {
    const z = (x - prev - 0.1) / 0.5;
    want += -0.5 * Math.log(2 * Math.PI) - Math.log(0.5) - 0.5 * z * z;
    prev = x;
  }
  for (const extras of ['drift, scale', 'drift = drift, scale = scale', '(drift = drift, scale = scale)']) {
    const src = `${step}M = markovchain(step, 0.0, 3, ${extras})\nlp = logdensityof(M, [${path.join(', ')}])\n`;
    assert.deepEqual(errors(src), []);
    near(await value(src, 'lp'), [want]);
  }
});

test('markovchain: a kernel input left unbound or given too many arguments is refused', () => {
  const step = 'step(state, drift, scale) = Normal(state + drift, scale)\n';
  assert.ok(errors(`${step}M = markovchain(step, 0.0, 3, 0.1)\n`).length > 0);
  assert.ok(errors(`${step}M = markovchain(step, 0.0, 3, drift = 0.1)\n`).some(m => /scale/.test(m)));
  assert.ok(errors(`${step}M = markovchain(step, 0.0, 3, 0.1, 0.5, 2.0)\n`).length > 0);
});

const KS = 'kd(s, x, w) = Dirac(value = s + x * w)\n';

test('kscan: sequences are traversed together, positionally or by name, and singletons repeat', async () => {
  // s: 0 + 10*2 = 20, then 20 + 20*3 = 80.
  assert.deepEqual(await value(`${KS}r ~ kscan(kd, 0.0, [10.0, 20.0], [2.0, 3.0])\n`, 'r'), [20, 80]);
  assert.deepEqual(await value(`${KS}r ~ kscan(kd, 0.0, x = [10.0, 20.0], w = [2.0, 3.0])\n`, 'r'), [20, 80]);
  assert.deepEqual(await value(`${KS}r ~ kscan(kd, 0.0, [10.0, 20.0], [2.0])\n`, 'r'), [20, 60]);
  assert.deepEqual(await value('k1(s, x) = Dirac(value = s + x)\nr ~ kscan(k1, 0.0, [10.0, 20.0])\n', 'r'), [10, 30]);
});

test('kscan: the density is the product of the per-step transitions', async () => {
  const src = 'k2(s, x, w) = Normal(s + x, w)\nM = kscan(k2, 0.0, [1.0, 2.0], [0.5, 0.7])\n'
    + 'lp = logdensityof(M, [0.6, 2.9])\n';
  const step = (x: number, mean: number, sd: number) => -0.5 * Math.log(2 * Math.PI) - Math.log(sd)
    - 0.5 * ((x - mean) / sd) ** 2;
  near(await value(src, 'lp'), [step(0.6, 0 + 1.0, 0.5) + step(2.9, 0.6 + 2.0, 0.7)]);
});

test('kscan: a scalar held constant gives the same density as its repeated sequence', async () => {
  const step = (x: number, mean: number, sd: number) => -0.5 * Math.log(2 * Math.PI) - Math.log(sd)
    - 0.5 * ((x - mean) / sd) ** 2;
  const src = 'k2(s, x, w) = Normal(s + x, w)\nM = kscan(k2, 0.0, [1.0, 2.0], 0.5)\n'
    + 'lp = logdensityof(M, [0.6, 2.9])\n';
  near(await value(src, 'lp'), [step(0.6, 0 + 1.0, 0.5) + step(2.9, 0.6 + 2.0, 0.5)]);
});

test('kscan: unequal sequence lengths and a missing sequence are refused', () => {
  assert.ok(errors(`${KS}r ~ kscan(kd, 0.0, [1.0, 2.0, 3.0], [1.0, 2.0])\n`).some(m => /unequal lengths 3, 2/.test(m)));
  assert.ok(errors(`${KS}r ~ kscan(kd, 0.0)\n`).some(m => /at least one array or table/.test(m)));
  assert.ok(errors(`${KS}r ~ kscan(kd, 0.0, [1.0, 2.0], x = [1.0, 2.0])\n`).some(m => /all positional or all named/.test(m)));
});

// ---------------------------------------------------------------------------
// fchain: the additional arguments reach every step
// ---------------------------------------------------------------------------

test('fchain: additional arguments are passed unchanged to every step', async () => {
  // f3(f2(f1(x, p), p), p) with f1 = x + p, f2 = y * p, f3 = z - p at x = 1, p = 2: ((1+2)*2)-2.
  const steps = 'a1(x, p) = x + p\na2(y, p) = y * p\na3(z, p) = z - p\n';
  for (const call of ['F(1.0, 2.0)', 'F(x = 1.0, p = 2.0)', 'fchain(a1, a2, a3)(1.0, 2.0)']) {
    const src = `${steps}F = fchain(a1, a2, a3)\nr = ${call}\n`;
    assert.deepEqual(errors(src), []);
    assert.deepEqual(await value(src, 'r'), [4]);
  }
  // Two additional inputs: (1 + 2 + 3) * 2 - 3 = 9.
  const two = 'a1(x, p, q) = x + p + q\na2(y, p, q) = y * p - q\nF = fchain(a1, a2)\nr = F(1.0, 2.0, 3.0)\n';
  assert.deepEqual(errors(two), []);
  assert.deepEqual(await value(two, 'r'), [9]);
});

test('fchain: a single argument keeps its meaning', async () => {
  assert.deepEqual(await value('a1(x) = x + 1.0\na2(y) = y * 2.0\nF = fchain(a1, a2)\nr = F(1.0)\n', 'r'), [4]);
});

test('fchain: a step with a different number of inputs than the chain has is still refused', () => {
  const src = 'a1(x, p) = x + p\na2(y, p, q) = y * p * q\nF = fchain(a1, a2)\nr = F(1.0, 2.0)\n';
  assert.ok(errors(src).some(m => /fchain step boundary 0 → 1/.test(m)));
});

test('scan, kscan and filter hold a scalar input constant at every step', async () => {
  // §04: arrays and tables are traversed together; scalars repeat unchanged.
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, [10, 20], 2)\n`, 'r'), [20, 60]);
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, 2, [10, 20])\n`, 'r'), [20, 60]);
  // A constant beside two traversed arrays: step i adds x_i * w_i + b.
  assert.deepEqual(await value('f(acc, x, w, b) = acc + x * w + b\nr = scan(f, 0, [10, 20], [2, 3], 1)\n', 'r'), [21, 82]);
  assert.deepEqual(await value('keep(x, p) = x > p\nr = filter(keep, [1.0, 4.0, 9.0], 3.0)\n', 'r'), [4, 9]);
  assert.deepEqual(await value('kd(s, x, w) = Dirac(value = s + x * w)\nw = 3.0\nr ~ kscan(kd, 0.0, [1.0, 2.0], w)\n', 'r'), [3, 9]);
  // A one-row table supplies the same record at every step.
  assert.deepEqual(await value('g(acc, x, row) = acc + x * row.w\nr = scan(g, 0, [10, 20], table(w = [2]))\n', 'r'), [20, 60]);
  // A parametric constant stays live in a scored callable.
  const live = 'p = elementof(reals)\nkeep(x, q) = x > q\nkept = filter(keep, [1.0, 2.0, 3.0], p)\nM = Normal(sum(kept), 1.0)\n'
    + 'score = functionof(logdensityof(M, 0.0), p)\na = score(0.5)\nb = score(2.5)\n';
  // p = 0.5 keeps 1, 2, 3 (sum 6); p = 2.5 keeps 3 (sum 3): log N(0; sum, 1).
  near(await value(live, 'a'), [-0.5 * Math.log(2 * Math.PI) - 0.5 * 36]);
  near(await value(live, 'b'), [-0.5 * Math.log(2 * Math.PI) - 0.5 * 9]);
});

test('a vector of vectors is traversed by element, and [A] supplies A at every step', async () => {
  const H = 'h(acc, x, v) = acc + x * sum(v)\nA = [1.0, 2.0]\n';
  // [A] repeats A: 0 + 10 * 3, then + 20 * 3.
  assert.deepEqual(await value(`${H}r = scan(h, 0, [10, 20], [A])\n`, 'r'), [30, 90]);
  // Two vectors are two elements: 0 + 10 * 3, then + 20 * 7.
  assert.deepEqual(await value(`${H}r = scan(h, 0, [10, 20], [A, [3.0, 4.0]])\n`, 'r'), [30, 170]);
  assert.deepEqual(await value('hit(x, v) = x > sum(v)\nA = [1.0, 2.0]\nr = filter(hit, [1.0, 4.0, 9.0], [A])\n', 'r'), [4, 9]);
});

test('a record, a tuple, a higher-dimensional array, or no array or table input is a static error', () => {
  const KD = 'kd(s, x, w) = Dirac(value = s + x * w)\n';
  const KEEP = 'keep(x, p) = x > p\n';
  const MATRIX = 'M = rowstack([[1.0, 2.0], [3.0, 4.0]])\n';
  const tuple = /an additional input is a tuple/;
  const record = /an additional input is a record/;
  const rank = /higher-dimensional array/;
  const none = /at least one array or table input/;
  for (const [src, expected] of [
    [`${STEP}r = scan(step, 0, [10, 20], (1.0, 2.0))\n`, tuple],
    ['g(acc, x, row) = acc + x * row.w\nr = scan(g, 0, [10, 20], record(w = 2))\n', record],
    ['kd(s, x, row) = Dirac(value = s + x * row.w)\nr ~ kscan(kd, 0.0, [1.0, 2.0], record(w = 2))\n', record],
    ['keep(x, row) = x > row.w\nr = filter(keep, [1.0, 4.0], record(w = 2))\n', record],
    ['g(acc, x, w, b) = acc + x * w + b\nr = scan(g, 0, [10, 20], record(w = 2, b = 1))\n', record],
    [`${KD}r ~ kscan(kd, 0.0, [1.0, 2.0], (1.0, 2.0))\n`, tuple],
    [`${KEEP}r = filter(keep, [1.0, 4.0], (1.0, 2.0))\n`, tuple],
    [`${STEP}${MATRIX}r = scan(step, 0, [10, 20], M)\n`, rank],
    [`${KD}${MATRIX}r ~ kscan(kd, 0.0, [1.0, 2.0], M)\n`, rank],
    [`${KEEP}${MATRIX}r = filter(keep, [1.0, 4.0], M)\n`, rank],
    [`${STEP}r = scan(step, 0, 5, 2)\n`, none],
    [`${KD}r ~ kscan(kd, 0.0, 3, 2.0)\n`, none],
    ['g(acc, row) = acc + row.w\nr = scan(g, 0, record(w = 2))\n', record],
  ] as Array<[string, RegExp]>) assert.ok(errors(src).some(m => expected.test(m)), src);
});

// ---------------------------------------------------------------------------
// Violations that static checking pre-empts are still refused when evaluated
// ---------------------------------------------------------------------------

// A model with such a violation never reaches evaluation (static checking refuses it), so build
// the call IR by hand to reach the evaluator's own refusals.
const irLit = (value: any) => ({ kind: 'lit', value });
const irVec = (...values: number[]) => ({ kind: 'call', op: 'vector', args: values.map(irLit) });
const irRef = (name: string) => ({ kind: 'ref', ns: '%local', name });
const irFn = (params: string[], body: any) => ({ kind: 'call', op: 'functionof', params, body });
const irAdd = (a: any, b: any) => ({ kind: 'call', op: 'add', args: [a, b] });
const irMul = (a: any, b: any) => ({ kind: 'call', op: 'mul', args: [a, b] });
const irStep = irFn(['acc', 'x', 'w'], irAdd(irRef('acc'), irMul(irRef('x'), irRef('w'))));
const irKeep = irFn(['x', 'p'], { kind: 'call', op: 'gt', args: [irRef('x'), irRef('p')] });

function evaluates(op: string, args: any[], kwargs?: Record<string, any>) {
  const sampler = require('../sampler.ts');
  return sampler.evaluateExpr({ kind: 'call', op, args, ...(kwargs ? { kwargs } : {}) }, {});
}

test('scan and filter refuse a malformed set of additional inputs when evaluated', () => {
  const x = irVec(10, 20), w = irVec(2, 3);
  assert.throws(() => evaluates('scan', [irStep, irLit(0), irVec(10, 20, 30), w]), /unequal lengths 3, 2/);
  assert.throws(() => evaluates('scan', [irStep, irLit(0)], { w, x, v: x }), /no remaining input named "v"/);
  assert.throws(() => evaluates('scan', [irStep, irLit(0), x], { w }), /all positional or all named/);
  assert.throws(() => evaluates('scan', [irStep, irLit(0), x, w, w]), /no input for additional argument 3/);
  assert.throws(() => evaluates('scan', [irStep, irLit(0), x]), /not bound by the additional arguments/);
  assert.throws(() => evaluates('scan', [irStep, irLit(0), irLit(5), irLit(2)]), /at least one array or table input/);
  assert.throws(() => evaluates('scan', [irFn(['acc'], irRef('acc')), irLit(0)]), /at least one array or table input/);
  assert.throws(() => evaluates('scan', [irStep]), /expected \(function, init, sequences/);
  assert.throws(() => evaluates('filter', [irKeep, irVec(1, 4, 9), irVec(3, 5)]), /has length 2, but data has 3/);
  assert.throws(() => evaluates('filter', [irKeep, irVec(1, 4, 9)], { q: irVec(3) }), /no remaining input named "q"/);
  assert.throws(() => evaluates('filter', [irKeep]), /expected \(predicate, data/);
  // A record is refused as for the operands of `broadcast`.
  const rec = { kind: 'call', op: 'record', fields: [{ name: 'w', value: irLit(2) }] };
  assert.throws(() => evaluates('scan', [irStep, irLit(0), x, rec]), /may not be records or tuples/);
  assert.throws(() => evaluates('filter', [irKeep, irVec(1, 4, 9), rec]), /may not be records or tuples/);
  // An input bound twice.
  assert.throws(() => evaluates('scan', [irStep, irLit(0)], { x, w, y: w }), /no remaining input named "y"/);
});

test('kscan and markovchain describe a malformed step as a located refusal', () => {
  const { describeStepKernel } = require('../markovchain.ts');
  const lit = (value: number) => ({ kind: 'lit', value });
  const kernel = (params: string[], body: any) => ({ kind: 'call', op: 'functionof', params, body });
  const normal = (mu: any) => ({ kind: 'call', op: 'Normal', kwargs: { mu, sigma: lit(1) } });
  const ref = (name: string) => ({ kind: 'ref', ns: '%local', name });
  const none = { args: [], kwargs: {} };
  const reason = (ir: any, op: string, extras: any, bindings: any = new Map()) =>
    describeStepKernel(ir, bindings, op, extras).reason;
  const two = kernel(['s', 'x'], normal(ref('x')));
  assert.match(reason(kernel([], normal(lit(0))), 'markovchain', none), /has no inputs/);
  assert.match(reason(two, 'kscan', none), /at least one array or table input/);
  assert.match(reason(two, 'kscan', { args: [lit(1)], kwargs: { x: lit(1) } }), /all positional or all named/);
  assert.match(reason(two, 'kscan', { args: [], kwargs: { v: lit(1) } }), /no input named "v"/);
  assert.match(reason(two, 'kscan', { args: [lit(1), lit(2)], kwargs: {} }), /1 inputs after its state, but 2 additional/);
  const three = kernel(['s', 'x', 'w'], normal(ref('x')));
  const self = (name: string) => ({ kind: 'ref', ns: 'self', name });
  assert.match(reason(three, 'markovchain', { args: [self('rec')], kwargs: {} }), /statically visible fields/);
  assert.match(reason(three, 'markovchain', { args: [], kwargs: { x: lit(1) } }), /not bound/);
  // A sole record whose fields are visible, through a binding, binds by name.
  const record = { kind: 'call', op: 'record', fields: [{ name: 'x', value: lit(1) }, { name: 'w', value: lit(2) }] };
  const bindings = new Map([['rec', { ir: record }]]);
  const ok = describeStepKernel(three, bindings, 'markovchain', { args: [self('rec')], kwargs: {} });
  assert.equal(ok.reason, undefined);
});

test('checkTrajectoryKernels reports unresolved and unequal kscan sequence lengths', () => {
  const { checkTrajectoryKernels } = require('../markovchain.ts');
  const src = 'k2(s, x, w) = Normal(s + x, w)\nxs = [1.0, 2.0, 3.0]\nws = [1.0, 2.0]\n'
    + 'a ~ kscan(k2, 0.0, xs, ws)\n';
  const { bindings } = processSource(src);
  const built = require('../orchestrator.ts').buildDerivations(bindings);
  const messages = checkTrajectoryKernels(built.bindings).map((d: any) => d.message);
  assert.ok(messages.some((m: string) => /unequal lengths 3, 2/.test(m)), messages.join('; '));
  const unresolved = processSource('k2(s, x, w) = Normal(s + x, w)\nxs = filter(fn(_ > 0.0), [1.0, -1.0, 2.0])\n'
    + 'a ~ kscan(k2, 0.0, xs, [1.0])\n');
  const built2 = require('../orchestrator.ts').buildDerivations(unresolved.bindings);
  assert.ok(checkTrajectoryKernels(built2.bindings).some((d: any) => /length does not resolve/.test(d.message)));
});
