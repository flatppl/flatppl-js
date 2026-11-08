'use strict';

// Spec §04 "Explicit callback parameters": `weighted(f, M, p)`, `logweighted(f, M, p)`,
// `pushfwd(f, M, p)`, `reduce(f, xs, p)`, `filter(pred, data, p)`, `scan(f, init, xs, p)`,
// `markovchain(kernel, init, n, p)` and `kscan(kernel, init, xs, p)` take at most one
// final parameter `p`, passed whole to the callback. `scan` and `kscan` traverse one
// sequence `xs`. A second value after the leading arguments, or any keyword argument
// after them, is a located error. On the
// heads that do not traverse, a sole record or table `p` splats its fields or columns by
// name when two or more inputs remain (#215).
//
// ORACLES are closed forms. weighted(w, Normal(0, 1), p) has the log density
// log w(x, p) + log N(x) at x; logweighted adds the weight; the pushforward of
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

const escaped = (form: string) => form.replace(/[()]/g, '\\$&');
const oneParameter = (form: string) => new RegExp('takes one optional final parameter: `'
  + escaped(form) + '`, with `p` passed whole');
const positionalOnly = (form: string) => new RegExp('the final parameter is positional: `'
  + escaped(form) + '`; pass several values as one record `p`');

const W = 'w(x, p, q) = p * x * x + q\n';
const AT = 0.5;
const WANT = Math.log(3 * AT * AT + 1) + logN(AT);

test('weighted: a sole record p splats onto two remaining inputs', async () => {
  near(await value(`${W}m = weighted(w, Normal(0, 1), record(p = 3.0, q = 1.0))\nlp = logdensityof(m, ${AT})\n`, 'lp'),
    [WANT]);
});

test('weighted: a record p passes whole when one input remains', async () => {
  // w(x, p) = x * p.a with p = record(a = 2): the weight at 0.5 is 1.
  near(await value('w(x, p) = x * p.a\nm = weighted(w, Normal(0, 1), record(a = 2.0))\nlp = logdensityof(m, 0.5)\n', 'lp'),
    [logN(0.5)]);
});

test('weighted: a table p passes whole when one input remains', async () => {
  const src = 'w(x, t) = t.p[1] * x * x + t.q[1]\nt = table(p = [3.0], q = [1.0])\n'
    + 'm = weighted(w, Normal(0, 1), t)\nlp = logdensityof(m, 0.5)\n';
  near(await value(src, 'lp'), [WANT]);
});

test('weighted and logweighted: a second value or a keyword after M is a located error', () => {
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), 3.0, 1.0)\n`)
    .some(m => oneParameter('weighted(f, M, p)').test(m)));
  for (const extras of ['p = 3.0, q = 1.0', '3.0, q = 1.0', 'p = record(p = 3.0, q = 1.0)']) {
    assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), ${extras})\n`)
      .some(m => positionalOnly('weighted(f, M, p)').test(m)), extras);
  }
  const lw = 'lw(x, p, q) = p * x + q\n';
  assert.ok(errors(`${lw}m = logweighted(lw, Normal(0, 1), 2.0, 1.0)\n`)
    .some(m => oneParameter('logweighted(f, M, p)').test(m)));
});

test('logweighted: a record p splats onto the remaining inputs', async () => {
  const lw = 'lw(x, p, q) = p * x + q\n';
  near(await value(`${lw}m = logweighted(lw, Normal(0, 1), record(p = 2.0, q = 1.0))\nlp = logdensityof(m, ${AT})\n`, 'lp'),
    [2.0 * AT + 1.0 + logN(AT)]);
});

test('an unbound or unknown input is a static error', () => {
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), 3.0)\n`).some(m => /callback receives 2 whole inputs, but declares 3/.test(m)));
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), record(p = 3.0, r = 1.0))\n`)
    .some(m => /no remaining input named "r"/.test(m)));
});

test('the leading inputs of weighted stay positional', () => {
  assert.ok(errors(`${W}m = weighted(weight = w, base = Normal(0, 1), record(p = 3.0, q = 1.0))\n`)
    .some(m => /never by keyword argument/.test(m)));
});

test('pushfwd: p reaches the function, its inverse and its log-volume', async () => {
  const bij = 'f(x, p) = p * exp(x)\nfi(y, p) = log(y / p)\nlv(x, p) = x + log(p)\nb = bijection(f, fi, lv)\n';
  const want = logN(Math.log(3 / 2)) - Math.log(3);
  const src = `${bij}m = pushfwd(b, Normal(0, 1), 2.0)\nlp = logdensityof(m, 3.0)\n`;
  assert.deepEqual(errors(src), []);
  near(await value(src, 'lp'), [want]);
  assert.ok(errors(`${bij}m = pushfwd(b, Normal(0, 1), 2.0, 1.0)\n`)
    .some(m => oneParameter('pushfwd(f, M, p)').test(m)));
  assert.ok(errors(`${bij}m = pushfwd(b, Normal(0, 1), p = 2.0)\n`)
    .some(m => positionalOnly('pushfwd(f, M, p)').test(m)));
});

test('reduce: p follows the two inputs and stays fixed in every call', async () => {
  const f = 'f(a, b, p, q) = a + b * p + q\n';
  // ((1 + 2*2 + 1) + 3*2 + 1) = 13
  assert.deepEqual(await value(`${f}r = reduce(f, [1.0, 2.0, 3.0], record(p = 2.0, q = 1.0))\n`, 'r'), [13]);
  assert.deepEqual(await value('g(a, b, p) = a + b * p.s + p.t\nr = reduce(g, [1.0, 2.0, 3.0], record(s = 2.0, t = 1.0))\n', 'r'),
    [13]);
  assert.deepEqual(await value('r = reduce(add, [10, 20] .* [2, 3])\n', 'r'), [80]);
  assert.deepEqual(await value('r = reduce(add, [7.0])\n', 'r'), [7]);
  assert.ok(errors(`${f}r = reduce(f, [1.0, 2.0], 2.0, 1.0)\n`).some(m => oneParameter('reduce(f, xs, p)').test(m)));
});

// ---------------------------------------------------------------------------
// scan and filter: one traversed sequence, p passed whole
// ---------------------------------------------------------------------------

const STEP = 'step(acc, x, w) = acc + x * w\n';

test('scan: xs is traversed and a scalar p is the same at every step', async () => {
  // acc: 0 + 10*2 = 20, then 20 + 20*2 = 60.
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, [10, 20], 2)\n`, 'r'), [20, 60]);
  assert.deepEqual(await value('s1(acc, x) = acc + x\nr = scan(s1, 0, [1, 2, 3])\n', 'r'), [1, 3, 6]);
  assert.deepEqual(await value(`${STEP}r = scan(step, 0, fill(0, 0), 2)\n`, 'r'), []);
});

test('scan: a vector p of the length of xs is passed whole, not traversed', async () => {
  // w = [2, 3] at every step: 0 + 10*2 + 3 = 23, then 23 + 20*2 + 3 = 66.
  const src = 'stepv(acc, x, w) = acc + x * w[1] + w[2]\nr = scan(stepv, 0, [10, 20], [2, 3])\n';
  assert.deepEqual(errors(src), []);
  assert.deepEqual(await value(src, 'r'), [23, 66]);
});

test('scan: a table xs is traversed by rows, and a table or record p passes whole', async () => {
  assert.deepEqual(await value('g(acc, row) = acc + row.x * row.w\nr = scan(g, 0, table(x = [10, 20], w = [2, 3]))\n', 'r'),
    [20, 80]);
  assert.deepEqual(await value('g(acc, x, t) = acc + x * t.w[1]\nr = scan(g, 0, [10, 20], table(w = [2]))\n', 'r'), [20, 60]);
  assert.deepEqual(await value('g(acc, x, p) = acc + x * p.w\nr = scan(g, 0, [10, 20], record(w = 2))\n', 'r'), [20, 60]);
});

test('scan: a tuple or a matrix p passes whole', async () => {
  assert.deepEqual(await value('g(acc, x, t) = acc + x * t[1] + t[2]\nr = scan(g, 0, [10, 20], (2.0, 3.0))\n', 'r'), [23, 66]);
  assert.deepEqual(await value('M = rowstack([[1.0, 2.0], [3.0, 4.0]])\ng(acc, x, A) = acc + x * A[1, 2]\n'
    + 'r = scan(g, 0, [10, 20], M)\n', 'r'), [20, 60]);
});

test('scan: a second value after xs, or no sequence xs, is a static error', () => {
  assert.ok(errors('f(acc, x, w, b) = acc + x * w + b\nr = scan(f, 0, [10, 20], [2, 3], 1)\n')
    .some(m => oneParameter('scan(f, init, xs, p)').test(m)));
  assert.ok(errors(`${STEP}r = scan(step, 0, [10, 20], w = 2)\n`)
    .some(m => positionalOnly('scan(f, init, xs, p)').test(m)));
  assert.ok(errors(`${STEP}r = scan(step, 0, 5, 2)\n`).some(m => /traverses one sequence `xs`/.test(m)));
  assert.ok(errors('g(acc, row) = acc + row.w\nr = scan(g, 0, record(w = 2))\n').some(m => /traverses one sequence `xs`/.test(m)));
  assert.ok(errors(`${STEP}r = scan(step, 0)\n`).some(m => /at least three arguments/.test(m)));
});

test('a vector of vectors xs is traversed by element, and a vector p passes whole', async () => {
  // Elements [1, 2] and [3, 4]: 0 + 3, then 3 + 7.
  assert.deepEqual(await value('h(acc, v) = acc + sum(v)\nr = scan(h, 0, [[1.0, 2.0], [3.0, 4.0]])\n', 'r'), [3, 10]);
  // A = [1, 2] at every step: 0 + 10 * 3, then + 20 * 3.
  assert.deepEqual(await value('h(acc, x, v) = acc + x * sum(v)\nA = [1.0, 2.0]\nr = scan(h, 0, [10, 20], A)\n', 'r'),
    [30, 90]);
});

test('filter: p is the same for every element, also a vector p', async () => {
  assert.deepEqual(await value('keep(x, p) = x > p\nr = filter(keep, [1.0, 4.0, 9.0], 3.0)\n', 'r'), [4, 9]);
  // The bounds [0, 5] apply to every element: 1 and 4 are kept.
  assert.deepEqual(await value('within(x, b) = b[1] <= x <= b[2]\nr = filter(within, [1, 4, 9], [0, 5])\n', 'r'), [1, 4]);
  assert.deepEqual(await value('g(x, p) = x > p.w\nr = filter(g, [1, 4, 9], record(w = 2))\n', 'r'), [4, 9]);
  assert.ok(errors('within(x, low, high) = low <= x <= high\nr = filter(within, [1, 4, 9], 0, 10)\n')
    .some(m => oneParameter('filter(pred, data, p)').test(m)));
});

test('filter: a table is filtered by rows and empty data stays empty', async () => {
  const src = 'above(row, p) = row.value > p\nt = table(value = [1, 2, 3], other = [10, 20, 30])\n'
    + 'kept = filter(above, t, 1)\nr = kept.other\n';
  assert.deepEqual(await value(src, 'r'), [20, 30]);
  assert.deepEqual(await value('keep(x, p) = x > p\nr = filter(keep, fill(1.0, 0), 3.0)\n', 'r'), []);
});

test('a parametric p stays live in a scored callable', async () => {
  const live = 'p = elementof(reals)\nkeep(x, q) = x > q\nkept = filter(keep, [1.0, 2.0, 3.0], p)\nM = Normal(sum(kept), 1.0)\n'
    + 'score = functionof(logdensityof(M, 0.0), p)\na = score(0.5)\nb = score(2.5)\n';
  // p = 0.5 keeps 1, 2, 3 (sum 6); p = 2.5 keeps 3 (sum 3): log N(0; sum, 1).
  near(await value(live, 'a'), [-0.5 * Math.log(2 * Math.PI) - 0.5 * 36]);
  near(await value(live, 'b'), [-0.5 * Math.log(2 * Math.PI) - 0.5 * 9]);
});

// ---------------------------------------------------------------------------
// markovchain and kscan
// ---------------------------------------------------------------------------

function chainLogDensity(path: number[], drift: number, scale: number): number {
  let prev = 0, want = 0;
  for (const x of path) {
    const z = (x - prev - drift) / scale;
    want += -0.5 * Math.log(2 * Math.PI) - Math.log(scale) - 0.5 * z * z;
    prev = x;
  }
  return want;
}

test('markovchain: a record p splats onto the kernel inputs or is read whole', async () => {
  const path = [0.3, 0.1, 0.9];
  const want = chainLogDensity(path, 0.1, 0.5);
  for (const src of [
    'step(state, drift, scale) = Normal(state + drift, scale)\nM = markovchain(step, 0.0, 3, (drift = 0.1, scale = 0.5))\n',
    'step(state, p) = Normal(state + p.drift, p.scale)\nM = markovchain(step, 0.0, 3, (drift = 0.1, scale = 0.5))\n',
  ]) {
    const full = `${src}lp = logdensityof(M, [${path.join(', ')}])\n`;
    assert.deepEqual(errors(full), []);
    near(await value(full, 'lp'), [want]);
  }
});

test('markovchain: a second value after n, or an unbound kernel input, is refused', () => {
  const step = 'step(state, drift, scale) = Normal(state + drift, scale)\n';
  assert.ok(errors(`${step}M = markovchain(step, 0.0, 3, 0.1, 0.5)\n`)
    .some(m => oneParameter('markovchain(kernel, init, n, p)').test(m)));
  assert.ok(errors(`${step}M = markovchain(step, 0.0, 3, drift = 0.1, scale = 0.5)\n`)
    .some(m => positionalOnly('markovchain(kernel, init, n, p)').test(m)));
  assert.ok(errors(`${step}M = markovchain(step, 0.0, 3, 0.1)\n`).length > 0);
});

const KS = 'kd(s, x, w) = Dirac(value = s + x * w)\n';

test('kscan: xs is traversed, and a scalar or vector p passes whole', async () => {
  // s: 0 + 10*2 = 20, then 20 + 20*2 = 60.
  assert.deepEqual(await value(`${KS}r ~ kscan(kd, 0.0, [10.0, 20.0], 2.0)\n`, 'r'), [20, 60]);
  assert.deepEqual(await value('k1(s, x) = Dirac(value = s + x)\nr ~ kscan(k1, 0.0, [10.0, 20.0])\n', 'r'), [10, 30]);
  // w = [2, 3] at every step: 0 + 10*2 + 3 = 23, then 23 + 20*2 + 3 = 66.
  assert.deepEqual(await value('kv(s, x, w) = Dirac(value = s + x * w[1] + w[2])\nr ~ kscan(kv, 0.0, [10.0, 20.0], [2.0, 3.0])\n',
    'r'), [23, 66]);
});

test('kscan: per-step inputs come from a table xs, and the density is the product of the transitions', async () => {
  const step = (x: number, mean: number, sd: number) => -0.5 * Math.log(2 * Math.PI) - Math.log(sd)
    - 0.5 * ((x - mean) / sd) ** 2;
  const src = 'k2(s, row) = Normal(s + row.x, row.w)\nM = kscan(k2, 0.0, table(x = [1.0, 2.0], w = [0.5, 0.7]))\n'
    + 'lp = logdensityof(M, [0.6, 2.9])\n';
  near(await value(src, 'lp'), [step(0.6, 0 + 1.0, 0.5) + step(2.9, 0.6 + 2.0, 0.7)]);
  const held = 'k2(s, x, w) = Normal(s + x, w)\nM = kscan(k2, 0.0, [1.0, 2.0], 0.5)\nlp = logdensityof(M, [0.6, 2.9])\n';
  near(await value(held, 'lp'), [step(0.6, 0 + 1.0, 0.5) + step(2.9, 0.6 + 2.0, 0.5)]);
});

test('kscan: a second value after xs, or no sequence, is refused', () => {
  assert.ok(errors(`${KS}r ~ kscan(kd, 0.0, [1.0, 2.0], 1.0, 2.0)\n`)
    .some(m => oneParameter('kscan(kernel, init, xs, p)').test(m)));
  assert.ok(errors(`${KS}r ~ kscan(kd, 0.0)\n`).some(m => /at least one array or table/.test(m)));
  assert.ok(errors(`${KS}r ~ kscan(kd, 0.0, 3, 2.0)\n`).some(m => /traverses one sequence `xs`/.test(m)));
});

test('sequence callbacks receive record parameters whole', async () => {
  // §04 "Explicit callback parameters": p stays whole at every step.
  for (const [source, want] of [
    ['g(acc,x,p)=acc+x*p.w\nr=scan(g,0,[10,20],record(w = 2))', [20, 60]],
    ['g(s,x,p)=Dirac(s+x*p.w)\nr~kscan(g,0,[1,2],record(w = 2))', [2, 6]],
    ['g(x,p)=x>p.w\nr=filter(g,[1,4,9],record(w = 2))', [4, 9]],
  ] as Array<[string, number[]]>) {
    assert.deepEqual(errors(source), []);
    assert.deepEqual(await value(source, 'r'), want);
  }
});

test('a keyword parameter is refused on every head', () => {
  for (const [source, form] of [
    ['w(x, p) = p.a * x\nm = weighted(w, Normal(0, 1), p = record(a = 2.0))', 'weighted(f, M, p)'],
    ['w(x, p) = p.a * x\nm = logweighted(w, Normal(0, 1), p = record(a = 2.0))', 'logweighted(f, M, p)'],
    ['f(x, p) = p.a * x\nm = pushfwd(f, Normal(0, 1), p = record(a = 2.0))', 'pushfwd(f, M, p)'],
    ['f(a, b, p) = a + b * p.a\nr = reduce(f, [1.0, 2.0], p = record(a = 2.0))', 'reduce(f, xs, p)'],
    ['s(x, p) = Normal(x, p.a)\nM = markovchain(s, 0.0, 3, p = record(a = 2.0))', 'markovchain(kernel, init, n, p)'],
    ['g(acc, x, p) = acc + x * p.a\nr = scan(g, 0.0, [1.0, 2.0], p = record(a = 2.0))', 'scan(f, init, xs, p)'],
    ['k(s, x, p) = Normal(s + x, p.a)\nM = kscan(k, 0.0, [1.0, 2.0], p = record(a = 2.0))', 'kscan(kernel, init, xs, p)'],
    ['g(x, p) = x > p.a\nr = filter(g, [1.0, 4.0], p = record(a = 2.0))', 'filter(pred, data, p)'],
  ] as Array<[string, string]>) {
    const name = form.slice(0, form.indexOf('('));
    assert.ok(errors(source + '\n').some(m => m.startsWith('`' + name + '`: ') && positionalOnly(form).test(m)
      && m.endsWith('(spec §04 "Explicit callback parameters")')), source);
  }
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

test('scan and filter refuse a malformed parameter when evaluated', () => {
  const x = irVec(10, 20), w = irVec(2, 3);
  assert.throws(() => evaluates('scan', [irStep, irLit(0), x, w, w]), oneParameter('scan(f, init, xs, p)'));
  assert.throws(() => evaluates('scan', [irStep, irLit(0), x, w], { v: w }), positionalOnly('scan(f, init, xs, p)'));
  assert.throws(() => evaluates('scan', [irStep, irLit(0), x], { w }), positionalOnly('scan(f, init, xs, p)'));
  assert.throws(() => evaluates('scan', [irStep, irLit(0), x]), /takes 3 inputs, but receives 2/);
  assert.throws(() => evaluates('scan', [irStep, irLit(0), irLit(5), irLit(2)]), /`xs` must be a one-dimensional array or a table/);
  assert.throws(() => evaluates('scan', [irStep, irLit(0)]), /traverses one sequence `xs`/);
  assert.throws(() => evaluates('scan', [irStep]), /expected \(function, init, xs, optional p\)/);
  assert.throws(() => evaluates('filter', [irKeep, irVec(1, 4, 9), irLit(3), irLit(5)]), oneParameter('filter(pred, data, p)'));
  assert.throws(() => evaluates('filter', [irKeep, irVec(1, 4, 9)], { p: irLit(3) }), positionalOnly('filter(pred, data, p)'));
  assert.throws(() => evaluates('filter', [irKeep]), /expected \(predicate, data, optional p\)/);
});

test('scan passes a vector p whole when evaluated', () => {
  // w = [2, 3] at every step: 0 + 10*2 + 3 = 23, then 23 + 20*2 + 3 = 66.
  const get = (v: any, i: number) => ({ kind: 'call', op: 'get', args: [v, irLit(i)] });
  const stepv = irFn(['acc', 'x', 'w'],
    irAdd(irRef('acc'), irAdd(irMul(irRef('x'), get(irRef('w'), 1)), get(irRef('w'), 2))));
  const r = evaluates('scan', [stepv, irLit(0), irVec(10, 20), irVec(2, 3)]);
  assert.deepEqual(Array.from(r.data ?? r), [23, 66]);
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
  assert.match(reason(two, 'kscan', none), /traverses one sequence `xs`/);
  assert.match(reason(two, 'kscan', { args: [lit(1), lit(2)], kwargs: {} }), /takes no parameter `p`/);
  assert.match(reason(two, 'kscan', { args: [lit(1), lit(2), lit(3)], kwargs: {} }),
    oneParameter('kscan(kernel, init, xs, p)'));
  const three = kernel(['s', 'x', 'w'], normal(ref('x')));
  const self = (name: string) => ({ kind: 'ref', ns: 'self', name });
  assert.match(reason(three, 'markovchain', { args: [self('rec')], kwargs: {} }), /read its fields as `p.name`/);
  assert.match(reason(three, 'markovchain', { args: [], kwargs: { x: lit(1) } }),
    positionalOnly('markovchain(kernel, init, n, p)'));
  assert.match(reason(three, 'markovchain', { args: [lit(1), lit(2)], kwargs: {} }),
    oneParameter('markovchain(kernel, init, n, p)'));
  // A sole record whose fields are visible, through a binding, binds by name.
  const record = { kind: 'call', op: 'record', fields: [{ name: 'x', value: lit(1) }, { name: 'w', value: lit(2) }] };
  const bindings = new Map([['rec', { ir: record }]]);
  const ok = describeStepKernel(three, bindings, 'markovchain', { args: [self('rec')], kwargs: {} });
  assert.equal(ok.reason, undefined);
});

test('checkTrajectoryKernels passes a vector p whole and reports an unresolved sequence length', () => {
  const { checkTrajectoryKernels } = require('../markovchain.ts');
  const src = 'k2(s, x, w) = Normal(s + x, w[1])\nxs = [1.0, 2.0, 3.0]\nws = [1.0, 2.0]\n'
    + 'a ~ kscan(k2, 0.0, xs, ws)\n';
  const { bindings } = processSource(src);
  const built = require('../orchestrator.ts').buildDerivations(bindings);
  assert.deepEqual(checkTrajectoryKernels(built.bindings).map((d: any) => d.message), []);
  const unresolved = processSource('k2(s, x, w) = Normal(s + x, w)\nxs = filter(fn(_ > 0.0), [1.0, -1.0, 2.0])\n'
    + 'a ~ kscan(k2, 0.0, xs, 1.0)\n');
  const built2 = require('../orchestrator.ts').buildDerivations(unresolved.bindings);
  assert.ok(checkTrajectoryKernels(built2.bindings).some((d: any) => /length does not resolve/.test(d.message)));
});
