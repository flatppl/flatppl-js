'use strict';
// A function body keeps its intermediate bindings as a `let` binding graph:
// each binding is evaluated once, where copying each into every use made the
// cost exponential in the depth of sharing. The amplitude models that weight
// Lebesgue by `rate_fn` had a 3e7-node body, and their weight IR reached
// 7e9 nodes once user calls were inlined by substitution.
//
// Oracles, closed form: a_k = a_{k-1} + a_{k-1} with a_0 = x gives
// a_30 = 2^30 · x. For dd(x, y) = x − 2y, dd(y, x) = y − 2x.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const DEPTH = 30;
let chain = 'x = elementof(interval(0.0, 1.0))\na0 = x\n';
for (let k = 1; k <= DEPTH; k++) chain += `a${k} = a${k - 1} + a${k - 1}\n`;
chain += `f = functionof(a${DEPTH}, x)\n`;

async function value(src: string, name: string, n = 1) {
  const { ctx } = makeMatCtx(src, { sampleCount: n, rootSeed: 612 });
  return ctx.getMeasure(name);
}
const first = (m: any) => (m.value ? m.value.data : m.samples)[0];

test('a function body with deep sharing is a let graph of its bindings', async () => {
  const { ctx } = makeMatCtx(chain, { sampleCount: 1 });
  const body = ctx.bindings.get('f').ir.body;
  assert.equal(body.op, 'let');
  // Each a_k with k < DEPTH is read twice and is bound once. a_DEPTH is read
  // once, by the root, and is inlined there. a0 = x reads as x itself.
  assert.deepEqual(body.assigns.map((a: any) => a.name),
    Array.from({ length: DEPTH - 1 }, (_, k) => `a${k + 1}`));
  assert.equal(body.body.op, 'add');
  assert.deepEqual(body.body.args.map((a: any) => a.name), [`a${DEPTH - 1}`, `a${DEPTH - 1}`]);
});

test('a call of it evaluates each binding once', async () => {
  const t = Date.now();
  const m = await value(chain + 'v = f(0.5)\nok = checked(v, abs(v - 0.5 * 2.0^30) <= 1e-6)\n', 'ok');
  assert.equal(first(m), 0.5 * 2 ** DEPTH);
  assert.ok(Date.now() - t < 20000, `took ${Date.now() - t} ms`);
});

test('a weight and a record-valued pushforward evaluate it once per atom', async () => {
  const src = chain + 'B = Lebesgue(support = interval(0.0, 1.0))\nM = weighted(f, B)\n'
    + 'lp = logdensityof(M, 0.25)\n'
    + `g = functionof(record(a = a${DEPTH}, b = a${DEPTH} + 1.0), x)\nQ = pushfwd(g, normalize(M))\n`;
  assert.equal(first(await value(src, 'lp')), Math.log(0.25 * 2 ** DEPTH));
  const { ctx } = makeMatCtx(src, { sampleCount: 64, rootSeed: 612 });
  const p = await ctx.getMeasure('Q');
  const a = p.fields.a.samples, b = p.fields.b.samples;
  for (let i = 0; i < a.length; i++) {
    assert.ok(a[i] >= 0 && a[i] <= 2 ** DEPTH, `a ${a[i]}`);
    assert.equal(b[i], a[i] + 1);
  }
});

test('a user call binds its arguments in the caller scope', async () => {
  // dd(y, x) for dd(x, y) = x − 2y reads the caller's y and x, even though
  // the call rebinds both names.
  const src = 'x = elementof(interval(0.0, 1.0))\ny = elementof(interval(0.0, 2.0))\n'
    + 'diff = x - 2.0 * y\ndd = functionof(diff, x, y)\n'
    + 'e = dd(y, x) + 10.0\nF = functionof(e, x, y)\n'
    + 'S = cartprod(x = interval(0.0, 1.0), y = interval(0.0, 2.0))\n'
    + 'M = weighted(F, Lebesgue(support = S))\nlp = logdensityof(M, record(x = 0.5, y = 1.5))\n'
    + 'v = F(0.5, 1.5)\n';
  assert.deepEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error'), []);
  // y − 2x + 10 at (0.5, 1.5) = 1.5 − 1.0 + 10.
  assert.equal(first(await value(src, 'v')), 10.5);
  assert.ok(Math.abs(first(await value(src, 'lp')) - Math.log(10.5)) < 1e-12);
});

test('a record-valued pushforward evaluates a shared binding once for all fields', async () => {
  // Count batched evaluations of log1p, which only the shared binding uses.
  const batched = require('../sampler-eval-batched.ts');
  const original = batched.ARITH_OPS_N.log1p;
  let calls = 0;
  batched.ARITH_OPS_N.log1p = function (...args: any[]) { calls++; return original.apply(this, args); };
  try {
    const src = 'x = elementof(interval(0.0, 1.0))\nshared = log1p(x)\n'
      + 'g = functionof(record(a = shared, b = shared + 1.0, c = 2.0 * shared), x)\n'
      + 'Q = pushfwd(g, Uniform(interval(0.0, 1.0)))\n';
    const { ctx } = makeMatCtx(src, { sampleCount: 16, rootSeed: 612 });
    const q = await ctx.getMeasure('Q');
    assert.equal(calls, 1);
    for (let i = 0; i < 16; i++) {
      assert.equal(q.fields.b.samples[i], q.fields.a.samples[i] + 1);
      assert.equal(q.fields.c.samples[i], 2 * q.fields.a.samples[i]);
    }
  } finally {
    batched.ARITH_OPS_N.log1p = original;
  }
});

test('a record-valued pushforward evaluates a deeply shared binding once', async () => {
  // Count batched evaluations of log1p, which only s0 uses. s_k doubles
  // s_{k-1}, so s10 = 1024 · log1p(x).
  const batched = require('../sampler-eval-batched.ts');
  const original = batched.ARITH_OPS_N.log1p;
  let calls = 0;
  batched.ARITH_OPS_N.log1p = function (...args: any[]) { calls++; return original.apply(this, args); };
  try {
    let src = 'x = elementof(interval(0.0, 1.0))\ns0 = log1p(x)\n';
    for (let k = 1; k <= 10; k++) src += `s${k} = s${k - 1} + s${k - 1}\n`;
    src += 'g = functionof(record(a = s10, b = s10 + 1.0, c = 2.0 * s10), x)\n'
      + 'Q = pushfwd(g, Uniform(interval(0.0, 1.0)))\n';
    const { ctx } = makeMatCtx(src, { sampleCount: 16, rootSeed: 612 });
    const q = await ctx.getMeasure('Q');
    assert.equal(calls, 1);
    for (let i = 0; i < 16; i++) {
      // a = 1024 · log1p(x) ∈ [0, 1024 · log 2].
      assert.ok(q.fields.a.samples[i] >= 0 && q.fields.a.samples[i] <= 1024 * Math.LN2);
      assert.equal(q.fields.b.samples[i], q.fields.a.samples[i] + 1);
      assert.equal(q.fields.c.samples[i], 2 * q.fields.a.samples[i]);
    }
  } finally {
    batched.ARITH_OPS_N.log1p = original;
  }
});

test('a user call inside a pushforward body and a vector intermediate evaluate per atom', async () => {
  // shift(v) = v + [1, 2] is a user function; mid = shift([x, 2x]) is a
  // per-atom vector intermediate shared by both fields.
  const src = 'x = elementof(interval(0.0, 1.0))\nv = elementof(cartpow(reals, 2))\n'
    + 'w = v + [1.0, 2.0]\nshift = functionof(w, v)\n'
    + 'mid = shift([x, 2.0 * x])\ng = functionof(record(p = mid, s = sum(mid)), x)\n'
    + 'Q = pushfwd(g, Uniform(interval(0.0, 1.0)))\n';
  const { ctx } = makeMatCtx(src, { sampleCount: 16, rootSeed: 612 });
  const q = await ctx.getMeasure('Q');
  const p = q.fields.p.value, s = q.fields.s.samples;
  assert.deepEqual(p.shape, [16, 2]);
  for (let i = 0; i < 16; i++) {
    const x = p.data[2 * i] - 1;
    assert.ok(Math.abs(p.data[2 * i + 1] - (2 * x + 2)) < 1e-12);
    assert.ok(Math.abs(s[i] - (3 * x + 3)) < 1e-12);
  }
});

test('a broadcast of a deeply shared function evaluates per cell', async () => {
  const t = Date.now();
  const m = await value(chain + 'v = f.([0.25, 0.5])\n', 'v');
  const data = m.value ? m.value.data : m.samples;
  assert.deepEqual(Array.from(data).slice(0, 2), [0.25 * 2 ** DEPTH, 0.5 * 2 ** DEPTH]);
  assert.ok(Date.now() - t < 20000, `took ${Date.now() - t} ms`);
});

test('a let never reaches FlatPIR printing', () => {
  // Building derivations rewrites function bodies into let form; the FlatPIR
  // writer reads the lowered module, which must keep the surface shape.
  const { processSource: process, orchestrator, pirSexpr } = require('..');
  const src = chain + 'g(y) = f(y) + 1.0\nB = Lebesgue(support = interval(0.0, 1.0))\n'
    + 'M = weighted(f, B)\nQ = pushfwd(g, normalize(M))\n';
  const r = process(src);
  const built = orchestrator.buildDerivations(r.bindings);
  assert.equal(built.bindings.get('f').ir.body.op, 'let', 'derivations use the let form');
  const text = pirSexpr.toSexpr(r.loweredModule);
  assert.ok(!/\(let[\s)]/.test(text), 'no let in FlatPIR');
  assert.ok(!/%arg:/.test(text), 'no internal argument names in FlatPIR');
});

test('a body without sharing keeps its inlined shape', () => {
  // Recognisers read these shapes; a binding read once is inlined as before.
  const { ctx } = makeMatCtx('x = elementof(interval(0.0, 1.0))\nb = 2.0 * x\nc = b + 1.0\nf = functionof(c, x)\n',
    { sampleCount: 1 });
  const body = ctx.bindings.get('f').ir.body;
  assert.equal(body.op, 'add');
  assert.equal(body.args[0].op, 'mul');
});

test('a kernel body binds a shared parameter expression once and keeps its measure node', async () => {
  // The kernel's Normal stays a measure node; its mean, which reads the
  // shared s twice, takes the let form. Normal(2θ+2θ, 1) at y = 1.5, θ = 0.25.
  const src = 'theta = elementof(reals)\ns = 2.0 * theta\nm = s + s\n'
    + 'y ~ Normal(m, 1.0)\nK = kernelof(y, theta = theta)\nlp = logdensityof(K(0.25), 1.5)\n';
  const want = -0.5 * Math.log(2 * Math.PI) - 0.5 * (1.5 - 1.0) ** 2;
  assert.ok(Math.abs(first(await value(src, 'lp')) - want) < 1e-12);
});

test('a normalizer finds the jumps of a weight in let form', async () => {
  // The windows |2x − 1| < 1e-4 and |4x − 1| < 1e-4 have widths 1e-4 and
  // 5e-5, so Z = ∫₀¹ w = 2 + 100 · (1e-4 + 5e-5) = 2.015. Unsplit, a
  // window narrower than the node spacing scores as absent.
  let src = 'x = elementof(interval(0.0, 1.0))\nc0 = x * 0.0 + 1.0\n';
  for (let k = 1; k <= DEPTH; k++) src += `c${k} = c${k - 1} * c${k - 1}\n`;
  src += `g = functionof(c${DEPTH} * ifelse(abs(x - 1.0) < 1e-4, 101.0, 1.0), x)\n`
    + 'w = functionof(g(2.0 * x) + g(4.0 * x), x)\n'
    + 'M = normalize(weighted(w, Lebesgue(support = interval(0.0, 1.0))))\n'
    + 'out = logdensityof(M, 0.75)\nin_window = logdensityof(M, 0.5)\n';
  assert.ok(Math.abs(first(await value(src, 'out')) - Math.log(2 / 2.015)) < 1e-9);
  assert.ok(Math.abs(first(await value(src, 'in_window')) - Math.log(102 / 2.015)) < 1e-9);
});
