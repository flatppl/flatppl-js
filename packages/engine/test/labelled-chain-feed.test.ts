'use strict';

// Spec §06 `jointchain` "Keyword form": the variate of `jointchain(a = M, b = K)` is that of
// `jointchain(relabel(M, ["a"]), relabel(K, ["b"]))`, and "each later kernel receives the record of
// the preceding named variates by the ordinary calling convention, so `jointchain(a = M, b = K)`
// calls `K(record(a = a))`". §04 "Auto-splatting" then decides: a kernel with one input takes the
// record whole, a kernel with two or more inputs splats it by field name. Before, a labelled step
// whose one input was a real (`fn(Normal(_, 1.0))`) was handed `record(aa = x)` and scored NaN with
// no diagnostic. Every expected value is the closed-form sum of Normal log densities.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const logN = (x: number, mean: number, sd = 1) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(sd) - 0.5 * ((x - mean) / sd) ** 2;

async function value(src: string, name: string, n = 1): Promise<any> {
  const { ctx } = ctxFor(src, n);
  return ctx.getMeasure(name);
}

async function score(src: string): Promise<number> {
  const m = await value(src, 'lp');
  return (m.value ? m.value.data : m.samples)[0];
}

const errors = (src: string) => processSource(src).diagnostics.filter((d: any) => d.severity === 'error');

// aa ~ N(0, 1), bb | aa ~ N(aa, 1), at aa = bb = 0.5.
const TWO = logN(0.5, 0) + logN(0.5, 0.5);

test('a one-input kernel takes the record of the preceding variates whole', async () => {
  const src = 'k(r) = Normal(r.aa, 1.0)\nA = jointchain(aa = Normal(0.0, 1.0), bb = k)\n'
    + 'lp = logdensityof(A, record(aa = 0.5, bb = 0.5))\n';
  assert.deepEqual(errors(src), []);
  const lp = await score(src);
  assert.ok(Number.isFinite(lp), 'the density is finite, not NaN');
  assert.ok(Math.abs(lp - TWO) < 1e-12, `got ${lp}, want ${TWO}`);
});

test('a record-taking step and the positional chain agree', async () => {
  const labelled = await score('k(r) = Normal(r.aa, 1.0)\nA = jointchain(aa = Normal(0.0, 1.0), bb = k)\n'
    + 'lp = logdensityof(A, record(aa = 0.5, bb = 0.5))\n');
  const positional = await score('k(a) = Normal(a, 1.0)\nA = jointchain(Normal(0.0, 1.0), k)\n'
    + 'lp = logdensityof(A, [0.5, 0.5])\n');
  assert.ok(Math.abs(labelled - TWO) < 1e-12);
  assert.ok(Math.abs(positional - TWO) < 1e-12);
});

test('a one-input real kernel receiving the record is a located static error, not NaN', () => {
  for (const step of ['functionof(Normal(mu = a, sigma = 1.0), aa = a)', 'kernelof(y, aa = a)']) {
    const src = 'a = elementof(reals)\ny ~ Normal(a, 1.0)\n'
      + `A = jointchain(aa = Normal(0.0, 1.0), bb = ${step})\n`
      + 'lp = logdensityof(A, record(aa = 0.5, bb = 0.5))\n';
    const errs = errors(src);
    assert.equal(errs.length, 1, step + ': ' + JSON.stringify(errs.map((e: any) => e.message)));
    assert.match(errs[0].message, /keyword form each later kernel receives the record of the preceding named variates \("aa"\)/);
    assert.match(errs[0].message, /cannot unify previous step's result type record with fields aa: real/);
    assert.equal(errs[0].loc.start.line, 2);
  }
});

test('a kernel with several inputs splats the record by field name, and every input must be named', () => {
  const head = 'k(r) = Normal(r.aa, 1.0)\n';
  assert.deepEqual(errors(head + 'kc(aa, bb) = Normal(aa + bb, 1.0)\n'
    + 'A = jointchain(aa = Normal(0.0, 1.0), bb = k, cc = kc)\n'), []);
  // `kz` takes a third input no label names.
  const missing = errors(head + 'kz(aa, zz) = Normal(aa, 1.0)\n'
    + 'A = jointchain(aa = Normal(0.0, 1.0), bb = k, cc = kz)\n');
  assert.equal(missing.length, 1);
  assert.match(missing[0].message, /expects the record to hold|missing field "zz"|no input|"zz"/);
  // A surplus field is an error too: `kb` does not take `aa`.
  const surplus = errors(head + 'kb(bb, dd) = Normal(bb, 1.0)\n'
    + 'A = jointchain(aa = Normal(0.0, 1.0), bb = k, cc = kb)\n');
  assert.equal(surplus.length, 1);
});

test('a third step splats the accumulated labelled variates by name', async () => {
  // cc | aa, bb ~ N(aa + bb, 1) at aa = 0.5, bb = 0.25, cc = 3.
  const src = 'k(r) = Normal(r.aa, 1.0)\nkc(aa, bb) = Normal(aa + bb, 1.0)\n'
    + 'A = jointchain(aa = Normal(0.0, 1.0), bb = k, cc = kc)\n'
    + 'lp = logdensityof(A, record(aa = 0.5, bb = 0.25, cc = 3.0))\n';
  assert.deepEqual(errors(src), []);
  const want = logN(0.5, 0) + logN(0.25, 0.5) + logN(3.0, 0.75);
  assert.ok(Math.abs(await score(src) - want) < 1e-12, `want ${want}`);
});

test('label-based projection marginalizes the later steps', async () => {
  // pushfwd(get(_, ["aa", "bb"]), A) drops cc, which integrates out to the first two factors.
  const src = 'k(r) = Normal(r.aa, 1.0)\nkc(aa, bb) = Normal(aa + bb, 1.0)\n'
    + 'A = jointchain(aa = Normal(0.0, 1.0), bb = k, cc = kc)\n'
    + 'P = pushfwd(fn(get(_, ["aa", "bb"])), A)\n'
    + 'lp = logdensityof(P, record(aa = 0.5, bb = 0.25))\n';
  assert.deepEqual(errors(src), []);
  const want = logN(0.5, 0) + logN(0.25, 0.5);
  assert.ok(Math.abs(await score(src) - want) < 1e-12, `want ${want}`);
});

test('a labelled chain samples bb given aa', async () => {
  const m = await value('k(r) = Normal(r.aa, 1.0)\nA = jointchain(aa = Normal(0.0, 1.0), bb = k)\ny ~ A\n', 'y', 4000);
  const aa = Array.from(m.fields.aa.samples) as number[];
  const bb = Array.from(m.fields.bb.samples) as number[];
  const diff = bb.map((x, i) => x - aa[i]);
  const mean = diff.reduce((a, b) => a + b, 0) / diff.length;
  const variance = diff.reduce((a, b) => a + (b - mean) ** 2, 0) / diff.length;
  // bb - aa ~ N(0, 1): standard error of the mean 0.016 over 4000 draws.
  assert.ok(Math.abs(mean) < 0.08, `mean ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.12, `variance ${variance}`);
});

// A step input of unknown type (`any`) hides the record from the static check. The record then used
// to reach a distribution parameter or an arithmetic operand and score or sample NaN with no
// diagnostic; the runtime refuses it with a located error instead.
const UNTYPED = [
  ['an fn hole', 'A = jointchain(aa = Normal(0.0, 1.0), bb = fn(Normal(_, 1.0)))', 'Normal', 'parameter "mu"'],
  ['a bare placeholder', 'A = jointchain(aa = Normal(0.0, 1.0), '
    + 'bb = functionof(Normal(mu = _a_, sigma = 1.0), aa = _a_))', 'Normal', 'parameter "mu"'],
  ['an unannotated definition', 'k(aa) = Normal(aa, 1.0)\nA = jointchain(aa = Normal(0.0, 1.0), bb = k)',
    'Normal', 'parameter "mu"'],
  ['an arithmetic operand', 'k(aa) = Normal(aa + 1.0, 1.0)\nA = jointchain(aa = Normal(0.0, 1.0), bb = k)',
    'add', 'the first operand'],
  ['a divisor', 'k(aa) = Normal(1.0 / aa, 1.0)\nA = jointchain(aa = Normal(0.0, 1.0), bb = k)',
    'divide', 'the second operand'],
] as const;

for (const [name, chain, op, slot] of UNTYPED) {
  test(`a record fed to ${name} is refused when scored and when sampled, not NaN`, async () => {
    const want = new RegExp(`${op}.*: ${slot} is a record with fields aa, not a number`);
    await assert.rejects(score(chain + '\nlp = logdensityof(A, record(aa = 0.5, bb = 0.5))\n'), want);
    await assert.rejects(value(chain + '\ny ~ A\n', 'y', 4), want);
  });
}
