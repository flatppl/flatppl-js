'use strict';

// Static diagnostics for refusals that used to surface only at evaluation, or
// not at all, each located on the offending expression:
//   - a qualified name naming no member of its standard module;
//   - a size or count that depends on a draw;
//   - a negative `addaxes` count.
// Each refusing case has a well-formed control that analyses with no error and
// scores against a closed form worked by hand.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');
const typeinfer = require('../typeinfer.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function errors(src: string): any[] {
  return (processSource(src).diagnostics || []).filter((d: any) => d.severity === 'error');
}

async function scalar(src: string, name: string): Promise<number> {
  const { ctx } = makeMatCtx(src);
  const m = await ctx.getMeasure(name);
  return m.value ? m.value.data[0] : m.samples[0];
}

const HEP = 'hepphys = standard_module("particle-physics", "0.1")\n';

// §04 "Name resolution": a name that resolves nowhere is a static error. A
// qualified `alias.member` resolves through the §09 module's members only.
test('an unknown standard-module member is refused statically, once per use', () => {
  const cases: Array<[string, string]> = [
    [`${HEP}x = hepphys.NoSuchThing(1.0)\nM = Normal(x, 1.0)`, 'NoSuchThing'],
    [`${HEP}f = hepphys.NoSuchThing`, 'NoSuchThing'],
    [`${HEP}M = hepphys.NoSuchDist(1.0, 2.0)\ny ~ M`, 'NoSuchDist'],
  ];
  for (const [src, member] of cases) {
    const errs = errors(src);
    assert.equal(errs.length, 1, errs.map((e: any) => e.message).join('; '));
    assert.equal(errs[0].message, `'hepphys.${member}': standard module `
      + `'particle-physics@0.1' has no binding '${member}'`);
    assert.equal(errs[0].loc.start.line, 1);
    assert.equal(errs[0].loc.start.col, 4);
  }
});

// §09 ext-linear-algebra has no qr: qr is the §07 base builtin.
test('a base builtin is not a member of a standard module', () => {
  const src = 'la = standard_module("ext-linear-algebra", "0.1")\n'
    + 'A = rowstack([[3.0, 0.0], [4.0, 5.0]])\nq = la.qr(A)\nr = qr(A)';
  assert.deepEqual(errors(src).map((e: any) => [e.message, e.loc.start.line, e.loc.start.col]), [
    ["'la.qr': standard module 'ext-linear-algebra@0.1' has no binding 'qr'", 2, 4],
  ]);
});

test('a standard module the engine does not provide is refused at each use', () => {
  const errs = errors('h = standard_module("nosuch", "0.1")\nx = h.f(1.0)\ny = h.f(2.0)');
  assert.deepEqual(errs.map((e: any) => [e.message, e.loc.start.line]), [
    ["standard module 'nosuch@0.1' is not provided by this engine", 1],
    ["standard module 'nosuch@0.1' is not provided by this engine", 2],
  ]);
});

// §09 interp_pwlin: for α ≥ 0, center + α·(right − center) = 2 + 0.5·1 = 2.5.
// The §09 distributions are members too, though they live in the density layer.
test('existing members of every kind resolve with no error', async () => {
  const src = `${HEP}v = hepphys.interp_pwlin(1.0, 2.0, 3.0, 0.5)\n`
    + 'g = hepphys.interp_pwlin\n'
    + 'M = hepphys.CrystalBall(0.0, 1.0, 1.5, 3.0)\n'
    + 'B = hepphys.ContinuedPoisson(2.0)';
  assert.deepEqual(errors(src), []);
  assert.equal(await scalar(src, 'v'), 2.5);
});

// A draw has no value at inference, so a size that reads one cannot fold.
test('a stochastic size is refused as stochastic, on the size expression', () => {
  const cases: Array<[string, string, number, number]> = [
    ['n ~ Poisson(3.0)\nM = iid(Normal(0.0, 1.0), n)', "'n'", 1, 26],
    ['n ~ Poisson(3.0)\nm = n + 1\nM = iid(Normal(0.0, 1.0), m)', "'m'", 2, 26],
    ['n ~ Poisson(3.0)\nM = iid(Normal(0.0, 1.0), n + 1)', "'n'", 1, 26],
    ['M = iid(Normal(0.0, 1.0), draw(Poisson(3.0)))', 'an inline draw', 0, 26],
    ['n ~ Poisson(3.0)\nz = zeros(n)', "'n'", 1, 10],
    ['n ~ Poisson(2.0)\nv = [1.0, 2.0]\nw = addaxes(v, n, 0)', "'n'", 2, 15],
  ];
  for (const [src, what, line, col] of cases) {
    const errs = errors(src);
    assert.equal(errs.length, 1, errs.map((e: any) => e.message).join('; '));
    assert.ok(errs[0].message.endsWith(`${what} is stochastic, but a size or count must be fixed`),
      errs[0].message);
    assert.doesNotMatch(errs[0].message, /fixed-phase \(simple\) evaluation/);
    assert.equal(errs[0].loc.start.line, line);
    assert.equal(errs[0].loc.start.col, col);
  }
});

// A size with no draw under it keeps the op-gap wording. The mock resolver
// stands in for an op the fixed-phase evaluator lacks.
test('an op gap in a fixed size is not reported as stochastic', () => {
  const r = processSource('n = 5\nxs = fill(0.0, n + 1)');
  const UNSUP = Symbol('mock-unsupported');
  const mock: any = (ir: any) => (ir && ir.kind === 'call' && ir.op === 'add') ? UNSUP : undefined;
  mock.UNSUPPORTED = UNSUP;
  const errs = typeinfer.inferTypes(r.loweredModule, { resolveFixed: mock })
    .filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1, errs.map((e: any) => e.message).join('; '));
  assert.match(errs[0].message, /not supported in fixed-phase \(simple\) evaluation$/);
});

// log N(0.1; 0, 1) + log N(0.2; 0, 1) + log N(0.3; 0, 1)
//   = −(3/2)·log(2π) − (0.01 + 0.04 + 0.09)/2.
test('a fixed size derived from a fixed binding folds and scores', async () => {
  const src = 'n = 2 + 1\nM = iid(Normal(0.0, 1.0), n)\nL = logdensityof(M, [0.1, 0.2, 0.3])';
  assert.deepEqual(errors(src), []);
  const want = -1.5 * Math.log(2 * Math.PI) - 0.14 / 2;
  assert.ok(Math.abs(await scalar(src, 'L') - want) < 1e-12);
});

// §07 `addaxes(A, n_leading, n_trailing)`: both counts are non-negative integers.
test('a negative addaxes count is refused on the count', () => {
  const cases: Array<[string, string, number, number, number]> = [
    ['v = [1.0, 2.0]\nw = addaxes(v, -1, 0)', 'n_leading', -1, 1, 15],
    ['v = [1.0, 2.0]\nw = addaxes(v, 0, -2)', 'n_trailing', -2, 1, 18],
    ['v = [1.0, 2.0]\nn = 1 - 3\nw = addaxes(v, n, 0)', 'n_leading', -2, 2, 15],
  ];
  for (const [src, name, v, line, col] of cases) {
    const errs = errors(src);
    assert.deepEqual(errs.map((e: any) => [e.message, e.loc.start.line, e.loc.start.col]), [
      [`addaxes: ${name} must be a non-negative integer, got ${v} (spec §07)`, line, col],
    ]);
  }
});

test('non-negative addaxes counts add singleton axes', () => {
  const src = 'v = [1.0, 2.0]\nw = addaxes(v, 1, 2)';
  assert.deepEqual(errors(src), []);
  const { built } = makeMatCtx(src);
  const w = built.fixedValues.get('w');
  assert.deepEqual(w.shape, [1, 2, 1, 1]);
  assert.deepEqual(Array.from(w.data), [1, 2]);
});
