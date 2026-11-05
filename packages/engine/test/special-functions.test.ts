'use strict';

// Spec §09 `special-functions` 0.1: erf and erfc.
// Oracle: scipy.special 1.18 for the core, mpmath (40 digits) for the erfc
// tail, where scipy flushes erfc(27) to 0. scipy.stats.norm for the density.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const sm = require('../standard-modules.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const SF = 'sf = standard_module("special-functions", "0.1")\n';

function impl(name: string) {
  const b = sm.lookupStandardModule('special-functions', '0.1')?.bindings.get(name);
  assert.ok(b && typeof b.impl === 'function', `${name} has an impl`);
  return b.impl;
}

function close(got: number, want: number, rtol: number, what: string) {
  const err = want === 0 ? Math.abs(got) : Math.abs(got - want) / Math.abs(want);
  assert.ok(err <= rtol, `${what}: got ${got}, want ${want}`);
}

async function value(src: string, name: string): Promise<number[]> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure(name);
  const v = m.samples ?? m.value?.data ?? m.value;
  return typeof v === 'number' ? [v] : Array.from(v as ArrayLike<number>);
}

test('erf and erfc match the oracle at negative, zero and positive x', () => {
  const cases: Array<[number, number, number]> = [
    [-1.3, -0.9340079449406524, 1.9340079449406524],
    [0.0, 0.0, 1.0],
    [0.7, 0.6778011938374184, 0.32219880616258156],
  ];
  for (const [x, erf, erfc] of cases) {
    close(impl('erf')(x), erf, 1e-15, `erf(${x})`);
    close(impl('erfc')(x), erfc, 1e-15, `erfc(${x})`);
  }
  // Odd symmetry near zero, where 1 - erfc(x) would lose all relative digits.
  close(impl('erf')(1e-10), 1.1283791670955126e-10, 1e-15, 'erf(1e-10)');
});

test('erfc keeps the upper tail that 1 - erf(x) loses', () => {
  assert.equal(1 - impl('erf')(10), 0);
  close(impl('erfc')(10), 2.088487583762545e-45, 1e-15, 'erfc(10)');
  close(impl('erfc')(26), 5.663192408856143e-296, 1e-15, 'erfc(26)');
  // A subnormal: only about 5 significant digits survive.
  close(impl('erfc')(27), 5.237048923789256e-319, 1e-4, 'erfc(27)');
});

test('module calls evaluate, with a named argument and under broadcast', async () => {
  const src = SF + 'a = sf.erf(0.7)\nb = sf.erfc(x = 10.0)\nc = sf.erfc.([-1.3, 0.0, 26.0])\n';
  assert.deepEqual(processSource(src).diagnostics, []);
  close((await value(src, 'a'))[0], 0.6778011938374184, 1e-15, 'a');
  close((await value(src, 'b'))[0], 2.088487583762545e-45, 1e-15, 'b');
  const c = await value(src, 'c');
  [1.9340079449406524, 1.0, 5.663192408856143e-296].forEach((want, i) =>
    close(c[i], want, 1e-15, `c[${i}]`));
});

// Oracle: scipy.stats.norm.logpdf(0.5, loc=erf(0.3), scale=1).
const PROBIT_LD = -0.9336229269914126;

test('erf feeds a density parameter', async () => {
  const src = SF + 'mu ~ Normal(0, 1)\nx ~ Normal(mu = sf.erf(mu), sigma = 1.0)\n'
    + 'fk = kernelof(record(x = x), mu = mu)\nL = likelihoodof(fk, record(x = 0.5))\n'
    + 'ld = logdensityof(L, record(mu = 0.3))\n';
  close((await value(src, 'ld'))[0], PROBIT_LD, 1e-14, 'ld');
});

// The FlatPDL shape `flatppl determinize` emits: the module call stays a call.
test('determinised output scores the kept module call', async () => {
  const src = SF + 'mu = 0.3\n'
    + 'lp = builtin_logdensityof(Normal, record(mu = sf.erf(mu), sigma = 1.0), 0.5)\n';
  assert.deepEqual(processSource(src).diagnostics, []);
  close((await value(src, 'lp'))[0], PROBIT_LD, 1e-14, 'lp');
});

// Oracle: mpmath at 60 digits, exp(x²)·erfc(x).
test('erfcx stays finite where erfc underflows', async () => {
  const cases: Array<[number, number]> = [
    [-1.3, 10.481318741176324], [0.0, 1.0], [0.7, 0.525930337349441],
    [10, 0.05614099274382259], [27, 0.02088160799042094], [1e4, 5.641895807268084e-05],
  ];
  for (const [x, want] of cases) close(impl('erfcx')(x), want, 1e-15, `erfcx(${x})`);
  const src = SF + 'a = sf.erfcx(x = 27.0)\n';
  assert.deepEqual(processSource(src).diagnostics, []);
  close((await value(src, 'a'))[0], 0.02088160799042094, 1e-15, 'a');
});

test('an unimplemented §09 member refuses by name', () => {
  const { diagnostics } = processSource(SF + 'a = sf.bessel_j(1.0, 2.0)\n');
  assert.equal(diagnostics.length, 1);
  assert.match(diagnostics[0].message,
    /'special-functions\.bessel_j' is not implemented by this engine/);
});
