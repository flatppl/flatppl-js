'use strict';

// §06 `likelihoodof`: the parameter point is a record with one field per
// input, bound by name for any number of inputs. `joint_likelihood` sums its
// terms at one point, so the point binds by name over the union of the terms'
// inputs, also when a term's kernel declares its input positionally.
//
// ORACLE: closed form, log N(1.5; theta, 1) + log N(3.2; 2 mu, 0.5).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const H = 'k(theta) = Normal(theta, 1.0)\nmu = elementof(reals)\n'
  + 'm2 = functionof(Normal(mu = 2.0 * mu, sigma = 0.5))\n'
  + 'L = joint_likelihood(likelihoodof(k, 1.5), likelihoodof(m2, 3.2))\n';

const errors = (point: string) => processSource(H + `lp = logdensityof(L, ${point})\n`)
  .diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);

test('a joint point with a field no term takes is a static error', () => {
  const errs = errors('record(theta = 0.3, mu = 0.1, zz = 1.0)');
  assert.equal(errs.length, 1, JSON.stringify(errs));
  assert.match(errs[0], /^logdensityof: the point has no input named "zz"/);
});

test('a joint point missing an input of a positional term is a static error', () => {
  const errs = errors('record(theta = 0.3)');
  assert.equal(errs.length, 1, JSON.stringify(errs));
  assert.match(errs[0], /^logdensityof: the point has no field for the likelihood input "mu"/);
});

test('the full joint point scores the sum of the terms', async () => {
  const src = H + 'lp = logdensityof(L, record(theta = 0.3, mu = 0.1))\n';
  assert.deepEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error'), []);
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('lp');
  const got = (m.value ? m.value.data : m.samples)[0];
  const logN = (x: number, mean: number, s: number) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - mean) / s) ** 2;
  const want = logN(1.5, 0.3, 1) + logN(3.2, 0.2, 0.5);
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
});
