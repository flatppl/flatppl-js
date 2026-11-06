'use strict';

// §07 filter applies a predicate per element. A predicate may call helpers
// that read module-level values (spec §04 functions capture the module scope).
// Fixed-phase evaluation of the filter must resolve both the helpers and the
// globals they read.
//
// ORACLE: closed form. cells = [1, 2, 6] select counts [1, 3, 5]; the
// posterior at mu = 2.5 is log Gamma(2.5; 2, 1) + sum log Poisson(k; 2.5).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const logFact = (k: number) => { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; };
const MU = 2.5;
const WANT = Math.log(MU) - MU + [1, 3, 5].reduce((s, k) => s + k * Math.log(MU) - MU - logFact(k), 0);

test('filter with helpers that read a global scalar', async () => {
  const src = [
    'n = 3',
    'counts = rowstack([[1, 2], [3, 4], [0, 5]])',
    'cell_day(k) = mod(k - 1, n) + 1',
    'cell_country(k) = div(k - 1, n) + 1',
    'observed_count(k) = counts[cell_day(k), cell_country(k)]',
    'mask = rowstack([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])',
    'keep(k) = mask[cell_day(k), cell_country(k)] > 0.0',
    'cells = filter(keep, indicesof(zeros(6)))',
    'mu ~ Gamma(shape = 2, rate = 1)',
    'y ~ Poisson.(fill(mu, 3))',
    'L = likelihoodof(kernelof(record(y = y), mu = mu), record(y = observed_count.(cells)))',
    'P = bayesupdate(L, lawof(record(mu = mu)))',
    `lp = logdensityof(P, record(mu = ${MU}))`,
  ].join('\n') + '\n';
  for (const N of [1, 3]) {
    const { ctx } = ctxFor(src, N);
    const m = await ctx.getMeasure('lp');
    const got = Array.from(m.value ? m.value.data : m.samples) as number[];
    assert.ok(got.length > 0);
    got.forEach((g) => assert.ok(Math.abs(g - WANT) < 1e-12, `got ${g}, want ${WANT}`));
  }
});
