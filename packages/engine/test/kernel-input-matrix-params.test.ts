'use strict';

// §08 declares MvNormal's `cov` and Wishart/InverseWishart's `scale` as matrices,
// and §03 "Arrays" does not read a vector of vectors as a matrix. The kernel form
// of the §07 primitives takes those parameters in a record; a vector of vectors
// there scored as `undefined` with no diagnostic. It is refused like the
// constructor form. Oracle for the matrix case:
// scipy.stats.multivariate_normal([0, 1], [[2, .5], [.5, 1]]).logpdf([0.3, 0.7])
// = -2.2205421032341994.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const engine = require('../index.ts');
const orchestrator = require('../orchestrator.ts');

function errorsOf(src: string): string[] {
  const r = engine.processSource(`flatppl_compat = "0.1"\n${src}`);
  return r.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

const NESTED = '[[2.0, 0.5], [0.5, 1.0]]';
const REFUSAL = /`MvNormal`'s `cov` must be a matrix \(spec §08\), but this argument is a vector of vectors/;

test('a vector-of-vectors matrix parameter is refused in every kernel-form primitive', () => {
  for (const src of [
    `lp = builtin_logdensityof(MvNormal, record(mu = [0.0, 1.0], cov = ${NESTED}), [0.3, 0.7])`,
    `k = record(mu = [0.0, 1.0], cov = ${NESTED})\nlp = builtin_logdensityof(MvNormal, k, [0.3, 0.7])`,
    `lp = builtin_tonormal(MvNormal, record(mu = [0.0, 1.0], cov = ${NESTED}), [0.3, 0.7])`,
    `s = rnginit(1)\nlp, s2 = builtin_sample(s, MvNormal, record(mu = [0.0, 1.0], cov = ${NESTED}))`,
    `lp = logdensityof(MvNormal([0.0, 1.0], ${NESTED}), [0.3, 0.7])`,
  ]) {
    assert.match(errorsOf(src).join('\n'), REFUSAL, src);
  }
  assert.match(
    errorsOf(`lp = builtin_logdensityof(Wishart, record(nu = 3.0, scale = ${NESTED}), rowstack(${NESTED}))`)
      .join('\n'),
    /`Wishart`'s `scale` must be a matrix/);
});

test('a matrix built with rowstack scores', () => {
  const src = `lp = builtin_logdensityof(MvNormal, record(mu = [0.0, 1.0], cov = rowstack(${NESTED})), [0.3, 0.7])`;
  const r = engine.processSource(`flatppl_compat = "0.1"\n${src}`);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const lp = orchestrator.buildDerivations(r.bindings, r.loweredModule).fixedValues.get('lp');
  assert.ok(Math.abs(lp - -2.2205421032341994) < 1e-12, String(lp));
});
