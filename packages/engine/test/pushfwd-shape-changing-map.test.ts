'use strict';

// A map that changes the variate's kind, such as `sum` of a vector or `fill`
// of a scalar, has no synthesised inverse: every synthesised forward map keeps
// the variate's kind (§06 "Engine contract for `pushfwd` density evaluation").
// Its density refusal names the map and that rule. It used to report the base's
// footprint ("iid expects array variate, got real") as the reason.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

function errors(measure: string, point: string): string[] {
  const r = processSource(`M = ${measure}\nlp = logdensityof(M, ${point})\n`);
  return r.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

const RULE = /every synthesised forward map preserves kind, so .*bijection\(f, f_inv, logvolume\) \(§06 "Engine contract for `pushfwd` density evaluation"\)/;

test('a typed codomain of another kind is named as the change', () => {
  const errs = errors('pushfwd(fn(fill(_, 3)), Normal(0.0, 1.0))', '[0.5, 0.5, 0.5]');
  assert.equal(errs.length, 1);
  assert.match(errs[0], /pushfwd of this map changes the variate's kind from real to array of real \(length 3\)/);
  assert.match(errs[0], RULE);
});

test('an untyped codomain compares the point with the base variate', () => {
  // `sum` is typed `any`: its result is complex for a complex input.
  for (const map of ['sum', 'fn(sum(_))']) {
    const errs = errors(`pushfwd(${map}, iid(Normal(0.0, 1.0), 3))`, '6.0');
    assert.equal(errs.length, 1, map);
    assert.match(errs[0], /pushfwd of 'sum': the point's kind real does not match the base variate's kind array of real \(length 3\)/, map);
    assert.match(errs[0], RULE, map);
  }
});

test('a compound map is not named by its outer op', () => {
  const errs = errors('pushfwd(fn(2.0 * sum(_)), iid(Normal(0.0, 1.0), 3))', '6.0');
  assert.equal(errs.length, 1);
  assert.match(errs[0], /^logdensityof: pushfwd of this map changes the variate's kind from array of real \(length 3\) to real/);
});

test('a kind-preserving map is not refused', () => {
  assert.deepEqual(errors('pushfwd(exp, Normal(0.0, 1.0))', '2.0'), []);
  assert.deepEqual(errors('pushfwd(fn(exp.(_)), iid(Normal(0.0, 1.0), 3))', '[1.0, 2.0, 3.0]'), []);
});

test('a relabel onto a record is not refused', () => {
  assert.deepEqual(errors('relabel(iid(Normal(0.0, 1.0), 2), ["a", "b"])', 'record(a = 0.1, b = 0.2)'), []);
});
