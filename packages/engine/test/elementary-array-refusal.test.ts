'use strict';

// §07's elementary functions take scalars. In source, typeinfer refuses an
// array operand once, also under a reduction. At run time, internally built
// IR gets a located refusal instead of a NaN read of the container.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const sampler = require('../sampler.ts');
const vl = require('../value.ts');

for (const red of ['sum', 'mean', 'prod']) {
  test(`${red}(exp(v)) reports the scalar-only refusal once`, () => {
    const errs = processSource(`v = [1.0, 2.0]\ny = ${red}(exp(v))\n`).diagnostics
      .filter((d: any) => d.severity === 'error');
    assert.equal(errs.length, 1, JSON.stringify(errs.map((d: any) => d.message)));
    assert.match(errs[0].message, /^exp: arg 1 expects a scalar/);
  });
}

test('a table reduction keeps its column-wise type', () => {
  const r = processSource('t = table(a = [1.0, 2.0])\ny = sum(t)\n');
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal(r.bindings.get('y').inferredType.kind, 'record');
});

test('an elementary call over an array in internal IR is refused, not NaN', () => {
  const ir = { kind: 'call', op: 'exp', args: [{ kind: 'ref', ns: 'self', name: 'v' }] };
  assert.throws(() => sampler.evaluateExpr(ir, { v: vl.vector([0, 1]) }),
    /^Error: exp: §07 gives exp scalars only, got an array of shape \[2\]/);
  assert.equal(sampler.evaluateExpr(ir, { v: 0 }), 1);
});
