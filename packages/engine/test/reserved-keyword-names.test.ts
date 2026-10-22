'use strict';

// Spec §05: `KeywordArg ::= Name "=" Expression`, and the reserved words "are
// recognized before `Name`", so none of them names a keyword argument.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

const RESERVED = ['in', 'true', 'false', 'all', 'only'];

function errors(source: string) {
  return processSource(source).diagnostics.filter((d: any) => d.severity === 'error');
}

test('a reserved word is not a keyword argument name', () => {
  for (const word of RESERVED) {
    for (const source of [
      `g(x) = x\nz = g(${word} = 2.0)`,
      `y ~ Normal(mu = 0.0, ${word} = 1.0)`,
      `r = record(${word} = 1.0)`,
      `r = (${word} = 1.0)`,
      `r = (a = 2.0, ${word} = 1.0)`,
      `f = functionof(_x_ + 1.0, ${word} = _x_)`,
      `a = elementof(reals)\nk = kernelof(draw(Normal(a, 1.0)), ${word} = a)`,
    ]) {
      const found = errors(source).find((d: any) => d.message
        === `\`${word}\` is a reserved word and cannot be a keyword argument name (spec §05)`);
      assert.ok(found, source);
      assert.ok(found.loc, `${source}: error is not located`);
    }
  }
});

test('self, base and reserved prefixes stay legal keyword names', () => {
  for (const source of [
    'f = functionof(_x_ + 1.0, self = _x_)\nz = f(self = 2.0)',
    'r = (self = 1.0, base = 2.0)',
    'r = record(inside = 1.0, allx = 2.0, only_one = 3.0)',
  ]) {
    assert.deepEqual(errors(source), [], source);
  }
});
