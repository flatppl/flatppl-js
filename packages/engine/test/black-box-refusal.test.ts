'use strict';

// A black-box interface (flatppl-bridge) is a module of `external(anything)`
// members. Only a host that binds it to a box server can query it, and this
// engine is not a box client. Every use of a member must refuse with an
// error that names the member and the remedy, never evaluate.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');

const INTERFACE = 'posterior = external(anything)\nK = external(anything)\nf = external(anything)\n';

function errors(model: string, sources: Record<string, string> = { 'iface.flatppl': INTERFACE }) {
  const r = processSource(model, { bundle: { sources } });
  return (r.diagnostics || []).filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

for (const [use, member] of [
  ['lp = logdensityof(bb.posterior, [0.1, 0.2])', 'posterior'],
  ['y ~ bb.K(mu = 0.0, sigma = 1.0)', 'K'],
  ['z = bb.f([1.0, 2.0, 3.0])', 'f'],
  ['z = rand(bb.posterior, rnginit(0))', 'posterior'],
]) {
  test(`a black-box member refuses: ${use}`, () => {
    const found = new Set(errors(`bb = load_module("iface.flatppl")\n${use}\n`));
    assert.equal(found.size, 1, String([...found]));
    const [message] = found;
    assert.match(message, new RegExp(`'bb\\.${member}' is a black-box member`));
    assert.match(message, /does not query black boxes/);
    assert.match(message, /flatppl-python/);
  });
}

test('a substituted external(anything) input is an ordinary value', () => {
  const found = errors(
    'm = load_module("lib.flatppl", c = 2.0)\nz = m.y\n',
    { 'lib.flatppl': 'c = external(anything)\ny = c + 1.0\n' },
  );
  assert.deepEqual(found, []);
});

test('an external with a declared set is not a black-box member', () => {
  const found = errors(
    'm = load_module("lib.flatppl")\nz = m.c\n',
    { 'lib.flatppl': 'c = external(reals)\n' },
  );
  assert.ok(!found.some((message: string) => message.includes('black-box')), String(found));
});
