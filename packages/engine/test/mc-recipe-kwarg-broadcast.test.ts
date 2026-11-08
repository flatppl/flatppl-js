// `broadcast(transport, x = xs, pars = [pars])` is the same broadcast as
// `transport.(xs, [pars])` (§04), but `_buildFromGenerative` mapped only
// positional args to kernel params, so the keyword spelling got no
// MC-marginal density form. Oracle: the positional spelling's form.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource, orchestrator } = require('../index.ts');
const { buildMcMarginalForm } = require('../mc-recipe.ts');

const FIXTURE = fs.readFileSync(path.join(__dirname, 'fixtures', 'simple-transport1.flatppl'), 'utf8');
const POSITIONAL = 'ys ~ transport.(xs, [pars])';

function formFor(line: string) {
  const r = processSource(FIXTURE.replace(POSITIONAL, line));
  const built = orchestrator.buildDerivations(r.bindings);
  const env = { derivations: built.derivations, bindings: built.bindings };
  return buildMcMarginalForm(orchestrator.expandMeasure('zs', env), built.bindings,
    (nm: string) => orchestrator.expandMeasure(nm, env));
}

test('the keyword broadcast of a generative kernel gets the same MC-marginal form', () => {
  assert.ok(FIXTURE.includes(POSITIONAL));
  const want = formFor(POSITIONAL);
  const got = formFor('ys ~ broadcast(transport, x = xs, pars = [pars])');
  assert.ok(got, 'no MC-marginal form for the keyword spelling');
  const strip = (f: any) => JSON.parse(JSON.stringify(f, (k, v) => (k === 'loc' ? undefined : v)));
  assert.deepEqual(strip(got), strip(want));
});
