'use strict';

// A scan with a record accumulator returns a table (§06: record trajectories of
// kscan, "a stochastic version of scan", and of markovchain are tables). Its
// columns, rows and row-wise broadcast follow §03 tables.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('..');
const { toJS } = require('./_value-helpers.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const STEP = `
step(state, x) = record(total = state.total + x, inner = record(n = state.inner.n + 1))
states = scan(step, record(total = 0.0, inner = record(n = 0)), [2.0, -1.0, 4.0])
`;

function fixed(source: string) {
  const lifted = processSource(source);
  assert.deepEqual(lifted.diagnostics.filter((d: any) => d.severity === 'error'), []);
  return orchestrator.buildDerivations(lifted.linkedBindings).fixedValues;
}

test('a column read gives the column as a vector', () => {
  const values = fixed(`${STEP}totals = states.total\n`);
  assert.deepEqual(toJS(values.get('totals')), [2, 1, 5]);
});

test('a row read gives a record', () => {
  const values = fixed(`${STEP}second = states[2]\n`);
  assert.deepEqual(toJS(values.get('second')), { total: 1, inner: { n: 2 } });
});

test('a broadcast traverses the rows', () => {
  const values = fixed(`${STEP}total(row) = row.total\ntotals = total.(states)\n`);
  assert.deepEqual(toJS(values.get('totals')), [2, 1, 5]);
});

test('a nested record field is a nested table column', () => {
  const values = fixed(`${STEP}counts = states.inner.n\n`);
  const states = values.get('states');
  assert.equal(states.columns.inner.__table__, true);
  assert.deepEqual(toJS(values.get('counts')), [1, 2, 3]);
});

test('an empty record scan is an empty table with the same columns', () => {
  const values = fixed(
    'step(state, x) = record(total = state.total + x)\n'
    + 'states = scan(step, record(total = 0.0), fill(0.0, 0))\nn = lengthof(states)\n');
  const states = values.get('states');
  assert.equal(states.__table__, true);
  assert.deepEqual(Object.keys(states.columns), ['total']);
  assert.equal(toJS(values.get('n')), 0);
});

// The R9c chain-sample lowering reads the trajectory column and the successor
// rng state from the last row.
test('an rng-carrying scan reads its value column and last rng state', async () => {
  const { ctx } = makeMatCtx(`
s0 = rnginit([4, 5, 6, 7])
step = (c, x) -> record(state = rand(Normal(c.state, 1), c.rng)[1], rng = rand(Normal(c.state, 1), c.rng)[2])
traj = scan(step, record(state = 0.0, rng = s0), [0, 0, 0])
values = traj.state
last = get0(traj.rng, 2)
x1, s1 = rand(Normal(0.0, 1), s0)
x2, s2 = rand(Normal(x1, 1), s1)
x3, s3 = rand(Normal(x2, 1), s2)
expected = [x1, x2, x3]
y, t = rand(Normal(0, 1), last)
z, u = rand(Normal(0, 1), s3)
`);
  assert.deepEqual(toJS((await ctx.getMeasure('values')).value),
    toJS((await ctx.getMeasure('expected')).value));
  assert.equal(toJS((await ctx.getMeasure('y')).value), toJS((await ctx.getMeasure('z')).value));
});
