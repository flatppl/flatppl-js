'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { toJS } = require('./_value-helpers.ts');
const { createWorkerHandler } = require('../worker.ts');

// §04 Reductions: each binder owns its accumulator and input at each step.
for (const { name, source, expected } of [
  {
    name: 'reduce',
    source: `
inner = (a, b) -> (a + b) + 0 * (a + b)
step = (s, x) -> (s + x) + (s + x) + reduce(inner, [x, 2, 3])
result = scan(step, 0, [1, 2, 3])
`,
    expected: [8, 27, 68],
  },
  {
    name: 'scan',
    source: `
inner = (a, b) -> (a + b) + (a + b)
step = (s, x) -> (s + x) + (s + x) + sum(scan(inner, 0, [x, 1]))
result = scan(step, 0, [1, 2, 3])
`,
    expected: [10, 38, 102],
  },
]) {
  test(`scan keeps nested ${name} iterations in distinct frames`, async () => {
    const { ctx } = makeMatCtx(source);
    assert.deepEqual(toJS((await ctx.getMeasure('result')).value), expected);
  });
}

test('scan advances explicit RNG state like sequential calls', async () => {
  const { ctx } = makeMatCtx(`
s0 = rnginit([4, 5, 6, 7])
step = (s, x) -> record(rstate = rand(s.rstate, Normal(0, 1))[2],
    value = rand(s.rstate, Normal(0, 1))[1])
states = scan(step, record(rstate = s0, value = 0), [0, 0, 0])
value = s -> s.value
result = value.(states)
x1, s1 = rand(s0, Normal(0, 1))
x2, s2 = rand(s1, Normal(0, 1))
x3, s3 = rand(s2, Normal(0, 1))
expected = [x1, x2, x3]
`);
  assert.deepEqual(toJS((await ctx.getMeasure('result')).value),
    toJS((await ctx.getMeasure('expected')).value));
});

test('scan profile recovers after a step throws', () => {
  const ref = (name: string) => ({ kind: 'ref', ns: '%local', name });
  const lit = (value: any) => ({ kind: 'lit', value });
  const call = (op: string, args: any[]) => ({ kind: 'call', op, args });
  const term = () => call('add', [ref('s'), ref('x')]);
  const step = { kind: 'call', op: 'functionof', params: ['s', 'x'], body:
    call('checked', [call('add', [term(), term()]), call('gt', [ref('x'), lit(0)])]) };
  const ir = call('sum', [call('scan', [step, lit(0), call('vector', [ref('input'), lit(1)])])]);
  const worker = createWorkerHandler();
  const reply = worker.handle({ type: 'profileN', mode: 'function', ir,
    sweepName: 'input', range: [-1, 2], count: 4 });
  assert.deepEqual(Array.from(reply.samples), [NaN, NaN, 8, 14]);
});
