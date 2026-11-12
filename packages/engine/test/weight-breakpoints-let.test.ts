'use strict';
// The jump detector reads a `let` by scope: an inner assign that reuses a name
// shadows the outer one only inside its body.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { weightDiscontinuities } = require('../weight-breakpoints.ts');
const { evaluateExpr } = require('../sampler.ts');

const lit = (value: number) => ({ kind: 'lit', value });
const ref = (name: string) => ({ kind: 'ref', ns: '%local', name });
const call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });
const let_ = (name: string, value: any, body: any) => ({ kind: 'call', op: 'let', assigns: [{ name, value }], body });
const isV = (n: any) => !!n && n.kind === 'ref' && n.name === 'v';
const mentions = (n: any): boolean => {
  if (!n || typeof n !== 'object') return false;
  if (Array.isArray(n)) return n.some(mentions);
  if (isV(n)) return true;
  for (const k in n) if (k !== 'loc' && k !== 'meta' && mentions(n[k])) return true;
  return false;
};
const window = (t: any) => call('ifelse', call('lt', call('abs', call('sub', t, lit(1))), lit(1e-4)), lit(101), lit(1));

test('a shadowed let name gives the cuts of its own scope', () => {
  // t = 2v outside, t = 4v inside: |t − 1| < 1e-4 is v = 1/2 ± 5e-5 and
  // v = 1/4 ± 2.5e-5.
  const body = let_('t', call('mul', lit(2), ref('v')),
    call('add', window(ref('t')), let_('t', call('mul', lit(4), ref('v')), window(ref('t')))));
  const d = weightDiscontinuities(body, isV, mentions);
  assert.equal(d.refuse, null);
  const got = d.cuts.map((c: any) => +evaluateExpr(c, {})).sort((p: number, q: number) => p - q);
  const want = [0.25 - 2.5e-5, 0.25 + 2.5e-5, 0.5 - 5e-5, 0.5 + 5e-5];
  assert.equal(got.length, want.length, 'cuts ' + got);
  got.forEach((c: number, i: number) => assert.ok(Math.abs(c - want[i]) < 1e-15, c + ' vs ' + want[i]));
});

test('a let name read as a test bound or a set resolves to its value', () => {
  // lo = 0.2 is bound once. v < lo cuts at 0.2, v in interval(lo, 0.7) at
  // 0.2 and 0.7.
  const body = let_('lo', lit(0.2), call('add',
    call('ifelse', call('lt', ref('v'), ref('lo')), lit(1), lit(2)),
    let_('S', call('interval', ref('lo'), lit(0.7)), call('ifelse', call('in', ref('v'), ref('S')), lit(1), lit(2)))));
  const d = weightDiscontinuities(body, isV, mentions);
  assert.equal(d.refuse, null);
  const got = d.cuts.map((c: any) => +evaluateExpr(c, {})).sort((p: number, q: number) => p - q);
  assert.deepEqual(got, [0.2, 0.2, 0.7]);
});
