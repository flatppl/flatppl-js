'use strict';

// =====================================================================
// Fixed-phase evaluation of a HIGHER-ORDER op, and `x in S` for a set
// held in a binding.
// =====================================================================
//
// §04's phases make a size expression a fixed-phase value position, and the
// demand-driven resolver (engine-concepts §17.1) folds it so typeinfer can
// resolve the shape. Two things stopped it for the region-restricted idiom
// `n = lengthof(filter(fn(_ in window), data))`:
//
//   * `_evalCall` evaluated EVERY argument to a value before dispatching. A
//     higher-order op's callable argument has no value, so the synthesised
//     call lost the reification's params and body, `evaluateExpr` rejected an
//     operand-less `functionof`, and the resolver mapped that to its
//     UNSUPPORTED op-gap sentinel — reporting "could not compute the value of
//     'n' … an operation not supported in fixed-phase (simple) evaluation" for
//     an op the sampler implements.
//
//   * `x in S` read its right operand as a set DESCRIPTOR IR, so a named set
//     resolved but a set held in a BINDING did not. §03 makes a set a value,
//     and `window = interval(-3.0, 3.0)` is the idiom's own spelling.
//
// Both are checked here against hand-computed answers, with no dependency on
// the sibling corpus checkout.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const fixedEval = require('../fixed-eval.ts');
const sampler = require('../sampler.ts');

function resolve(src: string, name: string): any {
  const lifted = processSource(src);
  const errs = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [], 'source analyses cleanly');
  const res = fixedEval.makeResolver({ loweredModule: lifted.loweredModule });
  return res({ kind: 'ref', ns: 'self', name });
}

test('a size derived through `filter` folds in fixed-phase evaluation', () => {
  const v = resolve(`
obs_values = [8.2, 9.4, 12.1, 1.0, -2.5]
window = interval(-3.0, 3.0)
kept = filter(fn(_ in window), obs_values)
n = lengthof(kept)
`, 'n');
  // 1.0 and -2.5 lie in [-3, 3]; 8.2, 9.4 and 12.1 do not.
  assert.equal(v, 2);
});

test('an EMPTY filter result folds to size 0, not to an op gap', () => {
  const src = `
obs_values = [8.2, 9.4, 12.1]
window = interval(-3.0, 3.0)
kept = filter(fn(_ in window), obs_values)
n = lengthof(kept)
`;
  const v = resolve(src, 'n');
  assert.notEqual(v, fixedEval.UNSUPPORTED,
    'a supported op must not report the fixed-phase op gap');
  assert.equal(v, 0);
});

test('a NAMED predicate binding resolves the same as an inline one', () => {
  const v = resolve(`
xs = [1.0, 5.0, 9.0]
lo_hi = interval(0.0, 6.0)
keep = fn(_ in lo_hi)
kept = filter(keep, xs)
n = lengthof(kept)
`, 'n');
  assert.equal(v, 2);
});

test('an unresolvable body ref leaves the size dynamic, not an op gap', () => {
  // `cut` is a free parameter, so the predicate has no fixed-phase value. That
  // is legitimately %dynamic (`undefined`), never the UNSUPPORTED sentinel.
  const v = resolve(`
xs = [1.0, 5.0, 9.0]
cut = elementof(reals)
kept = filter(fn(_ > cut), xs)
n = lengthof(kept)
`, 'n');
  assert.equal(v, undefined);
});

test('`in` tests membership of a set held in a binding', () => {
  const inIR = (x: number) => ({
    kind: 'call', op: 'in',
    args: [{ kind: 'lit', value: x }, { kind: 'ref', ns: 'self', name: 'win' }],
  });
  const env = { win: { kind: 'interval', a: -3, b: 3 } };
  // §03: `interval(a, b)` is the CLOSED interval, so both endpoints are in.
  assert.equal(sampler.evaluateExpr(inIR(0), env), true);
  assert.equal(sampler.evaluateExpr(inIR(-3), env), true);
  assert.equal(sampler.evaluateExpr(inIR(3), env), true);
  assert.equal(sampler.evaluateExpr(inIR(3.0001), env), false);
  assert.equal(sampler.evaluateExpr(inIR(-8.2), env), false);
});

test('`in` still refuses a set it cannot resolve', () => {
  assert.throws(() => sampler.evaluateExpr({
    kind: 'call', op: 'in',
    args: [{ kind: 'lit', value: 0 }, { kind: 'ref', ns: 'self', name: 'nope' }],
  }, {}), /unsupported set shape/);
});

test('a named set keeps its own reading, not the env lookup', () => {
  // `posreals` is a NAMED set; an env entry of the same name must not shadow
  // the spec's own membership rule.
  const ir = {
    kind: 'call', op: 'in',
    args: [{ kind: 'lit', value: -1 }, { kind: 'ref', ns: 'self', name: 'posreals' }],
  };
  assert.equal(sampler.evaluateExpr(ir, { posreals: { kind: 'interval', a: -5, b: 5 } }),
    false);
});

test('a caller-supplied env wins over the binding a body ref would resolve', () => {
  // Ref precedence in the resolver: caller env > baseEnv > binding cache. A
  // callable body's free refs follow the same order, so a local scope (the one
  // typeinfer passes for aggregate axis vars) overrides the module binding.
  const lifted = processSource(`
xs = [1.0, 5.0, 9.0]
win = interval(0.0, 2.0)
kept = filter(fn(_ in win), xs)
n = lengthof(kept)
`);
  const res = fixedEval.makeResolver({ loweredModule: lifted.loweredModule });
  const bare = res({ kind: 'ref', ns: 'self', name: 'kept' });
  assert.equal(bare.data.length, 1, 'only 1.0 is inside [0, 2]');
  const widened = res(lifted.loweredModule.bindings.get('kept').rhs,
    { win: { kind: 'interval', a: 0, b: 10 } });
  assert.equal(widened.data.length, 3, 'the caller env widened the window');
});
