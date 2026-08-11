'use strict';

// §04 auto-splatting for BUILTINS.
//
// §04 "Calling conventions": "`f(record(a = x, b = y, ...))` and `f(table(a = x,
// b = y, ...))` are equivalent to `f(a = x, b = y, ...)`. The order of fields or
// columns is not relevant. A call with field or column names that do not match
// the callable's argument names is a static error." Over the general rule that
// "Arguments are bound to inputs by name".
//
// The engine implemented this for user callables only; for §07 builtins it had
// no argument names at all, so a sole positional record or table could not bind
// and `atan2(record(y = …, x = …))` was rejected although §04 makes it valid.
// The name data now lives in `builtin-param-names.ts`, read from §07's Arguments
// column.
//
// ORACLES are hand arithmetic, never another engine's output:
//   atan2(y, x) — §07 Arguments "y, x"
//     atan2(1, 1) = π/4 = 0.7853981633974483
//     atan2(1, 2) =       0.4636476090008061
//     atan2(2, 1) =       1.1071487177940904   (≠ the previous line, so a
//                                               name↔order slip is visible)
//   pow(base, exponent) — 2^10 = 1024

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('..');

function errorsOf(src: string): string[] {
  return processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}
function valueOf(src: string, name: string) {
  const p = processSource(src);
  const errs = p.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [], 'unexpected errors');
  return orchestrator.buildDerivations(p.bindings).fixedValues.get(name);
}

test('atan2(record(y, x)) splats and binds by NAME (§04)', () => {
  assert.equal(valueOf('s = atan2(record(y = 1.0, x = 2.0))', 's'), 0.4636476090008061);
});

test('the record field ORDER is not relevant (§04)', () => {
  // Same call written both ways; §04: "The order of fields or columns is not
  // relevant." Both must give atan2(1, 2), never atan2(2, 1).
  const a = valueOf('s = atan2(record(y = 1.0, x = 2.0))', 's');
  const b = valueOf('s = atan2(record(x = 2.0, y = 1.0))', 's');
  assert.equal(a, 0.4636476090008061);
  assert.equal(b, a);
});

test('binding is by name, not by position — the swap changes the answer', () => {
  // If the splat bound by column ORDER instead of by name these two would be
  // equal. atan2(y=2, x=1) ≠ atan2(y=1, x=2).
  assert.equal(valueOf('s = atan2(record(y = 2.0, x = 1.0))', 's'), 1.1071487177940904);
  assert.equal(valueOf('s = atan2(record(y = 1.0, x = 2.0))', 's'), 0.4636476090008061);
});

test('a splatted call equals the positional spelling', () => {
  assert.equal(valueOf('s = atan2(record(y = 1.0, x = 2.0))', 's'),
               valueOf('s = atan2(1.0, 2.0)', 's'));
});

test('pow(record(base, exponent)) splats — a second order-sensitive row', () => {
  assert.equal(valueOf('s = pow(record(base = 2.0, exponent = 10.0))', 's'), 1024);
});

test('a record-typed BINDING splats, not just an inline literal', () => {
  assert.equal(valueOf('r = record(y = 1.0, x = 2.0)\ns = atan2(r)', 's'),
               0.4636476090008061);
});

test('a name mismatch is a located §04 static error', () => {
  const errs = errorsOf('s = atan2(record(zzq = 1.0, zzr = 2.0))');
  assert.equal(errs.length, 1, 'got: ' + errs.join(' | '));
  assert.match(errs[0], /spec §04/);
  assert.match(errs[0], /no argument is named "zzq", "zzr"/);
  assert.match(errs[0], /\(y, x\)/, 'the message must name the row arguments');
});

test('a surplus field is a §04 error even when the others match', () => {
  const errs = errorsOf('s = atan2(record(y = 1.0, x = 2.0, zz = 3.0))');
  assert.equal(errs.length, 1, 'got: ' + errs.join(' | '));
  assert.match(errs[0], /no argument is named "zz"/);
});

test('a missing argument is named in the §04 error', () => {
  const errs = errorsOf('s = pow(record(base = 2.0))');
  assert.equal(errs.length, 1, 'got: ' + errs.join(' | '));
  assert.match(errs[0], /nothing binds "exponent"/);
});

test('the nine §04 carve-out names still take the aggregate WHOLE', () => {
  // design#78. `sum`'s only argument is `xs`, so splatting would reject a call
  // §07's "Table reductions" paragraph defines.
  const T = 't = table(mass = [1.0, 2.0, 3.0], pt = [4.0, 5.0, 6.0])\n';
  assert.deepEqual(errorsOf(T + 's = sum(t)'), []);
  assert.deepEqual(valueOf(T + 's = sum(t)', 's'), { mass: 6, pt: 15 });
  assert.deepEqual(valueOf(T + 's = mean(t)', 's'), { mass: 2, pt: 5 });
  assert.equal(valueOf(T + 's = lengthof(t)', 's'), 3);
});

test('an ordinary positional call is untouched', () => {
  assert.equal(valueOf('s = atan2(1.0, 2.0)', 's'), 0.4636476090008061);
  assert.equal(valueOf('s = pow(2.0, 10.0)', 's'), 1024);
});

test('a record bound by keyword is not splatted — but the KEYWORD FORM of a '
  + 'builtin call is itself an unmet §04 gap', () => {
  // §04 excludes a keyword-bound record from splatting ("a record ... bound to a
  // parameter by keyword, is an ordinary value and is not splatted"), and this
  // wave does not splat one. It cannot be pinned positively yet, because §04
  // also says "All built-in ordinary callables ... accept both positional and
  // keyword arguments" and this engine accepts NO keyword form for a §07 builtin
  // whose signature stores its arguments positionally — `atan2(y = 1.0, x = 1.0)`
  // is rejected too. That is the same missing name data this wave adds, applied
  // to a different call form, and it is recorded in TODO-flatppl-js.md rather
  // than fixed here.
  //
  // Pinning today's state so the gap is visible and a future fix flips this.
  const errs = errorsOf('s = atan2(y = 1.0, x = 1.0)');
  assert.equal(errs.length, 1, 'got: ' + errs.join(' | '));
  assert.match(errs[0], /expects 2 positional argument\(s\), got 0/);
  // What matters for THIS wave: the keyword form is not silently mis-splatted.
  assert.doesNotMatch(errs[0], /spec §04/);
});
