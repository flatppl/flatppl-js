'use strict';

// =====================================================================
// fusion-b-kernel-broadcast.test.ts — kernel-broadcast inlining MVP
// =====================================================================
//
// Pins `_tryDissolveKernelBroadcast` in dissolver.ts (engine-concepts
// §20.10 / TODO-flatppl-js fusion thread (b)). The rewrite catches:
//
//   broadcast(<ref to user kernel binding>, args…)
//     where kernel binding's IR is
//       functionof(lawof(<BuiltinDist>(<kw…>)), <params…>)
//
// and rewrites to `broadcast(<BuiltinDist>, <substituted-kwargs>)` —
// directly invocable by matKernelBroadcast since the head is now a
// builtin dist name.
//
// MVP scope:
//   - Head is a self-ref to a `kernelof` binding (which lowered to
//     functionof(lawof(<single_dist>), kw…)).
//   - Inner measure is a SINGLE builtin distribution call (op name
//     uppercase; not a higher-order op).
//   - Args mappable positionally OR by kwarg (matching paramKwargs).
//
// Refusals:
//   - Inner is a measure-algebra expression (joint / iid / weighted
//     / …).
//   - Inner is a higher-order op (broadcast / aggregate inside the
//     kernel body).
//   - Mixed positional + kwarg broadcast args.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { _tryDissolveKernelBroadcast } = require('../dissolver.ts');

function mkBindings(entries: Array<[string, any]>) {
  const m = new Map();
  for (const [name, b] of entries) m.set(name, b);
  return m;
}

function loc(name: string) { return { kind: 'ref', ns: '%local', name }; }
function ref(name: string) { return { kind: 'ref', ns: 'self', name }; }
function lit(v: any, t = 'real') { return { kind: 'lit', value: v, numType: t }; }
function call(op: string, ...args: any[]) { return { kind: 'call', op, args }; }
function callKW(op: string, kwargs: Record<string, any>) {
  return { kind: 'call', op, kwargs };
}

// Helper: build a kernelof binding IR. `params` are the parameter
// names visible as %local inside the body; `paramKwargs` are surface
// names callers use at call sites. `inner` is the wrapped measure.
function kernelofBinding(
  params: string[], paramKwargs: string[], inner: any,
) {
  return {
    kind: 'call', op: 'functionof',
    params,
    paramKwargs,
    body: { kind: 'call', op: 'lawof', args: [inner] },
  };
}

// =====================================================================
// 1. Canonical fire: kernelof(Normal(mu=c, sigma=1), c)
// =====================================================================

test('fusion (b) MVP: broadcast over kernelof(Normal(mu=c, sigma=1)) — kwarg', () => {
  // c is the kernel's input parameter. broadcast supplies c via
  // kwarg form. The rewrite unwraps the kernelof and emits
  // broadcast(Normal, mu=<bc-arg>, sigma=1).
  const kIR = kernelofBinding(['c'], ['c'], callKW('Normal', {
    mu: loc('c'),
    sigma: lit(1),
  }));
  const bindings = mkBindings([['mykernel', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('mykernel')],
    kwargs: { c: ref('C') },
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.ok(r, 'rewrite should fire');
  assert.equal(r.op, 'broadcast');
  assert.equal(r.args[0].kind, 'ref');
  assert.equal(r.args[0].name, 'Normal');
  assert.equal(r.kwargs.mu.name, 'C');
  assert.equal(r.kwargs.sigma.value, 1);
});

test('fusion (b) MVP: positional broadcast arg form', () => {
  // broadcast(mykernel, C) with positional C.
  const kIR = kernelofBinding(['c'], ['c'], callKW('Normal', {
    mu: loc('c'),
    sigma: lit(2.5),
  }));
  const bindings = mkBindings([['mykernel', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('mykernel'), ref('C')],
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.ok(r, 'positional form fires');
  assert.equal(r.kwargs.mu.name, 'C');
  assert.equal(r.kwargs.sigma.value, 2.5);
});

// =====================================================================
// 2. Multi-parameter kernel
// =====================================================================

test('fusion (b) MVP: kernelof with two params (m, s) → Normal(mu=m, sigma=s)', () => {
  const kIR = kernelofBinding(
    ['m', 's'], ['m', 's'],
    callKW('Normal', { mu: loc('m'), sigma: loc('s') }),
  );
  const bindings = mkBindings([['nk', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('nk')],
    kwargs: { m: ref('M'), s: ref('S') },
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.ok(r);
  assert.equal(r.kwargs.mu.name, 'M');
  assert.equal(r.kwargs.sigma.name, 'S');
});

// =====================================================================
// 3. Closure-over-scope (self-ref preserved)
// =====================================================================

test('fusion (b) MVP: closed-over self-ref in kernel body preserved', () => {
  // mykernel = kernelof(Normal(mu = c, sigma = self.sigma_global), c)
  // — sigma comes from outer scope, not a kernel param.
  const kIR = kernelofBinding(
    ['c'], ['c'],
    callKW('Normal', { mu: loc('c'), sigma: ref('sigma_global') }),
  );
  const bindings = mkBindings([['mk', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('mk')],
    kwargs: { c: ref('C') },
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.ok(r);
  // sigma stays as a self-ref to sigma_global.
  assert.equal(r.kwargs.sigma.kind, 'ref');
  assert.equal(r.kwargs.sigma.name, 'sigma_global');
});

// =====================================================================
// 4. Refusals
// =====================================================================

test('fusion (b) MVP: refuse non-builtin-looking inner op (lowercase name)', () => {
  // A lowercase op name signals a value-domain op, not a dist.
  const kIR = kernelofBinding(['c'], ['c'], callKW('mything', { x: loc('c') }));
  const bindings = mkBindings([['mk', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('mk')],
    kwargs: { c: ref('C') },
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.equal(r, null);
});

test('fusion (b) MVP: refuse inner measure-algebra op (iid)', () => {
  const kIR = kernelofBinding(
    ['c'], ['c'],
    call('iid', call('Normal', loc('c'), lit(1)), lit(5)),
  );
  const bindings = mkBindings([['mk', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('mk')],
    kwargs: { c: ref('C') },
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.equal(r, null);
});

test('fusion (b) MVP: refuse inner higher-order op (broadcast)', () => {
  const kIR = kernelofBinding(
    ['c'], ['c'],
    { kind: 'call', op: 'broadcast', args: [ref('Normal'), loc('c'), lit(1)] },
  );
  const bindings = mkBindings([['mk', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('mk')],
    kwargs: { c: ref('C') },
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.equal(r, null);
});

test('fusion (b) MVP: refuse non-functionof head binding', () => {
  // Binding IR is a plain lit, not a kernelof / functionof.
  const bindings = mkBindings([['notakernel', { ir: lit(5) }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('notakernel'), ref('C')],
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.equal(r, null);
});

test('fusion (b) MVP: refuse functionof body that is not lawof(<Dist>)', () => {
  // body is `add(_local, 1)` — a function not a kernel.
  const kIR = {
    kind: 'call', op: 'functionof',
    params: ['c'],
    paramKwargs: ['c'],
    body: call('add', loc('c'), lit(1)),
  };
  const bindings = mkBindings([['myf', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('myf'), ref('C')],
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.equal(r, null);
});

test('fusion (b) MVP: refuse mixed positional + kwarg broadcast args', () => {
  const kIR = kernelofBinding(['m', 's'], ['m', 's'],
    callKW('Normal', { mu: loc('m'), sigma: loc('s') }));
  const bindings = mkBindings([['nk', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('nk'), ref('M')],   // positional m
    kwargs: { s: ref('S') },        // kwarg s
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.equal(r, null);
});

test('fusion (b) MVP: refuse arity mismatch (positional)', () => {
  const kIR = kernelofBinding(['m', 's'], ['m', 's'],
    callKW('Normal', { mu: loc('m'), sigma: loc('s') }));
  const bindings = mkBindings([['nk', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('nk'), ref('M')],   // only one arg, kernel needs 2
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.equal(r, null);
});

test('fusion (b) MVP: refuse unknown kwarg name', () => {
  const kIR = kernelofBinding(['c'], ['c'],
    callKW('Normal', { mu: loc('c'), sigma: lit(1) }));
  const bindings = mkBindings([['nk', { ir: kIR }]]);
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [ref('nk')],
    kwargs: { z: ref('Z') },   // 'z' doesn't match paramKwargs ['c']
  };
  const r = _tryDissolveKernelBroadcast(bcIR, bindings);
  assert.equal(r, null);
});
