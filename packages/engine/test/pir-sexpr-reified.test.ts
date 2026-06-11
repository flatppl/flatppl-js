'use strict';

// Reified-callable FlatPIR form (spec §11 "Reified callables",
// flatppl-design 2a9e3e9): fixed arity after the head —
//
//   (functionof <output> %specinputs ((<name> <ref>) ...))
//   (functionof <output> %autoinputs %deferred)
//   (functionof <output> %autoinputs ((<name> <ref>) ...))   ; metadata
//
// The old `(%params <node-names>)` wrapper is gone (it was lossy: a
// placeholder entry `x = _x_` declares call-name `x`, but %params kept
// only `_x_`). Body refs to identifier-form boundary inputs are plain
// `self` refs on the wire; only placeholders are `%local`. The
// engine-internal IR is spec-shaped too (lower.ts: `%local` is
// placeholders-only), so writer and reader carry bodies VERBATIM — no
// namespace translation at the serialization boundary.
//
// The concrete syntax is pinned against the spec §11 worked example
// (helpers.flatpir) and flatppl-rust's reified.flatpir round-trip
// fixture, so the two engines cannot drift apart silently.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');
const pirSexpr = require('../pir-sexpr.ts');

function buildModule(src: any) {
  return processSource(src).loweredModule;
}

// Deep-compare IR shapes ignoring location/meta annotations (the reader
// produces no locs; lowered IR carries them).
function stripped(ir: any): any {
  return JSON.parse(JSON.stringify(ir, (k, v) =>
    (k === 'loc' || k === 'meta' || k === 'originLoc') ? undefined : v));
}

function normWs(s: string): string {
  return s.replace(/\s+/g, ' ').trim();
}

// ---------------------------------------------------------------------
// %specinputs — the spec §11 helpers.flatpir worked example
// ---------------------------------------------------------------------

const HELPERS_SURFACE = `
center = elementof(reals)
spread = elementof(posreals)
obs_kernel = functionof(
    Normal(mu = center + _x_, sigma = spread),
    center = center, spread = spread, x = _x_)
shifted_value = center + 1.0
`;

// The spec's own emission for obs_kernel (11-flatpir.md, bare FlatPIR
// example; whitespace-normalised). Mirrored in
// flatppl-rust/crates/flatpir/tests/fixtures/helpers.flatpir.
const HELPERS_OBS_KERNEL_SEXPR = normWs(`
(functionof
  (Normal
    (%kwarg mu (add (%ref self center) (%ref %local _x_)))
    (%kwarg sigma (%ref self spread)))
  %specinputs
  ((center (%ref self center))
   (spread (%ref self spread))
   (x (%ref %local _x_))))
`);

test('writer: %specinputs emission matches the spec §11 worked example exactly', () => {
  const mod = buildModule(HELPERS_SURFACE);
  const out = normWs(pirSexpr.toSexpr(mod));
  assert.ok(out.includes('(%bind obs_kernel ' + HELPERS_OBS_KERNEL_SEXPR + ')'),
    'obs_kernel emission must match the spec example.\nGot:      ' + out
    + '\nExpected: …' + HELPERS_OBS_KERNEL_SEXPR + '…');
  // The lossy %params form must be gone.
  assert.ok(!out.includes('%params'), 'old %params form must not be emitted');
});

test('writer: identifier-form boundary refs are self on the wire AND internally; placeholders stay %local', () => {
  const mod = buildModule(HELPERS_SURFACE);
  const out = normWs(pirSexpr.toSexpr(mod));
  // Body: center/spread are identifier-form boundaries → plain self refs
  // on the wire, `_x_` the placeholder → %local.
  assert.ok(out.includes('(%kwarg mu (add (%ref self center) (%ref %local _x_)))'),
    'body must carry self refs for identifier boundaries, %local for placeholders');
  // The engine-internal IR is spec-shaped (lower.ts narrowing): the
  // writer emits the body verbatim, no un-alias pass.
  const rhs = mod.bindings.get('obs_kernel').rhs;
  assert.equal(rhs.body.kwargs.mu.args[0].ns, 'self',
    'engine-internal body ref must be self (spec-shaped, %local is placeholders-only)');
});

test('reader: %specinputs restores params/paramKwargs/paramSources; body verbatim', () => {
  const mod = buildModule(HELPERS_SURFACE);
  const text = pirSexpr.toSexpr(mod);
  const { module: mod2, diagnostics } = pirSexpr.fromSexpr(text);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const orig = mod.bindings.get('obs_kernel').rhs;
  const back = mod2.bindings.get('obs_kernel').rhs;
  assert.equal(back.op, 'functionof');
  assert.deepEqual(back.params, orig.params);             // node names
  assert.deepEqual(back.paramKwargs, orig.paramKwargs);   // call-names (was lossy before)
  assert.deepEqual(back.paramSources, orig.paramSources); // origins
  // Body carried verbatim: bit-identical to the lowered internal shape
  // (both spec-shaped — no namespace translation either way).
  assert.deepEqual(stripped(back.body), stripped(orig.body),
    'round-tripped body must equal the engine-internal shape');
});

test('reader: the spec helpers.flatpir text itself reconstructs the lowered surface form', () => {
  // Cross-engine conformance pin: reading the spec's own .flatpir text
  // must yield the same IR as lowering the surface FlatPPL. Mirrors
  // flatppl-rust/crates/flatpir/tests/fixtures/helpers.flatpir.
  const text = `
; Bare FlatPIR — spec §11 helpers.flatpir example.
(%module
  (%public center spread obs_kernel shifted_value)
  (%bind center (elementof reals))
  (%bind spread (elementof posreals))
  (%bind obs_kernel
    (functionof
      (Normal
        (%kwarg mu (add (%ref self center) (%ref %local _x_)))
        (%kwarg sigma (%ref self spread)))
      %specinputs
      ((center (%ref self center))
       (spread (%ref self spread))
       (x (%ref %local _x_)))))
  (%bind shifted_value (add (%ref self center) 1.0)))
`;
  const { module: mod2, diagnostics } = pirSexpr.fromSexpr(text);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const surface = buildModule(HELPERS_SURFACE).bindings.get('obs_kernel').rhs;
  const read = mod2.bindings.get('obs_kernel').rhs;
  assert.deepEqual(stripped(read), stripped(surface),
    'spec .flatpir text and lowered surface FlatPPL must agree');
});

// ---------------------------------------------------------------------
// %autoinputs — boundary-less reifications (rust reified.flatpir)
// ---------------------------------------------------------------------

test('writer: boundary-less functionof emits %autoinputs %deferred', () => {
  const mod = buildModule(`
a = elementof(reals)
b = a ^ 0.5
f = functionof(b)
`);
  const out = normWs(pirSexpr.toSexpr(mod));
  // Matches flatppl-rust reified.flatpir's `f` binding.
  assert.ok(out.includes('(%bind f (functionof (%ref self b) %autoinputs %deferred))'),
    'boundary-less reification must carry %autoinputs %deferred; got: ' + out);
});

test('reader: %autoinputs %deferred restores a boundary-less functionof', () => {
  const text = `
(%module
  (%bind f (functionof (%ref self b) %autoinputs %deferred)))
`;
  const { module: mod, diagnostics } = pirSexpr.fromSexpr(text);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const f = mod.bindings.get('f').rhs;
  assert.equal(f.op, 'functionof');
  assert.equal(f.params, undefined, 'no authored boundary → no params');
  assert.deepEqual(stripped(f.body), { kind: 'ref', ns: 'self', name: 'b' });
});

test('reader: a FILLED %autoinputs list is inference metadata — parsed and dropped', () => {
  // Spec §11: a filled %autoinputs list is strippable metadata (like
  // %meta); the engine re-infers. It must NOT become an authored
  // boundary (params/paramKwargs).
  const text = `
(%module
  (%bind g (functionof (%ref self b) %autoinputs ((a (%ref self a))))))
`;
  const { module: mod, diagnostics } = pirSexpr.fromSexpr(text);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const g = mod.bindings.get('g').rhs;
  assert.equal(g.op, 'functionof');
  assert.equal(g.params, undefined, 'filled %autoinputs must not restore params');
  assert.deepEqual(stripped(g.body), { kind: 'ref', ns: 'self', name: 'b' });
});

// ---------------------------------------------------------------------
// kernelof — desugared at the reader (the same lowering rule lower.ts
// applies at the surface ingest point)
// ---------------------------------------------------------------------

test('reader: kernelof desugars to functionof(lawof(output)) on ingest', () => {
  // Matches flatppl-rust reified.flatpir's `kernel` binding. JS never
  // emits kernelof (lower.ts canonicalises at the surface); rust-emitted
  // FlatPIR contains it, so the reader applies the same §04 rule —
  // post-read modules carry one uniform reification form.
  const text = `
(%module
  (%bind kernel
    (kernelof (draw (Normal (%kwarg mu (%ref self a)) (%kwarg sigma 1.0)))
      %autoinputs %deferred)))
`;
  const { module: mod, diagnostics } = pirSexpr.fromSexpr(text);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const k = mod.bindings.get('kernel').rhs;
  assert.equal(k.op, 'functionof', 'kernelof must canonicalise to functionof');
  assert.equal(k.body.op, 'lawof', 'kernelof output must be lawof-wrapped');
  assert.equal(k.body.args[0].op, 'draw');
});

test('reader: kernelof with %specinputs restores boundaries + lawof-wraps', () => {
  const text = `
(%module
  (%bind k
    (kernelof (Normal (%kwarg mu (%ref %local theta)) (%kwarg sigma 1.0))
      %specinputs ((theta (%ref self theta)))))
`;
  // Note: kernelof's spec body for an identifier boundary would carry
  // (%ref self theta); use the equivalent surface form for the oracle.
  const { module: mod, diagnostics } = pirSexpr.fromSexpr(text + ')');
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const k = mod.bindings.get('k').rhs;
  assert.equal(k.op, 'functionof');
  assert.equal(k.body.op, 'lawof');
  assert.deepEqual(k.paramKwargs, ['theta']);
  assert.deepEqual(k.paramSources, [{ kind: 'binding', name: 'theta' }]);
});

// ---------------------------------------------------------------------
// Nested reifications — spec-shaped bodies round-trip verbatim
// ---------------------------------------------------------------------

test('round-trip: nested reification with a re-declared boundary name round-trips verbatim', () => {
  // Hand-built SPEC-SHAPED IR (lower.ts narrowing: identifier-bound
  // boundary refs are plain `self` refs; only placeholders are %local):
  // an outer functionof with identifier boundary `a` whose body contains
  // a NESTED functionof that re-declares `a` as its own boundary. Both
  // scopes' bodies carry `self a`; the entry lists designate which scope
  // cuts where (the shadow logic lives in the scope-aware consumers —
  // walkIRScoped/mapIRScoped — not in serialization).
  const pir = require('../pir.ts');
  const inner = {
    kind: 'call', op: 'functionof',
    params: ['a', '_y_'], paramKwargs: ['a', 'y'],
    paramSources: [
      { kind: 'binding', name: 'a' },
      { kind: 'placeholder', name: '_y_' },
    ],
    body: { kind: 'call', op: 'mul', args: [
      { kind: 'ref', ns: 'self', name: 'a' },
      { kind: 'ref', ns: '%local', name: '_y_' },
    ] },
  };
  const outer = {
    kind: 'call', op: 'functionof',
    params: ['a'], paramKwargs: ['p'],
    paramSources: [{ kind: 'binding', name: 'a' }],
    body: { kind: 'call', op: 'add', args: [
      { kind: 'ref', ns: 'self', name: 'a' },   // outer scope's a
      inner,
    ] },
  };
  const mod = pir.loweredModule();
  mod.bindings.set('h', pir.loweredBinding('h', outer, {}));
  const text = pirSexpr.toSexpr(mod);
  // Both scopes' identifier boundaries emit as self refs in their own
  // bodies; the inner entry list names the inner scope.
  const flat = normWs(text);
  assert.ok(flat.includes('%specinputs ((p (%ref self a)))'), 'outer entry list');
  assert.ok(flat.includes('%specinputs ((a (%ref self a)) (y (%ref %local _y_)))'),
    'inner entry list');
  const { module: mod2, diagnostics } = pirSexpr.fromSexpr(text);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.deepEqual(stripped(mod2.bindings.get('h').rhs), stripped(outer),
    'nested reification must round-trip bit-identically');
});

// ---------------------------------------------------------------------
// End-to-end: a reified module round-trips and the call-names survive
// (the old form's lossiness, pinned shut)
// ---------------------------------------------------------------------

test('inferred kernel %inputs carry CALL-names on a PIR-read module (spec §11)', () => {
  // Spec §11: `%inputs` in inferred kernel/function/likelihood types
  // carries call-names (`center spread x`), not node names (`… _x_`).
  // inferReification already reads paramKwargs; the old reader dropped
  // them (only `%params` node names survived), so read modules fell
  // back to node names. The new entry list restores call-names.
  const text = `
(%module
  (%public center spread obs_kernel)
  (%bind center (elementof reals))
  (%bind spread (elementof posreals))
  (%bind obs_kernel
    (functionof
      (Normal
        (%kwarg mu (add (%ref self center) (%ref %local _x_)))
        (%kwarg sigma (%ref self spread)))
      %specinputs
      ((center (%ref self center))
       (spread (%ref self spread))
       (x (%ref %local _x_))))))
`;
  const { module: mod, diagnostics } = pirSexpr.fromSexpr(text);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const typeinfer = require('../typeinfer.ts');
  typeinfer.inferTypes(mod);
  const t = mod.bindings.get('obs_kernel').inferredType;
  assert.ok(t && t.kind === 'kernel', 'reified measure body infers a kernel type');
  assert.deepEqual(t.inputs.map((i: any) => i.name), ['center', 'spread', 'x'],
    '%inputs must carry call-names, not node names (no _x_)');
});

test('round-trip: lambda call-names survive (the %params lossiness is closed)', () => {
  const mod = buildModule(`
m = Normal(0, 1)
g = x -> exp(0 - x ^ 2)
w = weighted(g, m)
`);
  const text = pirSexpr.toSexpr(mod);
  const { module: mod2, diagnostics } = pirSexpr.fromSexpr(text);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const g = mod2.bindings.get('g').rhs;
  // The old (%params _x_) form dropped the call-name `x`.
  assert.deepEqual(g.paramKwargs, ['x'], 'lambda call-name must survive the round-trip');
  assert.deepEqual(g.params, ['_x_']);
  assert.deepEqual(g.paramSources, [{ kind: 'placeholder', name: '_x_' }]);
});
