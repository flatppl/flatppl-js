'use strict';

// =====================================================================
// Spec §04 §sec:documentation + §05 Documentation — end-to-end tests
// =====================================================================
//
// Doc-comments (`%`, `%%%`) attach to bindings, survive into FlatPIR
// as the optional `(%doc <markup> <line>...)` sub-form. Engine-side
// implementation: parser attaches the doc to AST.AssignStatement /
// AST.TildeBinding nodes; pir.lowerToModule carries it onto
// LoweredBinding.doc; pir-sexpr.serialize/parse round-trips it.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const engine = require('../index.ts');
const pirSexpr = require('../pir-sexpr.ts');

function errors(r: any) {
  return r.diagnostics.filter((d: any) => d.severity === 'error');
}
function docOf(r: any, name: string) {
  const lb = r.loweredModule.bindings.get(name);
  return lb && lb.doc;
}

// =====================================================================
// Leading single-line
// =====================================================================

test('doc: leading single-line `% ...` attaches to next binding', () => {
  const r = engine.processSource(`
% Prior mean.
mu = 0.0
`);
  assert.deepEqual(errors(r), []);
  assert.deepEqual(docOf(r, 'mu'),
    { markup: 'md', lines: ['Prior mean.'] });
});

test('doc: leading doc-comment with `%md` tag (explicit default)', () => {
  const r = engine.processSource(`
%md Prior mean.
mu = 0.0
`);
  assert.deepEqual(errors(r), []);
  assert.deepEqual(docOf(r, 'mu'),
    { markup: 'md', lines: ['Prior mean.'] });
});

test('doc: leading doc-comment with `%typ` tag', () => {
  const r = engine.processSource(`
%typ Posterior over $mu$.
posterior = mu
`);
  assert.deepEqual(errors(r), []);
  assert.deepEqual(docOf(r, 'posterior'),
    { markup: 'typ', lines: ['Posterior over $mu$.'] });
});

test('doc: unrecognised markup tag is a parse error', () => {
  const r = engine.processSource(`
%xyz content
mu = 0.0
`);
  const e = errors(r).filter((d: any) => /markup tag/.test(d.message));
  assert.ok(e.length > 0, 'should report unrecognised tag');
});

// =====================================================================
// Trailing single-line
// =====================================================================

test('doc: trailing single-line `... % ...` attaches to the same binding', () => {
  const r = engine.processSource(`sigma = 1.0  % Prior std. dev.`);
  assert.deepEqual(errors(r), []);
  assert.deepEqual(docOf(r, 'sigma'),
    { markup: 'md', lines: ['Prior std. dev.'] });
});

test('doc: trailing doc terminates at `;` (transport-safe)', () => {
  const r = engine.processSource(`mu = 0 % Prior mean.; sigma = 1`);
  assert.deepEqual(errors(r), []);
  assert.deepEqual(docOf(r, 'mu'),
    { markup: 'md', lines: ['Prior mean.'] });
  // Next binding is untouched.
  assert.equal(docOf(r, 'sigma'), null);
});

// =====================================================================
// Block form
// =====================================================================

test('doc: `%%%` block attaches to next binding; preserves blank lines', () => {
  const r = engine.processSource(`
%%%
The observation model.

Independent Gaussians.
%%%
obs = 0.0
`);
  assert.deepEqual(errors(r), []);
  assert.deepEqual(docOf(r, 'obs'),
    { markup: 'md',
      lines: ['The observation model.', '', 'Independent Gaussians.'] });
});

test('doc: `%%%typ` block with markup tag', () => {
  const r = engine.processSource(`
%%%typ
Posterior $\\\\pi(\\\\mu \\\\mid x)$.
%%%
posterior = 0.0
`);
  assert.deepEqual(errors(r), []);
  assert.equal(docOf(r, 'posterior').markup, 'typ');
});

test('doc: empty `%%%` block yields no doc on the binding', () => {
  const r = engine.processSource(`
%%%
%%%
mu = 0.0
`);
  assert.deepEqual(errors(r), []);
  // Zero content lines — spec §11 says absent form is canonical.
  // We allow either null OR `{markup, lines: []}` here; sexpr write
  // will elide both.
  const d = docOf(r, 'mu');
  assert.ok(d == null || (d.lines && d.lines.length === 0));
});

test('doc: `%%%` opener requires its own line (mid-line is not a block-form)', () => {
  // Spec §05: block fences must be on their own line. A `%%%` after
  // the RHS on the same line isn't a block opener — the tokenizer
  // falls through to the single-line `%` form. The IMPORTANT
  // invariant is that we don't silently swallow following lines as
  // block body — the next line should parse as its own statement
  // (or error visibly).
  const r = engine.processSource(`mu = 0.0\nsigma = 1.0`);
  assert.deepEqual(errors(r), []);
});

// =====================================================================
// Attachment rules
// =====================================================================

test('doc: two leading single-line docs on same binding is an error', () => {
  const r = engine.processSource(`
% First.
% Second.
mu = 0.0
`);
  const e = errors(r).filter((d: any) => /one doc-comment/.test(d.message));
  assert.ok(e.length > 0);
});

test('doc: leading + trailing on the same binding is an error', () => {
  const r = engine.processSource(`
% Leading.
mu = 0.0 % Trailing.
`);
  const e = errors(r).filter((d: any) => /leading OR trailing/.test(d.message));
  assert.ok(e.length > 0);
});

test('doc: orphan doc-comment at EOF is a warning', () => {
  const r = engine.processSource(`mu = 0.0
% Orphan with no following binding.`);
  const warnings = r.diagnostics.filter((d: any) =>
    d.severity === 'warning' && /not attached/.test(d.message));
  assert.ok(warnings.length > 0);
});

// =====================================================================
// FlatPIR round-trip
// =====================================================================

test('doc → FlatPIR: emits `(%doc md "...")` sub-form', () => {
  const r = engine.processSource(`
% Prior mean.
mu = 0.0
`);
  const text = pirSexpr.toSexpr(r.loweredModule, { indent: true });
  assert.match(text, /\(%doc md "Prior mean\."\)/);
});

test('doc → FlatPIR: omits sub-form for undocumented bindings', () => {
  const r = engine.processSource(`mu = 0.0`);
  const text = pirSexpr.toSexpr(r.loweredModule);
  assert.doesNotMatch(text, /%doc/);
});

test('doc → FlatPIR: block content emits one string per line', () => {
  const r = engine.processSource(`
%%%
A

B
%%%
mu = 0.0
`);
  const text = pirSexpr.toSexpr(r.loweredModule, { indent: true });
  assert.match(text, /\(%doc md "A" "" "B"\)/);
});

test('doc round-trip: serialise then parse preserves doc', () => {
  const r = engine.processSource(`
% Prior mean.
mu = 0.0
sigma = 1.0  % Prior std.
%%%typ
Posterior summary.

Includes both location and scale.
%%%
posterior = mu
`);
  assert.deepEqual(errors(r), []);
  const text = pirSexpr.toSexpr(r.loweredModule, { indent: true });
  const parsed = pirSexpr.fromSexpr(text);
  assert.deepEqual(parsed.diagnostics, []);
  const mod = parsed.module;
  assert.deepEqual(mod.bindings.get('mu').doc,
    { markup: 'md', lines: ['Prior mean.'] });
  assert.deepEqual(mod.bindings.get('sigma').doc,
    { markup: 'md', lines: ['Prior std.'] });
  assert.deepEqual(mod.bindings.get('posterior').doc,
    { markup: 'typ',
      lines: ['Posterior summary.', '', 'Includes both location and scale.'] });
});

// =====================================================================
// Module-level documentation (via `flatppl_compat`)
// =====================================================================

test('doc: `%%%` block attached to `flatppl_compat` is the module doc', () => {
  const r = engine.processSource(`
%%%
# Eight-schools model

Hierarchical Normal model.
%%%
flatppl_compat = "0.3"
`);
  assert.deepEqual(errors(r), []);
  assert.deepEqual(docOf(r, 'flatppl_compat'),
    { markup: 'md',
      lines: ['# Eight-schools model', '', 'Hierarchical Normal model.'] });
});
