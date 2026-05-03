'use strict';

// Tests for engine/lower.js — AST → FlatPIR-JSON lowering.
//
// Strategy: parse small FlatPPL snippets through the existing tokenizer
// + parser, lower the resulting AST, and assert on the IR shape. This
// gives us realistic AST inputs (no hand-crafted nodes that drift from
// the parser's actual output) while focusing the tests on lowering
// behavior.
//
// Coverage organized by IR shape category:
//   - Literals & built-in symbols (lit, const)
//   - References (ref vs const vs lit disambiguation)
//   - Operators desugared to function calls
//   - Indexing and field access
//   - Composite literals (vector, tuple)
//   - Built-in calls (Normal, draw, iid, …)
//   - User-defined calls (target.ns/name shape)
//   - Field-form calls (record, joint, jointchain)
//   - Reification: functionof, kernelof, fn
//   - Local-scope handling (placeholders, identifiers inside reified bodies)
//   - Module loads (load_module assigns)
//   - Source-position fidelity (loc preserved on every IR node)

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { tokenize } = require('../tokenizer');
const { parse } = require('../parser');
const { lowerExpr, lowerBinding } = require('../lower');

// Parse a single binding `<name> = <expr>` and return the lowered RHS.
function lowerOne(source) {
  const { tokens } = tokenize(source);
  const { ast, diagnostics } = parse(tokens);
  const errors = diagnostics.filter(d => d.severity === 'error');
  assert.equal(errors.length, 0, `parse errors: ${JSON.stringify(errors)}`);
  // First AssignStatement; lowerBinding handles the unwrap.
  const stmt = ast.body.find(s => s.type === 'AssignStatement');
  assert.ok(stmt, 'no AssignStatement in source');
  return lowerBinding(stmt);
}

// =====================================================================
// Literals & constants
// =====================================================================

test('lower: integer literal', () => {
  const ir = lowerOne('x = 42');
  assert.equal(ir.kind, 'lit');
  assert.equal(ir.value, 42);
  assert.ok(ir.loc);
});

test('lower: float literal', () => {
  const ir = lowerOne('x = 3.14');
  assert.equal(ir.kind, 'lit');
  assert.equal(ir.value, 3.14);
});

test('lower: string literal', () => {
  const ir = lowerOne('x = "hello"');
  assert.equal(ir.kind, 'lit');
  assert.equal(ir.value, 'hello');
});

test('lower: bool literal', () => {
  // FlatPPL parses true/false as identifiers; lowering decides they're bools.
  const irT = lowerOne('x = true');
  assert.equal(irT.kind, 'lit');
  assert.equal(irT.value, true);
  const irF = lowerOne('x = false');
  assert.equal(irF.kind, 'lit');
  assert.equal(irF.value, false);
});

test('lower: built-in constants (pi, inf)', () => {
  // ConstantRef nodes in the AST.
  const irPi = lowerOne('x = pi');
  assert.equal(irPi.kind, 'const');
  assert.equal(irPi.name, 'pi');
});

test('lower: built-in sets (reals)', () => {
  // SetRef nodes in the AST.
  const ir = lowerOne('x = elementof(reals)');
  assert.equal(ir.kind, 'call');
  assert.equal(ir.op, 'elementof');
  assert.equal(ir.args.length, 1);
  assert.equal(ir.args[0].kind, 'const');
  assert.equal(ir.args[0].name, 'reals');
});

// =====================================================================
// References
// =====================================================================

test('lower: bare identifier defaults to (ref self <name>)', () => {
  const ir = lowerOne('x = some_var');
  assert.deepEqual({ kind: ir.kind, ns: ir.ns, name: ir.name }, {
    kind: 'ref',
    ns: 'self',
    name: 'some_var',
  });
});

test('lower: identifier resolution prefers %local over self when in scope', () => {
  // Inside a kernelof body, `_par_` is a placeholder param.
  const ir = lowerOne('f = functionof(_par_ * 2, par = _par_)');
  assert.equal(ir.kind, 'call');
  assert.equal(ir.op, 'functionof');
  // Body is the multiply, with first arg being the placeholder ref.
  assert.equal(ir.body.kind, 'call');
  assert.equal(ir.body.op, 'mul');
  assert.equal(ir.body.args[0].kind, 'ref');
  assert.equal(ir.body.args[0].ns, '%local');
  assert.equal(ir.body.args[0].name, '_par_');
});

// =====================================================================
// Operator desugaring
// =====================================================================

test('lower: a + b → (add a b)', () => {
  const ir = lowerOne('x = a + b');
  assert.equal(ir.kind, 'call');
  assert.equal(ir.op, 'add');
  assert.equal(ir.args.length, 2);
  assert.equal(ir.args[0].name, 'a');
  assert.equal(ir.args[1].name, 'b');
});

test('lower: full set of binary operators desugar', () => {
  const cases = [
    { src: 'a - b',  op: 'sub' },
    { src: 'a * b',  op: 'mul' },
    { src: 'a / b',  op: 'div' },
    { src: 'a < b',  op: 'lt' },
    { src: 'a <= b', op: 'le' },
    { src: 'a > b',  op: 'gt' },
    { src: 'a >= b', op: 'ge' },
    { src: 'a == b', op: 'eq' },
    { src: 'a != b', op: 'ne' },
  ];
  for (const { src, op } of cases) {
    const ir = lowerOne(`x = ${src}`);
    assert.equal(ir.kind, 'call', `for ${src}`);
    assert.equal(ir.op, op, `for ${src}: expected op '${op}', got '${ir.op}'`);
  }
});

test('lower: unary minus → (neg x)', () => {
  const ir = lowerOne('x = -y');
  assert.equal(ir.kind, 'call');
  assert.equal(ir.op, 'neg');
  assert.equal(ir.args.length, 1);
  assert.equal(ir.args[0].name, 'y');
});

// =====================================================================
// Indexing and field access
// =====================================================================

test('lower: a[i, j] → (get a i j)', () => {
  const ir = lowerOne('x = a[i, j]');
  assert.equal(ir.kind, 'call');
  assert.equal(ir.op, 'get');
  assert.equal(ir.args.length, 3);
  assert.equal(ir.args[0].name, 'a');
  assert.equal(ir.args[1].name, 'i');
  assert.equal(ir.args[2].name, 'j');
});

test('lower: a[:, j] → (get a all j)', () => {
  const ir = lowerOne('x = a[:, j]');
  assert.equal(ir.op, 'get');
  // Second arg is the SliceAll → const "all"
  assert.equal(ir.args[1].kind, 'const');
  assert.equal(ir.args[1].name, 'all');
});

test('lower: a.field → (get_field a "field")', () => {
  const ir = lowerOne('x = a.field');
  assert.equal(ir.kind, 'call');
  assert.equal(ir.op, 'get_field');
  assert.equal(ir.args.length, 2);
  assert.equal(ir.args[0].name, 'a');
  assert.equal(ir.args[1].kind, 'lit');
  assert.equal(ir.args[1].value, 'field');
});

// =====================================================================
// Composite literals
// =====================================================================

test('lower: array literal [1, 2, 3] → (vector 1 2 3)', () => {
  const ir = lowerOne('x = [1, 2, 3]');
  assert.equal(ir.kind, 'call');
  assert.equal(ir.op, 'vector');
  assert.equal(ir.args.length, 3);
  assert.deepEqual(ir.args.map(a => a.value), [1, 2, 3]);
});

test('lower: nested array', () => {
  const ir = lowerOne('x = [[1, 2], [3, 4]]');
  assert.equal(ir.op, 'vector');
  assert.equal(ir.args.length, 2);
  assert.equal(ir.args[0].op, 'vector');
  assert.deepEqual(ir.args[0].args.map(a => a.value), [1, 2]);
});

// =====================================================================
// Built-in calls (with kwargs)
// =====================================================================

test('lower: Normal(mu = 0, sigma = 1) — kwargs as object', () => {
  const ir = lowerOne('x = Normal(mu = 0, sigma = 1)');
  assert.equal(ir.kind, 'call');
  assert.equal(ir.op, 'Normal');
  assert.ok(!ir.args, 'no positional args');
  assert.ok(ir.kwargs);
  assert.equal(ir.kwargs.mu.value, 0);
  assert.equal(ir.kwargs.sigma.value, 1);
});

test('lower: draw(M) — single positional arg', () => {
  const ir = lowerOne('x = draw(Normal(mu = 0, sigma = 1))');
  assert.equal(ir.op, 'draw');
  assert.equal(ir.args.length, 1);
  assert.equal(ir.args[0].op, 'Normal');
});

test('lower: iid(M, n) — purely positional builtin', () => {
  const ir = lowerOne('x = iid(Normal(mu = 0, sigma = 1), 100)');
  assert.equal(ir.op, 'iid');
  assert.equal(ir.args.length, 2);
  assert.equal(ir.args[1].kind, 'lit');
  assert.equal(ir.args[1].value, 100);
});

// =====================================================================
// User-defined calls
// =====================================================================

test('lower: user-defined call uses `target` not `op`', () => {
  const ir = lowerOne('x = my_helper_fn(1, 2)');
  assert.equal(ir.kind, 'call');
  assert.ok(!ir.op,    'no op for user-defined call');
  assert.deepEqual(ir.target, { ns: 'self', name: 'my_helper_fn' });
  assert.equal(ir.args.length, 2);
});

// =====================================================================
// Field-form calls (record, joint, jointchain)
// =====================================================================

test('lower: record(mu = 0, sigma = 1) — fields preserve order', () => {
  const ir = lowerOne('x = record(mu = 0, sigma = 1)');
  assert.equal(ir.kind, 'call');
  assert.equal(ir.op, 'record');
  assert.ok(!ir.args, 'no positional args');
  assert.ok(Array.isArray(ir.fields));
  assert.equal(ir.fields.length, 2);
  assert.equal(ir.fields[0].name, 'mu');
  assert.equal(ir.fields[1].name, 'sigma');
});

test('lower: joint(M1, M2) — purely positional', () => {
  const ir = lowerOne('x = joint(M1, M2)');
  assert.equal(ir.op, 'joint');
  assert.equal(ir.args.length, 2);
  assert.ok(!ir.fields, 'no fields when only positional');
});

test('lower: joint(a = M, b = N) — keyword form, ordered fields', () => {
  const ir = lowerOne('x = joint(a = M, b = N)');
  assert.equal(ir.op, 'joint');
  assert.ok(!ir.args);
  assert.equal(ir.fields.length, 2);
  assert.equal(ir.fields[0].name, 'a');
  assert.equal(ir.fields[1].name, 'b');
});

test('lower: jointchain keyword form preserves order', () => {
  const ir = lowerOne('x = jointchain(a = M1, b = K1, c = K2)');
  assert.equal(ir.op, 'jointchain');
  assert.equal(ir.fields.map(f => f.name).join(','), 'a,b,c');
});

// =====================================================================
// Reification: functionof, kernelof
// =====================================================================

test('lower: functionof with simple identifier params', () => {
  const ir = lowerOne('f = functionof(a + b, a = a, b = b)');
  assert.equal(ir.op, 'functionof');
  assert.deepEqual(ir.params, ['a', 'b']);
  assert.deepEqual(ir.paramKwargs, ['a', 'b']);
  // Body is `add(a, b)` with both as %local refs.
  assert.equal(ir.body.op, 'add');
  assert.equal(ir.body.args[0].ns, '%local');
  assert.equal(ir.body.args[0].name, 'a');
  assert.equal(ir.body.args[1].ns, '%local');
  assert.equal(ir.body.args[1].name, 'b');
});

test('lower: functionof with placeholder params', () => {
  const ir = lowerOne('f = functionof(c * _par_, par = _par_)');
  assert.equal(ir.op, 'functionof');
  assert.deepEqual(ir.params, ['_par_']);
  assert.deepEqual(ir.paramKwargs, ['par']);
  // Body is `c * _par_`: outer `c` is self-ref; `_par_` is %local.
  assert.equal(ir.body.op, 'mul');
  assert.equal(ir.body.args[0].ns, 'self');
  assert.equal(ir.body.args[0].name, 'c');
  assert.equal(ir.body.args[1].ns, '%local');
  assert.equal(ir.body.args[1].name, '_par_');
});

test('lower: kernelof with mixed identifier and placeholder params', () => {
  const ir = lowerOne(`
    k = kernelof(Normal(mu = theta1, sigma = _spread_),
                 theta1 = theta1, spread = _spread_)
  `);
  assert.equal(ir.op, 'kernelof');
  assert.deepEqual(ir.params, ['theta1', '_spread_']);
  assert.deepEqual(ir.paramKwargs, ['theta1', 'spread']);
});

test('lower: nested reification — inner scope shadows outer', () => {
  // Inner `kernelof` introduces its own param `x`; references to `x` in the
  // inner body should resolve to that, not to any outer-scope `x`.
  const ir = lowerOne(`
    f = functionof(
      kernelof(Normal(mu = _x_, sigma = 1), x = _x_),
      outer_param = outer_param)
  `);
  assert.equal(ir.op, 'functionof');
  assert.deepEqual(ir.params, ['outer_param']);
  // Body is the kernelof
  const inner = ir.body;
  assert.equal(inner.op, 'kernelof');
  assert.deepEqual(inner.params, ['_x_']);
  // Inner Normal's mu refs %local _x_.
  assert.equal(inner.body.kwargs.mu.ns, '%local');
  assert.equal(inner.body.kwargs.mu.name, '_x_');
});

test('lower: fn(_) lambda body is the expression with bare holes', () => {
  const ir = lowerOne('g = fn(_ + _)');
  assert.equal(ir.op, 'fn');
  assert.ok(ir.body, 'fn has body');
  // Body is `add(_, _)` where each arg is a hole.
  assert.equal(ir.body.op, 'add');
  assert.equal(ir.body.args[0].kind, 'hole');
  assert.equal(ir.body.args[1].kind, 'hole');
});

// =====================================================================
// Module loads
// =====================================================================

test('lower: load_module with substitutions → assigns', () => {
  const ir = lowerOne('helpers = load_module("helpers.flatppl", center = a)');
  assert.equal(ir.op, 'load_module');
  assert.equal(ir.args.length, 1);
  assert.equal(ir.args[0].kind, 'lit');
  assert.equal(ir.args[0].value, 'helpers.flatppl');
  assert.ok(ir.assigns);
  assert.equal(ir.assigns.center.kind, 'ref');
  assert.equal(ir.assigns.center.name, 'a');
});

test('lower: standard_module (positional only)', () => {
  const ir = lowerOne('hep = standard_module("particle-physics", "0.1")');
  assert.equal(ir.op, 'standard_module');
  assert.equal(ir.args.length, 2);
  assert.equal(ir.args[0].value, 'particle-physics');
  assert.equal(ir.args[1].value, '0.1');
  assert.ok(!ir.assigns, 'no substitutions');
});

// =====================================================================
// Source-position fidelity
// =====================================================================

test('lower: every IR node carries a loc field', () => {
  const ir = lowerOne(`
    x = Normal(mu = a + 1, sigma = [1, 2, 3])
  `);
  function check(node) {
    if (!node || typeof node !== 'object') return;
    if (node.kind) {
      assert.ok(node.loc, `node ${node.kind} missing loc`);
      assert.ok(node.loc.start, `node ${node.kind} loc missing start`);
    }
    for (const key of Object.keys(node)) {
      const v = node[key];
      if (Array.isArray(v)) v.forEach(check);
      else if (v && typeof v === 'object' && v.kind) check(v);
    }
  }
  check(ir);
});

// =====================================================================
// End-to-end: lower a realistic FlatPPL fragment
// =====================================================================

test('lower: realistic Bayesian model', () => {
  const ir = lowerOne(`
    obs = draw(iid(Normal(mu = theta1, sigma = theta2), 10))
  `);
  assert.equal(ir.op, 'draw');
  assert.equal(ir.args[0].op, 'iid');
  assert.equal(ir.args[0].args[0].op, 'Normal');
  assert.equal(ir.args[0].args[0].kwargs.mu.ns, 'self');
  assert.equal(ir.args[0].args[0].kwargs.mu.name, 'theta1');
  assert.equal(ir.args[0].args[1].value, 10);
});

test('lower: lawof(record(...)) joint measure', () => {
  const ir = lowerOne(`
    joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
  `);
  assert.equal(ir.op, 'lawof');
  assert.equal(ir.args.length, 1);
  // The record is a fields-form call.
  const rec = ir.args[0];
  assert.equal(rec.op, 'record');
  assert.equal(rec.fields.length, 3);
  assert.deepEqual(rec.fields.map(f => f.name), ['theta1', 'theta2', 'obs']);
});

test('lower: pure determinism — same source twice yields equal IR', () => {
  const a = lowerOne('x = Normal(mu = a + 1, sigma = b)');
  const b = lowerOne('x = Normal(mu = a + 1, sigma = b)');
  // loc fields will have same content but might differ object-identity-wise.
  // Use deepEqual which compares by value.
  assert.deepEqual(a, b);
});
