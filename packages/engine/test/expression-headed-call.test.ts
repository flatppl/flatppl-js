'use strict';

// =====================================================================
// Expression-headed user calls (spec §11 %call, flatppl-design b070d0a;
// §05 Postfix Call*): the call head is an EXPRESSION that must evaluate
// to a user-defined callable — a `(%ref …)` in the common case, or an
// inline callable such as a reification:
//
//   (%call (functionof (%ref self e) %specinputs ((p (%ref self a)))) 2.5)
//
// Engine realisation: the surface forms `functionof(e, p = a)(2.5)`,
// `(x -> 2*x)(3.0)`, `fn(2*_)(3.0)` parse (nested CallExpr), lower to
// the callee-form IR (`callee` child node; ref heads keep the `target`
// {ns,name} pair, unchanged on the wire), beta-reduce at lift via the
// same _inlineApplication machinery named calls use, and type via
// inferCall's callee branch. pir-sexpr writes/reads the head as an
// expression. Mirrors flatppl-rust 188f486.
//
// Before this landed, an expression-headed call was SILENTLY dropped:
// lower stored a lit-null placeholder with a stashed `lowerError` and
// no diagnostic — the binding just vanished from the model.
// =====================================================================

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const pirSexpr = require('../pir-sexpr.ts');

const SAMPLE_COUNT = 64;

function makeCtx(source: any) {
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [], 'source must process cleanly');
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 0xE0CA11 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: SAMPLE_COUNT,
    rootSeed:    0xE0CA11,
  };
  return { ctx, lowered: lifted.loweredModule };
}

// Strip loc/meta only — numType round-trips since the lexical-form fix
// (spec §11: a lit's type IS its wire form; `3` integer, `3.0` real).
function stripped(ir: any): any {
  return JSON.parse(JSON.stringify(ir, (k, v) =>
    (k === 'loc' || k === 'meta' || k === 'originLoc') ? undefined : v));
}

function normWs(s: string): string {
  return s.replace(/\s+/g, ' ').trim();
}

const SRC = `
a = elementof(reals)
e = 2.0 * a
g = functionof(e, p = a)(2.5)
`;

test('lower: inline-reification application lowers to the callee-form IR', () => {
  const { lowered } = makeCtx(SRC);
  const rhs = lowered.bindings.get('g').rhs;
  assert.equal(rhs.kind, 'call');
  assert.equal(rhs.target, undefined, 'no ref head — expression head');
  assert.equal(rhs.callee.op, 'functionof', 'callee is the inline reification');
  assert.deepEqual(rhs.callee.paramKwargs, ['p']);
  assert.equal(rhs.args.length, 1);
  assert.equal(rhs.args[0].value, 2.5);
});

test('typeinfer: the applied call carries the callable result type', () => {
  const { lowered } = makeCtx(SRC);
  assert.deepEqual(lowered.bindings.get('g').inferredType,
    { kind: 'scalar', prim: 'real' },
    'g = functionof(e, p = a)(2.5) types as the function result (real)');
});

test('e2e: functionof(e, p = a)(2.5) beta-reduces through the derived binding (= 5.0)', async () => {
  // e = 2*a reaches the boundary a through a derived binding; the lift
  // application substitutes a → 2.5 through the closure walk, so g is
  // exactly 2·2.5 — boundary substitution, not the module-graph `a`.
  const { ctx } = makeCtx(SRC);
  const g = await ctx.getMeasure('g');
  assert.equal(g.samples[0], 5.0, 'g = 2 * 2.5');
});

test('e2e: inline lambda and fn-hole applications evaluate', async () => {
  const { ctx } = makeCtx(`
y1 = (x -> 2.0 * x)(3.0)
y2 = fn(_ + 10.0)(3.0)
`);
  const [y1, y2] = await Promise.all([ctx.getMeasure('y1'), ctx.getMeasure('y2')]);
  assert.equal(y1.samples[0], 6.0, '(x -> 2x)(3) = 6');
  assert.equal(y2.samples[0], 13.0, 'fn(_+10)(3) = 13');
});

test('pir-sexpr writer: emits the spec §11 expression-head example shape exactly', () => {
  const { lowered } = makeCtx(SRC);
  const out = normWs(pirSexpr.toSexpr(lowered));
  assert.ok(out.includes(
    '(%bind g (%call (functionof (%ref self e) %specinputs ((p (%ref self a)))) 2.5))'),
    'must match the spec §11 example shape; got: ' + out);
});

test('pir-sexpr reader: the spec §11 expression-head line round-trips to the callee-form', () => {
  const text = `
(%module
  (%bind a (elementof reals))
  (%bind e (mul 2.0 (%ref self a)))
  (%bind g (%call (functionof (%ref self e) %specinputs ((p (%ref self a)))) 2.5)))
`;
  const { module: mod, diagnostics } = pirSexpr.fromSexpr(text);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const read = mod.bindings.get('g').rhs;
  assert.equal(read.callee.op, 'functionof');
  assert.equal(read.target, undefined);
  // Cross-check vs the lowered surface form (the §11 conformance pin).
  const { lowered } = makeCtx(SRC);
  assert.deepEqual(stripped(read), stripped(lowered.bindings.get('g').rhs),
    'spec .flatpir text and lowered surface FlatPPL must agree');
});

test('pir-sexpr: ref-headed %call is unchanged on the wire', () => {
  const { lowered } = makeCtx(`
f = x -> 2.0 * x
y = f(3.0)
`);
  const out = normWs(pirSexpr.toSexpr(lowered));
  assert.ok(out.includes('(%bind y (%call (%ref self f) 3.0))'),
    'ref heads keep the plain (%ref …) head form; got: ' + out);
});

test('chained call f(x)(y): parses + lowers liberally, types as the §11 condition failure', () => {
  // f(1.0) yields a SCALAR, not a callable — the "must evaluate to a
  // user-defined callable" typing condition fails, loudly typed (no
  // silent lit-null drop).
  const lifted = processSource(`
f = x -> 2.0 * x
z = f(1.0)(2.0)
`);
  const rhs = lifted.loweredModule.bindings.get('z').rhs;
  assert.equal(rhs.callee && rhs.callee.target.name, 'f', 'callee-form over the inner call');
  const t = lifted.loweredModule.bindings.get('z').inferredType;
  assert.equal(t.kind, 'failed', 'non-callable head → failed type, not silence');
});
