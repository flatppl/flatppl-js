'use strict';

// =====================================================================
// Inline composite-body steps and components
// =====================================================================
//
// A boundary input of `functionof` / `kernelof` is spelled either as a
// ref into the output's ancestor subgraph or as a `_x_` placeholder —
// spec §11 "Reified callables": "`<ref>` refers to a node in the
// ancestor subgraph of `<output>` … or a placeholder within `<output>`
// (`(%ref %local _x_)`)". §04 "Placeholders and holes" then scopes a
// placeholder: "The scope of a placeholder is the nearest enclosing
// `functionof` or `kernelof`." A measure call whose kwargs name such a
// placeholder therefore cannot be hoisted to a module-level anon
// binding, and reaches the composite-body recognisers INLINE.
//
// These tests pin the recognisers' acceptance of the inline spelling
// and the rejections that survive it.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const kbShape = require('../kernel-broadcast-shape.ts');

// A minimal `bindings`-like map: `get`/`has` over a plain object.
function bindingsOf(obj: Record<string, any>) {
  return {
    has: (k: string) => Object.prototype.hasOwnProperty.call(obj, k),
    get: (k: string) => obj[k],
  };
}

const local = (name: string) => ({ kind: 'ref', ns: '%local', name });
const selfRef = (name: string) => ({ kind: 'ref', ns: 'self', name });
const normal = (mu: any) => ({
  kind: 'call', op: 'Normal', kwargs: { mu, sigma: { kind: 'lit', value: 1 } },
});
// `functionof(Normal(mu = _a_, sigma = 1), a = _a_)` — a single-input step.
const stepKernel = {
  ir: {
    kind: 'call', op: 'functionof', params: ['_a_'], paramKwargs: ['a'],
    body: normal(local('_a_')),
  },
};

function chainKernel(args: any[]) {
  return {
    ir: {
      kind: 'call', op: 'functionof', params: ['_m_'], paramKwargs: ['m'],
      body: { kind: 'call', op: 'jointchain', args },
    },
  };
}

test('jointchain recogniser accepts an inline placeholder-scoped base step', () => {
  const bindings = bindingsOf({
    chain: chainKernel([normal(local('_m_')), selfRef('step')]),
    step: stepKernel,
  });
  const desc = kbShape.detectJointChainKernelBinding('chain', bindings);
  assert.ok(desc, 'inline base DistCall matches (no anon binding to deref)');
  assert.equal(desc.steps.length, 2);
  assert.equal(desc.steps[0].base.distOp, 'Normal');
  assert.deepEqual(desc.steps[0].base.distKwargs.mu, local('_m_'),
    'base kwargs keep the placeholder for per-cell substitution');
  assert.equal(desc.steps[1].kernel.inputParam, '_a_');
});

test('jointchain recogniser still derefs an anon-bound base step', () => {
  const bindings = bindingsOf({
    chain: chainKernel([selfRef('__anon0'), selfRef('step')]),
    __anon0: { ir: normal(selfRef('m')) },
    step: stepKernel,
  });
  const desc = kbShape.detectJointChainKernelBinding('chain', bindings);
  assert.ok(desc, 'the hoisted spelling keeps matching');
  assert.equal(desc.steps[0].base.distOp, 'Normal');
});

test('jointchain recogniser rejects a base step that is neither a ref nor a call', () => {
  const bindings = bindingsOf({
    chain: chainKernel([{ kind: 'lit', value: 3 }, selfRef('step')]),
    step: stepKernel,
  });
  assert.equal(kbShape.detectJointChainKernelBinding('chain', bindings), null);
});

test('jointchain recogniser rejects a missing or unresolvable step', () => {
  const cases: Array<[string, any[]]> = [
    ['absent arg', [null, selfRef('step')]],
    ['non-self namespace', [local('_m_'), selfRef('step')]],
    ['self-ref to no binding', [selfRef('nope'), selfRef('step')]],
    ['step ref to no binding', [normal(local('_m_')), selfRef('nope')]],
  ];
  for (const [why, args] of cases) {
    const bindings = bindingsOf({ chain: chainKernel(args), step: stepKernel });
    assert.equal(kbShape.detectJointChainKernelBinding('chain', bindings), null, why);
  }
});

test('jointchain recogniser rejects a step binding carrying no IR', () => {
  for (const dep of [null, {}]) {
    const bindings = bindingsOf({
      chain: chainKernel([selfRef('base'), selfRef('step')]),
      base: dep, step: stepKernel,
    });
    assert.equal(kbShape.detectJointChainKernelBinding('chain', bindings), null);
  }
});

test('jointchain recogniser rejects an inline step past step 0', () => {
  // Only step 0 may be inline: steps k ≥ 1 are kernels, which the lift
  // reaches through a binding ref.
  const bindings = bindingsOf({
    chain: chainKernel([normal(local('_m_')), normal(local('_m_'))]),
  });
  assert.equal(kbShape.detectJointChainKernelBinding('chain', bindings), null);
});

test('jointchain recogniser rejects a kernel-first chain', () => {
  const bindings = bindingsOf({
    chain: chainKernel([selfRef('step'), selfRef('step')]),
    step: stepKernel,
  });
  assert.equal(kbShape.detectJointChainKernelBinding('chain', bindings), null,
    'step 0 must be a sampleable DistCall, not a kernel');
});

function jointKernel(fields: Array<{ name: string; value: any }>) {
  return {
    ir: {
      kind: 'call', op: 'functionof', params: ['_m_'], paramKwargs: ['m'],
      body: { kind: 'call', op: 'joint', fields },
    },
  };
}

test('joint recogniser accepts an inline placeholder-scoped component', () => {
  const bindings = bindingsOf({
    jk: jointKernel([
      { name: 'a', value: normal(local('_m_')) },
      { name: 'b', value: selfRef('__anon0') },
    ]),
    __anon0: { ir: normal({ kind: 'lit', value: 0 }) },
  });
  const desc = kbShape.detectJointKernelBinding('jk', bindings);
  assert.ok(desc, 'a mix of inline and anon-bound components matches');
  assert.equal(desc.layout, 'keyword');
  assert.deepEqual(desc.components.map((c: any) => c.surfaceName), ['a', 'b']);
  assert.deepEqual(desc.components[0].distKwargs.mu, local('_m_'));
});

test('joint recogniser rejects a component that is neither a ref nor a call', () => {
  const bindings = bindingsOf({
    jk: jointKernel([{ name: 'a', value: { kind: 'lit', value: 0 } }]),
  });
  assert.equal(kbShape.detectJointKernelBinding('jk', bindings), null);
});

test('joint recogniser rejects a missing or unresolvable component', () => {
  const cases: Array<[string, any]> = [
    ['absent component', null],
    ['non-self namespace', local('_m_')],
    ['self-ref to no binding', selfRef('nope')],
  ];
  for (const [why, value] of cases) {
    const bindings = bindingsOf({ jk: jointKernel([{ name: 'a', value }]) });
    assert.equal(kbShape.detectJointKernelBinding('jk', bindings), null, why);
  }
});

test('joint recogniser rejects a component binding carrying no IR', () => {
  for (const dep of [null, {}]) {
    const bindings = bindingsOf({
      jk: jointKernel([{ name: 'a', value: selfRef('c') }]),
      c: dep,
    });
    assert.equal(kbShape.detectJointKernelBinding('jk', bindings), null);
  }
});

// ---------------------------------------------------------------------
// _collectKernelParamRefs — formal to SURFACE kwarg name
// ---------------------------------------------------------------------
//
// `_executeJointComposite` tests its vector-component set against
// `d.kwargIRs`, which is keyed by the surface kwarg name. Under the
// off-spec bare-name spelling formal and surface name coincide; under a
// `_x_` placeholder they do not, so the mapping has to be explicit.

const collectKernelParamRefs =
  require('../mat-broadcast.ts').collectKernelParamRefs;

function collected(expr: any) {
  const out = new Set<string>();
  collectKernelParamRefs(expr, ['_m_', '_s_'], ['m', 's'], out);
  return Array.from(out).sort();
}

test('collectKernelParamRefs maps a placeholder formal to its surface kwarg', () => {
  assert.deepEqual(collected(local('_m_')), ['m']);
  assert.deepEqual(collected(selfRef('_s_')), ['s'],
    'the post-lift `self` spelling of a formal maps too');
});

test('collectKernelParamRefs descends args and kwargs', () => {
  const expr = {
    kind: 'call', op: 'MvNormal',
    kwargs: {
      mu: { kind: 'call', op: 'add', args: [local('_m_'), selfRef('offset')] },
      cov: selfRef('cov_shared'),
    },
  };
  assert.deepEqual(collected(expr), ['m'],
    'a nested formal is found; closed-over refs are not formals');
});

test('collectKernelParamRefs ignores non-refs, other namespaces, and non-formals', () => {
  assert.deepEqual(collected({ kind: 'lit', value: 1 }), []);
  assert.deepEqual(collected({ kind: 'ref', ns: 'mod', name: '_m_' }), [],
    'a loaded-module ref is never a formal, whatever it is named');
  assert.deepEqual(collected(selfRef('cov_shared')), []);
});

// ---------------------------------------------------------------------
// Collector / substituter parity
// ---------------------------------------------------------------------
//
// `_executeJointComposite` skips every kwarg the collector names when it
// lays out the scalar-arg loop, on the understanding that the substituter
// bound it into the vector component. A formal one of the pair sees and
// the other does not is therefore a silently dropped cell axis. Both are
// defined over `_mapKernelParamRefs`, so the parity is structural; these
// pin the two positions where a hand-written pair would drift.

const substituteKernelParams =
  require('../mat-broadcast.ts').substituteKernelParams;

const BOUND = { m: { kind: 'lit', value: 7 }, s: { kind: 'lit', value: 8 } };

function substituted(expr: any) {
  return substituteKernelParams(expr, ['_m_', '_s_'], ['m', 's'], BOUND);
}

test('collector and substituter both skip a nested reification body', () => {
  // §04 "Placeholders and holes": "A placeholder in an inner `functionof`
  // or `kernelof` **must** be bound there", so `_m_` under the inner body
  // is the INNER scope's formal, never this kernel's.
  const nested = {
    kind: 'call', op: 'functionof', params: ['_m_'], paramKwargs: ['m'],
    body: { kind: 'call', op: 'Normal', kwargs: { mu: local('_m_') } },
  };
  const expr = { kind: 'call', op: 'MvNormal', kwargs: { mu: nested } };
  assert.deepEqual(collected(expr), [],
    'the inner scope re-declares _m_, so no outer formal is collected');
  assert.deepEqual(substituted(expr).kwargs.mu.body.kwargs.mu, local('_m_'),
    'and the substituter leaves that body untouched');
});

test('collector and substituter both descend a record `fields` array', () => {
  // `record` / keyword-form `joint` carry operands in `fields`, not
  // `kwargs`. A formal reached only that way must be seen by both.
  const expr = {
    kind: 'call', op: 'record',
    fields: [{ name: 'a', value: local('_m_') }],
  };
  assert.deepEqual(collected(expr), ['m']);
  assert.deepEqual(substituted(expr).fields[0].value, BOUND.m);
  assert.equal(substituted(expr).fields[0].name, 'a', 'field name survives');
});

test('substituteKernelParams leaves a formal with no broadcast arg alone', () => {
  // `_s_` is a formal, but the caller bound only `m`.
  const expr = { kind: 'call', op: 'Normal', kwargs: { mu: local('_s_') } };
  const out = substituteKernelParams(
    expr, ['_m_', '_s_'], ['m', 's'], { m: BOUND.m });
  assert.deepEqual(out.kwargs.mu, local('_s_'));
});

test('the shared traversal passes a valueless field entry through', () => {
  const expr = { kind: 'call', op: 'record', fields: [{ name: 'a' }, null] };
  const out = substituted(expr);
  assert.deepEqual(out.fields, [{ name: 'a' }, null]);
  assert.deepEqual(collected(expr), []);
});

test('collector and substituter agree on every formal in one expression', () => {
  const expr = {
    kind: 'call', op: 'MvNormal',
    kwargs: {
      mu: { kind: 'call', op: 'add', args: [local('_m_'), selfRef('offset')] },
      cov: { kind: 'call', op: 'record', fields: [{ name: 'd', value: selfRef('_s_') }] },
      inner: {
        kind: 'call', op: 'functionof', params: ['_s_'], paramKwargs: ['s'],
        body: local('_s_'),
      },
    },
  };
  const out = substituted(expr);
  const rewrittenKwargs = new Set<string>();
  for (const [surface, bound] of Object.entries(BOUND)) {
    const hit = JSON.stringify(out).includes(JSON.stringify(bound));
    if (hit) rewrittenKwargs.add(surface);
  }
  assert.deepEqual(Array.from(rewrittenKwargs).sort(), collected(expr),
    'the set the substituter rewrote is the set the collector named');
});

// ---------------------------------------------------------------------
// Located refusal for an inline jointchain step past the base
// ---------------------------------------------------------------------

test('checkJointChainKernelSteps refuses an inline step past the base', () => {
  const inlineStep = {
    kind: 'call', op: 'functionof', params: ['_a_'], paramKwargs: ['a'],
    body: normal(local('_a_')),
    loc: { start: { line: 5, col: 3 }, end: { line: 5, col: 40 } },
  };
  const bindings = bindingsOf({
    chain: chainKernel([normal(local('_m_')), inlineStep]),
    y: {
      ir: {
        kind: 'call', op: 'broadcast',
        args: [selfRef('chain')],
        kwargs: { m: selfRef('ms') },
      },
    },
  });
  (bindings as any).entries = () => Object.entries({
    chain: bindings.get('chain'), y: bindings.get('y'),
  });
  const out = kbShape.checkJointChainKernelSteps(bindings);
  assert.equal(out.length, 1);
  assert.equal(out[0].name, 'y');
  assert.equal(out[0].severity, 'error');
  assert.match(out[0].message, /jointchain argument 2 \(step kernel 1\)/);
  assert.match(out[0].message, /Reified callables/);
  assert.deepEqual(out[0].loc, inlineStep.loc, 'located at the inline step');
});

test('an inline jointchain step reaches the user as a located refusal', async () => {
  // End to end from source: before the refusal this shape carried no
  // diagnostic, kept an `evaluate` derivation, and threw
  // `TypeError: Cannot read properties of undefined (reading 'length')`
  // out of the materialiser.
  const { processSource } = require('../index.ts');
  const orchestrator = require('../orchestrator.ts');
  const materialiser = require('../materialiser.ts');
  const { createWorkerHandler } = require('../worker.ts');
  const src = 'flatppl_compat = "0.1"\n'
    + 'm_per_group = [0.0, 1.0, 2.0]\n'
    + 'group_chain = functionof(\n'
    + '  jointchain(Normal(mu = _m_, sigma = 1.0),\n'
    + '             functionof(Normal(mu = _a_, sigma = 1.0), a = _a_)),\n'
    + '  m = _m_)\n'
    + 'y = broadcast(group_chain, m = m_per_group)\n';
  const lifted = processSource(src);
  assert.deepEqual(lifted.diagnostics, [], 'the program is spec-legal');
  const built = orchestrator.buildDerivations(lifted.bindings);
  const errs = (built.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /jointchain argument 2 \(step kernel 1\)/);
  // Lines are 0-based here; the inline step sits on source line 5.
  assert.equal(errs[0].loc.start.line, 4, 'located at the inline step, not the binding');
  assert.equal(built.derivations.y, undefined,
    'the crashing value-evaluation derivation is dropped');

  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 9 });
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => materialiser.materialiseMeasure(n, ctx),
    sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
    sampleCount: 8, rootKey: [9, 0],
  };
  await assert.rejects(ctx.getMeasure('y'), (err: any) => {
    assert.ok(!(err instanceof TypeError), 'no internal TypeError: ' + err.message);
    return true;
  });
});

// An `entries()`-bearing bindings map, which `checkJointChainKernelSteps`
// iterates (it runs over post-lift bindings, a real Map, in production).
function walkableBindings(obj: Record<string, any>) {
  const b: any = bindingsOf(obj);
  b.entries = () => Object.entries(obj);
  return b;
}

const broadcastOf = (headName: string, extra?: any) => ({
  ir: Object.assign({
    kind: 'call', op: 'broadcast',
    args: [selfRef(headName)],
    kwargs: { m: selfRef('ms') },
  }, extra || {}),
});

test('checkJointChainKernelSteps passes a wholly binding-ref chain', () => {
  assert.deepEqual(kbShape.checkJointChainKernelSteps(walkableBindings({
    chain: chainKernel([normal(local('_m_')), selfRef('step')]),
    step: stepKernel,
    y: broadcastOf('chain'),
  })), []);
});

test('checkJointChainKernelSteps ignores everything that is not this shape', () => {
  const inline = { kind: 'call', op: 'functionof', params: ['_a_'], body: normal(local('_a_')) };
  const chain = chainKernel([normal(local('_m_')), inline]);
  const cases: Array<[string, Record<string, any>]> = [
    ['a binding with no IR', { y: {} }],
    ['a non-broadcast call', { y: { ir: { kind: 'call', op: 'iid', args: [selfRef('chain')] } } }],
    ['a broadcast with no args array', { chain, y: broadcastOf('chain', { args: undefined }) }],
    ['a non-ref broadcast head', { y: { ir: { kind: 'call', op: 'broadcast', args: [normal(local('_m_'))] } } }],
    ['a head naming no binding', { y: broadcastOf('absent') }],
    ['a head binding that is not a functionof', { chain: { ir: normal(local('_m_')) }, y: broadcastOf('chain') }],
    ['a kernel body that is not a jointchain', {
      chain: { ir: { kind: 'call', op: 'functionof', params: ['_m_'], body: normal(local('_m_')) } },
      y: broadcastOf('chain'),
    }],
    ['a jointchain with no args array', {
      chain: { ir: { kind: 'call', op: 'functionof', params: ['_m_'], body: { kind: 'call', op: 'jointchain' } } },
      y: broadcastOf('chain'),
    }],
  ];
  for (const [why, obj] of cases) {
    assert.deepEqual(kbShape.checkJointChainKernelSteps(walkableBindings(obj)), [], why);
  }
  assert.deepEqual(kbShape.checkJointChainKernelSteps(null), [],
    'no bindings map at all');
});

test('checkJointChainKernelSteps falls back to the jointchain loc', () => {
  // A step carrying no `loc` of its own (a synthesised node) still gets a
  // located message — the enclosing jointchain's span.
  const chainLoc = { start: { line: 2, col: 0 }, end: { line: 2, col: 30 } };
  const chain: any = chainKernel([normal(local('_m_')), { kind: 'lit', value: 3 }]);
  chain.ir.body.loc = chainLoc;
  const out = kbShape.checkJointChainKernelSteps(walkableBindings({
    chain, y: broadcastOf('chain'),
  }));
  assert.equal(out.length, 1);
  assert.deepEqual(out[0].loc, chainLoc);
});

test('joint recogniser rejects an inline non-distribution component', () => {
  const bindings = bindingsOf({
    jk: jointKernel([
      { name: 'a', value: { kind: 'call', op: 'iid', args: [normal(local('_m_'))] } },
    ]),
  });
  assert.equal(kbShape.detectJointKernelBinding('jk', bindings), null,
    'nested composites are not recognised here');
});
