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

test('joint recogniser rejects an inline non-distribution component', () => {
  const bindings = bindingsOf({
    jk: jointKernel([
      { name: 'a', value: { kind: 'call', op: 'iid', args: [normal(local('_m_'))] } },
    ]),
  });
  assert.equal(kbShape.detectJointKernelBinding('jk', bindings), null,
    'nested composites are not recognised here');
});
