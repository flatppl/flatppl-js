'use strict';

// =====================================================================
// ir-walk-sharing.test.ts
// =====================================================================
//
// The lowered IR is a DAG, not a tree. One binding's IR object is
// reached from many parents, so on a graph-shaped model — an amplitude
// body reusing one isobar across coherent terms — a tree walk pays a
// ~50x factor. Two walks used to throw that sharing away:
//
//   • materialiser-shared.inlineBoundaryDerivations rebuilt an object
//     per node, so its output was a fully un-shared tree.
//   • ir-shared.collectSelfRefs re-walked a shared subtree once per
//     parent.
//
// Both now preserve sharing. That makes object IDENTITY part of the
// contract rather than an accident, so this file pins it: a subtree the
// walk does not change comes back as the SAME object, and a subtree it
// does change is rebuilt ONCE and installed at every position.
//
// The identity contract is what makes the dedup in collectSelfRefs pay
// off, and it is also what the in-place `normalize` rewrite depends on
// staying out of (see normalize-rewrite-privacy.test.ts).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { inlineBoundaryDerivations } = require('../materialiser-shared.ts');
const { collectSelfRefs } = require('../ir-shared.ts');
const irWalk = require('../ir-walk.ts');
const { processSource, orchestrator, materialiser } = require('../index.ts');
const { createWorkerHandler } = require('../worker.ts');

const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
const call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });

// A reification whose params are IDENTIFIER-bound, which is the shape
// `ir-walk.identifierBoundParams` recognises: a `functionof` carrying
// `params` plus a `paramSources` entry of kind 'binding' for each. Only
// `body` descends under the extended scope.
const reified = (params: string[], body: any) => ({
  kind: 'call', op: 'functionof',
  params, paramKwargs: params.slice(),
  paramSources: params.map((name) => ({ kind: 'binding', name })),
  body,
});

// `inlineBoundaryDerivations` inlines a self-ref whose binding has a
// call-shaped `ir` and no derivation. Everything else is left alone.
function ctxOf(bindingIRs: Record<string, any>) {
  return { bindings: new Map(Object.entries(bindingIRs).map(([k, v]) => [k, { ir: v }])),
    derivations: {} };
}

// Distinct objects vs. positions, so a claim about sharing is measured
// rather than asserted.
function sizes(root: any) {
  const distinct = new Set<any>();
  let positions = 0;
  (function go(n: any) {
    if (!n || typeof n !== 'object') return;
    positions++;
    distinct.add(n);
    irWalk.forEachIRChild(n, go);
  })(root);
  return { positions, distinct: distinct.size };
}

// ── identity preservation ─────────────────────────────────────────────

test('an unchanged tree comes back as the very same object', () => {
  // No ref is inlinable, so the walk has nothing to rewrite.
  const shared = call('mul', ref('theta'), ref('theta'));
  const ir = call('add', shared, shared);
  const out = inlineBoundaryDerivations(ir, new Set(['theta']), ctxOf({}));
  assert.equal(out, ir, 'the root was rebuilt despite no change');
  assert.equal(out.args[0], shared, 'a child was rebuilt despite no change');
  assert.equal(out.args[0], out.args[1], 'the shared child was un-shared');
  assert.equal(out.args, ir.args, 'the args array was rebuilt despite no change');
});

test('an unchanged array position keeps its array object', () => {
  const ir = { kind: 'call', op: 'select', args: [ref('a')], branches: [ref('b'), ref('c')] };
  const out = inlineBoundaryDerivations(ir, new Set(['a', 'b', 'c']), ctxOf({}));
  assert.equal(out, ir);
  assert.equal(out.branches, ir.branches);
});

test('a changed subtree is rebuilt ONCE and installed at every position', () => {
  // `a` inlines to its body, so the shared subtree does change — and it
  // must change into ONE new object, not one per parent. That is the
  // whole point: the pre-fix walk produced a fresh copy per reference.
  const shared = call('mul', ref('a'), ref('theta'));
  const ir = call('add', shared, shared);
  const out = inlineBoundaryDerivations(ir, new Set(['theta']),
    ctxOf({ a: call('exp', ref('theta')) }));

  assert.notEqual(out, ir, 'the inlining did not happen');
  assert.equal(out.args[0], out.args[1], 'the shared subtree was rebuilt per parent');
  assert.equal(out.args[0].args[0].op, 'exp', 'the ref was not inlined');
  // The unchanged sibling positions still alias the input.
  assert.equal(out.args[0].args[1], shared.args[1], 'an unchanged leaf was copied');
  // The input is untouched: the walk rebuilds, it does not mutate.
  assert.equal(ir.args[0], shared);
  assert.equal(shared.args[0].name, 'a');
});

test('a changed array position yields a new array, unchanged siblings shared', () => {
  const keep = ref('theta');
  const ir = { kind: 'call', op: 'add', args: [keep, ref('a')] };
  const out = inlineBoundaryDerivations(ir, new Set(['theta']),
    ctxOf({ a: call('log', ref('theta')) }));
  assert.notEqual(out.args, ir.args, 'the array was reused despite a changed element');
  assert.equal(out.args[0], keep, 'an unchanged element was copied');
  assert.equal(out.args[1].op, 'log');
});

test('sharing survives one level of inlining: the memo is bypassed, identity is not', () => {
  // `outer` inlines to a body that itself references `inner`, so the
  // nested walk runs with the cycle guard non-empty — where the memo is
  // deliberately off, because the answer there depends on which names
  // are being inlined. Identity preservation still applies, so the
  // shared leaf must not be duplicated.
  const leaf = call('exp', ref('theta'));
  const innerBody = call('add', leaf, leaf);
  // ONE ref object at two positions. The memo is keyed on node identity,
  // not on value, so two equal-but-distinct ref nodes would legitimately
  // inline twice — the DAG the lowering hands over shares the object.
  const outerRef = ref('outer');
  const ir = call('mul', outerRef, outerRef);
  const out = inlineBoundaryDerivations(ir, new Set(['theta']),
    ctxOf({ outer: call('neg', ref('inner')), inner: innerBody }));

  assert.equal(out.args[0], out.args[1], 'the two `outer` refs inlined to different objects');
  const body = out.args[0].args[0];
  assert.equal(body.args[0], body.args[1], 'the shared leaf was un-shared inside the inlining');
  assert.equal(body.args[0], leaf, 'the unchanged leaf was rebuilt');
});

test('a self-referential binding is stopped by the cycle guard, not by the memo', () => {
  // `a = mul(a, theta)`: the inner `a` hits `visiting` and stays a ref.
  const body = call('mul', ref('a'), ref('theta'));
  const out = inlineBoundaryDerivations(call('neg', ref('a')), new Set(['theta']),
    ctxOf({ a: body }));
  assert.equal(out.args[0].op, 'mul');
  assert.deepEqual(out.args[0].args[0], ref('a'));
});

test('the walk collapses a wide DAG instead of expanding it', () => {
  // A binary tree of shared layers: 2^depth positions over depth+1
  // objects. The pre-fix walk returned 2^depth objects.
  let node: any = call('exp', ref('a'));
  for (let i = 0; i < 12; i++) node = call('add', node, node);
  const before = sizes(node);
  const out = inlineBoundaryDerivations(node, new Set(['theta']),
    ctxOf({ a: call('log', ref('theta')) }));
  const after = sizes(out);
  assert.ok(before.positions > 4000, 'the input is not wide enough to prove anything');
  // 12 `add` layers + the `exp` base + its `a` ref.
  assert.equal(before.distinct, 14);
  // 13 rebuilt layers, plus the inlined `log(theta)` and its ref — the
  // `a` ref is one object at 4096 positions, so the memo inlines it once.
  // Pre-fix this was one object PER POSITION.
  assert.equal(after.distinct, 15,
    'the walk un-shared the DAG: ' + before.distinct + ' objects in, '
    + after.distinct + ' out over ' + after.positions + ' positions');
  assert.ok(after.positions >= before.positions,
    'the inlining lost positions: ' + before.positions + ' -> ' + after.positions);
});

// ── collectSelfRefs over a DAG ────────────────────────────────────────
//
// The collector skips a (node, scope) pair it has already visited. That
// is observationally identical only because its visitor unions names
// into a set. These cases compare it against the plain tree walk on the
// shapes where the two could disagree: a diamond, and a name that is
// shadowed on one path and free on another.

function treeWalkRefs(ir: any) {
  const seen = new Set<string>();
  irWalk.walkIRScoped(ir, (n: any, shadowed: Set<string>) => {
    if (n && n.kind === 'ref' && n.ns === 'self' && !shadowed.has(n.name)) seen.add(n.name);
  });
  return seen;
}

test('collectSelfRefs on a diamond agrees with the tree walk', () => {
  const shared = call('add', ref('mu'), ref('sigma'));
  const ir = call('mul', shared, call('neg', shared));
  assert.deepEqual([...collectSelfRefs(ir)].sort(), ['mu', 'sigma']);
  assert.deepEqual([...collectSelfRefs(ir)].sort(), [...treeWalkRefs(ir)].sort());
});

test('collectSelfRefs still shadows a reified boundary param reached twice', () => {
  // `x` is an identifier-bound param of the callable, so a `self` ref to
  // it inside the body is the callable's INPUT, not an outer dependency.
  // The callable is reached from two parents; the skip must not let the
  // shadowed name leak out through the second visit, and must not
  // suppress the same name where it is genuinely free.
  const fn = reified(['x'], call('mul', ref('x'), ref('mu')));
  const ir = call('add', call('neg', fn), call('exp', fn), ref('x'));

  const got = [...collectSelfRefs(ir)].sort();
  assert.deepEqual(got, [...treeWalkRefs(ir)].sort(),
    'the sharing-aware walk disagrees with the tree walk');
  assert.ok(got.includes('mu'), 'lost the free outer ref');
  assert.ok(got.includes('x'), 'lost the OUTER `x`, which is not shadowed');
});

test('collectSelfRefs sees a name that is only free on the deeper path', () => {
  const fn = reified(['shadowed_only_inside'],
    call('add', ref('shadowed_only_inside'), ref('always_free')));
  const ir = call('mul', fn, fn);
  const got = [...collectSelfRefs(ir)].sort();
  assert.deepEqual(got, [...treeWalkRefs(ir)].sort());
  assert.deepEqual(got, ['always_free']);
});

// ── the model the change was measured on ──────────────────────────────

const FIXTURES = path.join(__dirname, 'fixtures');
const DALITZ = fs.readFileSync(
  path.join(FIXTURES, 'dminus-to-3pi-amplitude.flatppl'), 'utf8');

function buildCtx(src: string, N: number, seed = 3) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.linkedBindings || proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: N, rootKey: seed, rootSeed: seed, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

function bits(v: number) {
  const b = new Float64Array([v]);
  return Buffer.from(b.buffer).toString('hex');
}

// The value is pinned, AND checked against a path that does not go
// through the density walker at all.
//
// `amplitude_measure = weighted(phase_weight, phase_space)` over a
// Lebesgue box, so spec §06 makes its log-density at x exactly
// log w(x) — the reference measure contributes 0. `phase_weight(1.0,
// 0.35)` is a plain scalar binding, evaluated by the value path, so
// log of it is an engine-independent-of-the-density-walker reading of
// the same number.
//
// PROVENANCE of the hex: measured on main 1e2560f5 (before either walk
// fix), sampleCount 1, seed 3 — and unchanged at (N, seed) of (1, 99),
// (8, 3) and (256, 7), so it pins the number rather than a schedule.
const DALITZ_LD_BITS = 'c0378bb4e32c0e40';   // 3.7719186882959264

test('Dalitz amplitude: the density is bit-identical to the pre-fix value', async () => {
  const ctx = buildCtx(DALITZ
    + '\nprobe_weight = phase_weight(1.0, 0.35)\n'
    + 'ld_probe = logdensityof(amplitude_measure, [1.0, 0.35])\n', 1);
  const ld = await ctx.getMeasure('ld_probe');
  const got = ld.samples[0];

  assert.equal(bits(got), DALITZ_LD_BITS,
    'logdensityof(amplitude_measure, [1.0, 0.35]) moved: got ' + got
    + ' [' + bits(got) + '], pre-fix ' + DALITZ_LD_BITS);

  // Cross-path check, independent of the pin above.
  const w = await ctx.getMeasure('probe_weight');
  assert.equal(bits(got), bits(Math.log(w.samples[0])),
    'the density walker disagrees with log(phase_weight(1.0, 0.35)) = '
    + Math.log(w.samples[0]));
});

test('Dalitz amplitude: the walk receives a DAG and keeps it one', async () => {
  // The measurement behind the change, asserted rather than recorded in
  // a comment: the body handed to inlineBoundaryDerivations is ~50x
  // shared, and the output stays about as compact as the input. Pre-fix
  // the output was ~42k objects — larger than the input's position
  // count — so a wide margin here still fails loudly on a regression.
  const matShared = require('../materialiser-shared.ts');
  const orig = matShared.inlineBoundaryDerivations;
  const log: any[] = [];
  matShared.inlineBoundaryDerivations = function (ir: any, bs: any, c: any) {
    const out = orig.call(this, ir, bs, c);
    log.push({ inp: sizes(ir), out: sizes(out) });
    return out;
  };
  try {
    const ctx = buildCtx(DALITZ
      + '\nld_probe = logdensityof(amplitude_measure, [1.0, 0.35])\n', 1);
    await ctx.getMeasure('ld_probe');
  } finally {
    matShared.inlineBoundaryDerivations = orig;
  }

  const wide = log.filter((e) => e.inp.positions > 10000);
  assert.ok(wide.length > 0,
    'the density path no longer walks a wide body; sizes seen: '
    + JSON.stringify(log.map((e) => e.inp)));
  for (const e of wide) {
    assert.ok(e.inp.positions / e.inp.distinct > 20,
      'the input stopped being a DAG: ' + e.inp.positions + ' positions over '
      + e.inp.distinct + ' objects');
    assert.ok(e.out.distinct < e.inp.distinct * 4,
      'the walk un-shared its input: ' + e.inp.distinct + ' objects in, '
      + e.out.distinct + ' out');
  }
});
