'use strict';

// A weight function whose reified body applies a SECOND reification — the
// amplitude-analysis spelling, where `intensity_fn = functionof(intensity, m=m,
// c=c)` traces a subgraph that itself applies `angular_tensors = functionof(
// record(Z0, Z1, Z2), p1=p1, p2=p2, p3=p3)`.
//
// Spec §04 "Specifying reification boundaries": "A specified boundary node `a`
// can be thought of as being substituted with a new node, generated via
// `elementof(valueset(a))`, in the reified graph." Applying the callable binds
// those inputs, so the applied body is the traced graph with each boundary
// replaced by the argument expression — the lift performs exactly that
// substitution and stores the substituted CLONES as anonymous bindings.
//
// The clones carry the outer function's `%local` formals, so they are
// meaningless standalone: no derivation classifies them (a `vector` with
// computed elements is neither an all-literal array nor an all-ref tuple nor
// `isEvaluable`), and `fixedValues` cannot evaluate them. #184 unwrapped only
// the body's ROOT ref chain, so every clone BELOW the root stayed a bare ref,
// the cascade-prune found it unresolvable, and the whole measure vanished
// silently.
//
// The fix inlines a nested ref exactly when its binding transitively mentions
// one of this reification's formals. That criterion is what makes the deep walk
// sound where #184's rejected version was not: a formal-carrying binding CANNOT
// be resolved outside the body, so inlining it is forced; a binding that carries
// no formal is standalone-resolvable and stays a ref. No name is rewritten.
//
// ── THE PERMUTATION ORACLE ───────────────────────────────────────────────────
// Rewriting an inner boundary ref to the outer coordinate of the same NAME is
// unsound, because the composition may permute the arguments. The oracle below
// permutes them deliberately:
//
//   g(a, b) = a² + b          (reified over boundaries a, b)
//   f(x, y) = g(b = x, a = y) (reified over boundaries x, y — PERMUTED)
//   pw(p, q) = f(p, q)
//
// so pw(p, q) = q² + p. Name-based rewriting (a↦p, b↦q, matching by position in
// the formal list) would instead give p² + q. Both closed forms are derived by
// hand and BOTH are asserted — the right one must hold and the wrong one must
// not, since a test that only asserted the right value could pass on a build
// that happened to agree by symmetry.
//
//   at (p, q) = (0.3, 0.8):   right = 0.8² + 0.3 = 0.94
//                             wrong = 0.3² + 0.8 = 0.89
//
// The base is `Lebesgue` over the unit square (density 1) and the measure is
// UNNORMALIZED, so `logdensityof` is exactly log w(p, q) — a closed form with no
// quadrature and no sample-count dependence.
//
// A second, mass-level discriminator uses an ASYMMETRIC box p∈[0,1], q∈[0,2]:
//   Z_right = ∫₀¹∫₀² (q² + p) dq dp = ∫₀¹ (8/3 + 2p) dp = 8/3 + 1 = 11/3
//   Z_wrong = ∫₀¹∫₀² (p² + q) dq dp = ∫₀¹ (2p² + 2)  dp = 2/3 + 2 =  8/3
// The N-D box normalizer carries a pre-existing ~0.5% bias (#184's header), far
// inside the 27% gap between those two values.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const { collectSelfRefs } = require('../ir-shared.ts');

function processed(src: string) {
  const proc = processSource(src);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  return proc;
}

function derivationsOf(src: string) {
  return orchestrator.buildDerivations(processed(src).bindings).derivations;
}

async function scoreOf(src: string, binding: string, N = 4096, seed = 7) {
  const proc = processed(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: seed, rootSeed: seed, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return (await ctx.getMeasure(binding)).samples[0];
}

// The nested reification. `va` is a vector with COMPUTED elements, reached
// through an `aggregate` — the two ingredients that make the substituted clone
// unclassifiable and put a reducer head (`sum`) in the weight body.
function permutedModel(qHi: string) {
  return `
x = elementof(interval(0.0, 1.0))
y = elementof(interval(0.0, ${qHi}))
a = elementof(interval(0.0, 1.0))
b = elementof(interval(0.0, 1.0))
va = [a^2, b]
inner = aggregate(sum, [], va[.i])
g = functionof(inner, a = a, b = b)
outer = g(b = x, a = y)
f = functionof(outer, x = x, y = y)
pw(p, q) = f(p, q)
`;
}

const SQUARE = `
M = weighted(pw, Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 1.0))))
`;

// =====================================================================
// Derivation
// =====================================================================

test('a weight whose reified body applies a second reification derives', () => {
  const d = derivationsOf(permutedModel('1.0') + SQUARE);
  assert.equal(d['M'] && d['M'].kind, 'weighted',
    'the nested-reification weight was cascade-pruned: derivations are '
      + Object.keys(d).join(', '));
});

// =====================================================================
// The permutation oracle — assert the right closed form AND reject the wrong one
// =====================================================================

test('the nested weight scores the BETA-REDUCED closed form q² + p', async () => {
  const got = await scoreOf(
    permutedModel('1.0') + SQUARE + '\nld = logdensityof(M, [0.3, 0.8])\n', 'ld');
  const right = Math.log(0.94);      // 0.8² + 0.3, the permuted composition
  const wrong = Math.log(0.89);      // 0.3² + 0.8, name-based rewriting
  assert.ok(Math.abs(got - right) < 1e-12,
    'log density ' + got + ' ≠ log 0.94 = ' + right);
  assert.ok(Math.abs(got - wrong) > 1e-3,
    'log density ' + got + ' matches the NAME-REWRITTEN form log 0.89 = ' + wrong
      + ' — the inner boundaries were matched by name, not by application');
});

test('the score is exact — it does not move with the sample count', async () => {
  const src = permutedModel('1.0') + SQUARE + '\nld = logdensityof(M, [0.3, 0.8])\n';
  const vals: number[] = [];
  for (const N of [512, 4096, 32768]) vals.push(await scoreOf(src, 'ld', N));
  for (const v of vals) {
    assert.ok(Math.abs(v - vals[0]) < 1e-12,
      'an unnormalized weighted density needs no quadrature, got ' + vals.join(' / '));
  }
});

test('totalmass over the ASYMMETRIC box is 11/3, not 8/3', async () => {
  const got = await scoreOf(permutedModel('2.0') + `
Z = totalmass(weighted(pw, Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 2.0)))))
`, 'Z');
  assert.ok(Math.abs(got - 11 / 3) / (11 / 3) < 0.01,
    'totalmass ' + got + ' ≠ 11/3 = ' + (11 / 3) + ' within 1%');
  assert.ok(Math.abs(got - 8 / 3) / (8 / 3) > 0.05,
    'totalmass ' + got + ' matches the name-rewritten Z = 8/3 = ' + (8 / 3));
});

// =====================================================================
// The 1-level control, unchanged from #184
// =====================================================================

test('a plain 1-level reified-boundary weight still scores log w exactly', async () => {
  // w(t) = t², base Lebesgue([0,1]), unnormalized ⇒ log density = log(0.8²).
  const got = await scoreOf(`
t = elementof(interval(0.0, 1.0))
graph_out = t^2
w = functionof(graph_out, t = t)
M = weighted(w, Lebesgue(support = interval(0.0, 1.0)))
ld = logdensityof(M, 0.8)
`, 'ld');
  assert.ok(Math.abs(got - Math.log(0.64)) < 1e-12,
    '1-level control: ' + got + ' ≠ log 0.64 = ' + Math.log(0.64));
});

// =====================================================================
// The ref-histogram guard
// =====================================================================
//
// #184's rejected deep walk passed every test while splicing a traced subgraph
// into the weight body, and only a ref-histogram diff caught it. Assert the two
// halves of the invariant that makes THIS walk sound, on the rewritten body:
//
//   COMPLETENESS — no surviving ref carries a formal. A formal-carrying binding
//     resolves to nothing standalone, so one left behind would prune the measure.
//   SOUNDNESS — every surviving ref resolves (derivation, fixed value, or a
//     builtin with no binding at all). A ref that resolves to nothing means the
//     pass spliced in a subgraph it should have left alone.
//
// Non-vacuity is asserted too: the body must actually still hold refs, and the
// inlining must have happened (the derivation exists, checked above).

test('the rewritten weight body carries no formal-carrying ref, and every ref resolves', () => {
  const built = orchestrator.buildDerivations(processed(permutedModel('1.0') + SQUARE).bindings);
  const { bindings, derivations, fixedValues } = built;
  const pw = bindings.get('pw');
  const formals = new Set<string>(pw.ir.params);
  const refs = new Set<string>(collectSelfRefs(pw.ir.body));

  const carriesFormal = (name: string, seen = new Set<string>()): boolean => {
    if (seen.has(name)) return false;
    seen.add(name);
    const b = bindings.get(name);
    if (!b || !b.ir) return false;
    let found = false;
    (function scan(n: any): void {
      if (found || n == null || typeof n !== 'object') return;
      if (Array.isArray(n)) { n.forEach(scan); return; }
      if (n.kind === 'ref' && n.ns === '%local' && formals.has(n.name)) { found = true; return; }
      if (n.kind === 'ref' && n.ns === 'self' && n.name) {
        if (carriesFormal(n.name, seen)) found = true;
        return;
      }
      for (const k in n) scan(n[k]);
    })(b.ir);
    return found;
  };

  assert.ok(refs.size > 0, 'the body should still hold refs — the check is vacuous otherwise');
  for (const r of refs) {
    assert.ok(!carriesFormal(r),
      'ref ' + r + ' survived in the weight body but carries a formal of pw, '
        + 'so nothing can resolve it standalone');
    const b = bindings.get(r);
    const resolves = Object.prototype.hasOwnProperty.call(derivations, r)
      || (fixedValues && fixedValues.has(r))
      || !b;                                  // a builtin reducer head (sum)
    assert.ok(resolves,
      'ref ' + r + ' survived in the weight body but resolves to nothing — the '
        + 'pass spliced in a subgraph it should have left alone');
  }
});

test('an aggregate reducer head in a weight body is not demanded as a binding', () => {
  // `sum` is a builtin, absent from `bindings`. The generic ref walk exempts a
  // callable head; the weighted arm did not, so a reduction anywhere in a
  // reified weight body pruned the measure. Reachable only once the body is
  // inlined, which is why it ships here.
  const d = derivationsOf(`
u = elementof(interval(0.0, 1.0))
vv = [u, u^2]
tot = aggregate(sum, [], vv[.i])
w = functionof(tot, u = u)
M = weighted(w, Lebesgue(support = interval(0.0, 1.0)))
`);
  assert.equal(d['M'] && d['M'].kind, 'weighted',
    'a reducer head in the weight body pruned the measure: '
      + Object.keys(d).join(', '));
});

test('a reduced weight body scores log(t + t²)', async () => {
  // w(t) = t + t², unnormalized over Lebesgue([0,1]) ⇒ log density = log w(t).
  // At t = 0.6: 0.6 + 0.36 = 0.96.
  const got = await scoreOf(`
u = elementof(interval(0.0, 1.0))
vv = [u, u^2]
tot = aggregate(sum, [], vv[.i])
w = functionof(tot, u = u)
M = weighted(w, Lebesgue(support = interval(0.0, 1.0)))
ld = logdensityof(M, 0.6)
`, 'ld');
  assert.ok(Math.abs(got - Math.log(0.96)) < 1e-12,
    'reducer weight: ' + got + ' ≠ log 0.96 = ' + Math.log(0.96));
});
