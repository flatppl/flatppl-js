// Pinning tests for the `Y = polyeval.([C], X)`-shape Ref-wrap
// broadcast — broadcasts with an inline `vector(...)` arg and a user-
// defined functionof head. Both the lift's array-literal hoisting and
// `derivationRefsValid`'s callable-head exemption are required for
// these bindings to classify; without either, the binding has no
// derivation and the viewer reports "Not plottable".

const test = require('node:test');
const assert = require('node:assert');
const { processSource } = require('../index.ts');
const { buildDerivations } = require('../derivations.ts');

function getDerivation(src: string, name: string) {
  const proc = processSource(src);
  const out = buildDerivations(proc.bindings);
  return out.derivations[name];
}

test('polyeval-shape: Y = polyeval.([C], X) classifies as evaluate', () => {
  const d = getDerivation(`
polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
C = [2.3, 1.5, 0.7]
X ~ iid(Normal(0,1), 10)
Y = polyeval.([C], X)
`, 'Y');
  assert.ok(d, 'Y should have a derivation');
  assert.equal(d.kind, 'evaluate');
});

test('inline ArrayLiteral arg to broadcast is hoisted to anon binding', () => {
  // `[C]` inside the broadcast args lifts to an `__anonN` binding
  // that classifies as kind:'tuple' (the C ref makes it a tuple of
  // refs rather than a literal array). Y's IR then carries a bare
  // ref to that anon, which `isEvaluable` resolves cleanly.
  const proc = processSource(`
f = (a, b) -> a + b
A = [1.0, 2.0]
B = [3.0, 4.0, 5.0]
Y = f.([A], B)
`);
  const out = buildDerivations(proc.bindings);
  // The broadcast IR's first non-head arg should be a bare ref now,
  // not an inline vector(...) call.
  const y = out.bindings.get('Y');
  assert.ok(y && y.ir);
  assert.equal(y.ir.op, 'broadcast');
  const firstCollectionArg = y.ir.args[1];
  assert.equal(firstCollectionArg.kind, 'ref',
    'inline [A] arg should be hoisted to a ref');
});

test('user-fn-ref broadcast head doesn\'t cascade-prune dependents', () => {
  // The cascade-prune sweep in derivationRefsValid skips callable-
  // head refs (broadcast / aggregate args[0] bound to fn /
  // functionof / kernelof / bijection bindings) since those resolve
  // via env.__resolveFnBody at evaluation time, not via the
  // derivations / fixedValues map.
  const d = getDerivation(`
myFn = (a, b) -> a + b
X = [1.0, 2.0, 3.0]
Y = [4.0, 5.0, 6.0]
Z = myFn.(X, Y)
`, 'Z');
  assert.ok(d, 'Z should not cascade-prune because myFn is a callable binding');
  assert.equal(d.kind, 'evaluate');
});

test('inlineCallableRefs splices user-fn body in place of a self-ref', () => {
  // The materialiser's matEvaluate inlines callable refs before
  // sending the IR to the worker. Since fusion (a) Step 2 and the
  // existing single-op dissolver rewrite many broadcast-of-user-fn
  // shapes at dissolve time, the runtime inlining path is exercised
  // by directly constructing a broadcast IR that didn't dissolve —
  // here, a body containing a kernel reference (refused by both
  // fusion (a) and Phase 3 because the body isn't a value op).
  // The test purpose is the inlineCallableRefs MECHANISM, not any
  // particular surface example.
  const { inlineCallableRefs } = require('../materialiser-shared.ts');
  const proc = processSource(`
myfn = (a, b) -> a + b
A = [1.0, 2.0, 3.0]
B = [4.0, 5.0, 6.0]
`);
  const out = buildDerivations(proc.bindings);
  // Construct a broadcast IR explicitly that did NOT go through the
  // dissolver (no binding holds it). Inlining the self-ref `myfn`
  // head should produce a functionof.
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [
      { kind: 'ref', ns: 'self', name: 'myfn' },
      { kind: 'ref', ns: 'self', name: 'A' },
      { kind: 'ref', ns: 'self', name: 'B' },
    ],
  };
  assert.equal(bcIR.args[0].kind, 'ref');
  assert.equal(bcIR.args[0].name, 'myfn');
  const inlined = inlineCallableRefs(bcIR, out.bindings);
  assert.equal(inlined.args[0].kind, 'call');
  assert.equal(inlined.args[0].op, 'functionof');
  assert.ok(Array.isArray(inlined.args[0].params),
    'inlined functionof carries its params');
  assert.ok(inlined.args[0].body, 'inlined functionof carries its body');
});

test('regression: kernelof MEASURE position still cascade-prunes', () => {
  // The callable-head exemption is NARROW — it only applies to args[0]
  // of broadcast / aggregate. `logdensityof(k, x)` uses k in a
  // MEASURE position (the materialiser needs k to be derivable as
  // a measure); the cascade-prune sweep here must still fire when
  // k is a kernelof binding without a corresponding measure
  // derivation. Pinning so the polyeval fix doesn't accidentally
  // re-enable spec-violating density classifications.
  const proc = processSource(`
m  = Normal(mu = 0.0, sigma = 1.0)
k  = kernelof(m)
lp = logdensityof(k, 1.5)
`);
  const out = buildDerivations(proc.bindings);
  assert.ok(!('lp' in out.derivations),
    'logdensityof(kernelof, …) must cascade-prune (callable in measure position)');
});
