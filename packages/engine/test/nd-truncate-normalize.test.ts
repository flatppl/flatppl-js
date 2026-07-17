'use strict';

// parseTruncationBox (mat-density.ts) — set IR → per-axis truncation box for
// N-D quadrature (engine G2). Parses interval(...) (1-D) and cartprod(...)
// of intervals (N-D) into { lo, hi, kind } axes; null for any other set
// shape (caller defers to the existing by-name/normalizer path); throws
// above the dimension cap of 3. Consumer wiring is a separate task.

const test = require('node:test');
const assert = require('node:assert');
const matDensity = require('../mat-density.ts');
const { parseTruncationBox } = matDensity;
const { processSource, orchestrator } = require('../index.ts');
const shared = require('../materialiser-shared.ts');

const iv = (lo: any, hi: any) => ({ kind: 'call', op: 'interval', args: [lit(lo), lit(hi)] });
const lit = (v: any) => ({ kind: 'lit', value: v });
const cart = (...xs: any[]) => ({ kind: 'call', op: 'cartprod', args: xs });

test('1-D interval → one finite axis', () => {
  assert.deepEqual(parseTruncationBox(iv(-2, 3)),
    [{ lo: -2, hi: 3, kind: 'finite' }]);
});

test('2-D cartprod of intervals → two finite axes', () => {
  assert.deepEqual(parseTruncationBox(cart(iv(0, 1), iv(-4, 4))),
    [{ lo: 0, hi: 1, kind: 'finite' }, { lo: -4, hi: 4, kind: 'finite' }]);
});

test('semi-infinite interval(0, inf) → semi-lo', () => {
  assert.deepEqual(parseTruncationBox(iv(0, Infinity)),
    [{ lo: 0, hi: Infinity, kind: 'semi-lo' }]);
});

test('unrecognized set shape → null', () => {
  assert.equal(parseTruncationBox({ kind: 'call', op: 'stdsimplex', args: [lit(3)] }), null);
  assert.equal(parseTruncationBox(cart(iv(0, 1), { kind: 'ref', name: 'S' })), null);
});

test('over-cap dimension throws', () => {
  assert.throws(() => parseTruncationBox(cart(iv(0, 1), iv(0, 1), iv(0, 1), iv(0, 1))),
    /dimension 4 exceeds .*cap.*3/i);
});

// makeIntegrandND — weight IR + change-of-variables (Task 3). The weight
// body IR here is hand-built to the shape `samplerLib.evaluateExpr` (sampler.ts)
// actually consumes: `{kind:'lit',value}`, `{kind:'ref',name}` (resolveRef
// only reads `ir.name`, ns-agnostic), and `{kind:'call',op,args}` dispatched
// through ARITH_OPS. Confirmed against a REAL lowering — processSource('w =
// x -> 1.0/(1.0+x*x)') then orchestrator.buildDerivations(...).bindings.get
// ('w').ir — which lowers the surface `/` to op **'divide'** (true division),
// NOT 'div' (spec §07: `div` is integer floor division, `Math.floor(a/b)` in
// sampler.ts's ARITH_OPS — using it here would silently compute
// floor(1/(1+x²)) ≡ 0 a.e. and the π/2 test below would not integrate to
// the intended function at all).
const { makeIntegrandND } = matDensity;
const { adaptiveCubature } = require('../quadrature.ts');

// weight body IR for  1  (constant) as a functionof body; evaluateExpr resolves
// a bare literal. Use the sampler evaluator's literal path.
function litBody(v: number) { return { kind: 'lit', value: v }; }

test('makeIntegrandND: finite 2-D box of constant 1 integrates to area', () => {
  const axes = [{ lo: 0, hi: 2, kind: 'finite' }, { lo: -1, hi: 3, kind: 'finite' }];
  const integ = makeIntegrandND(litBody(1), ['x', 'y'], axes, {}, {});
  const r = adaptiveCubature(integ, 2);
  assert.ok(Math.abs(r.Z - 8.0) < 1e-8, `Z=${r.Z}`); // area 2×4
});

test('makeIntegrandND: semi-lo change-of-variables integrates 1/(1+x²) over (0,∞) = π/2', () => {
  // weight body: 1/(1+x*x). Build via the sampler expr IR the engine uses.
  const body = { kind: 'call', op: 'divide', args: [
    { kind: 'lit', value: 1 },
    { kind: 'call', op: 'add', args: [{ kind: 'lit', value: 1 },
      { kind: 'call', op: 'mul', args: [{ kind: 'ref', name: 'x' }, { kind: 'ref', name: 'x' }] }] },
  ] };
  const axes = [{ lo: 0, hi: Infinity, kind: 'semi-lo' }];
  const integ = makeIntegrandND(body, ['x'], axes, {}, {});
  const r = adaptiveCubature(integ, 1);
  assert.ok(Math.abs(r.Z - Math.PI / 2) < 1e-6, `Z=${r.Z} vs ${Math.PI / 2}`);
});

// =====================================================================
// resolveTruncateNormalizers, wired to the N-D path (Task 4) — oracle
// integration tests through the REAL surface→lowering pipeline.
//
// GROUNDING (probed against the actual lowered `D.ir`, not assumed):
//  - Anonymous-binding hoisting splits `normalize(truncate(weighted(...),
//    S), S)` across separate `__anonN` bindings, linked by `self` refs — the
//    top-level `D.ir` is just `normalize(<ref>)`. The real scoring path
//    (matLogdensityof → clm.lowerMeasure) fully inlines these via
//    materialiser-shared's `inlineBoundaryDerivations` before calling
//    `resolveTruncateNormalizers`; the test helper below does the same so it
//    exercises the resolver against the same fully-inlined shape production
//    code sees.
//  - `functionof.params` at the lowered stage is NOT the surface names —
//    it's placeholders (`['_x_', '_y_']`); the surface names live in
//    `paramKwargs` (`['x', 'y']`). `weightedBaseWeightFn`/`makeIntegrandND`
//    are already param-name-agnostic (they thread whatever's in
//    `fn.params` through to `evaluateExpr`'s ns-agnostic ref resolution),
//    so this needed no code change — noted here since the brief assumed
//    `['x','y']` literally.
//  - The spec §03 constant `inf` lowers to `{kind:'const', name:'inf'}`,
//    NOT a numeric `lit` — `interval(0.0, inf)` (O3's semi-infinite bound)
//    would otherwise be silently treated as "not a literal interval" and
//    deferred rather than recognised as a semi-lo axis. Fixed in
//    `numericBoundValue`/`truncateSetBounds` (mat-density.ts) to resolve a
//    `const` bound via the sampler's `resolveConst`, not just a `lit`.
// `op`/`args` names for `weighted`/`Lebesgue`/`cartprod`/`interval` all
// matched the brief's assumption exactly.
function resolvedNegLogZ(src: string) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const ctx = {
    bindings: built.bindings, derivations: built.derivations,
    fixedValues: built.fixedValues || new Map(),
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
  };
  const D = built.bindings.get('D');
  // Inline the anon-binding refs first (mirrors clm.lowerMeasure's own
  // prep) so resolveTruncateNormalizers's node-shape match on
  // `normalize(truncate(...))` actually sees the truncate inline rather
  // than through a `self` ref the walker doesn't dereference.
  const inlined = shared.inlineBoundaryDerivations(D.ir, new Set(), ctx);
  const ir = JSON.parse(JSON.stringify(inlined));
  matDensity.resolveTruncateNormalizers(ir, {}, ctx);
  // find the rewritten node: op became 'logweighted', args[0].value = -logZ
  let neg: number | null = null;
  const walk = (n: any) => {
    if (!n || typeof n !== 'object') return;
    if (n.kind === 'call' && n.op === 'logweighted' && n.args && n.args[0] && n.args[0].kind === 'lit')
      { neg = n.args[0].value; return; }
    for (const k in n) walk(n[k]);
  };
  walk(ir);
  return neg;
}

test('O1 separable 2-D normalizer matches scipy oracle', () => {
  const O1 = -0.2245614818793499;
  const src = `
D = normalize(truncate(weighted((x, y) -> exp(0.0 - 0.5*(x*x + y*y)) / (2.0*pi),
    Lebesgue(cartprod(interval(-2.0, 3.0), interval(-1.0, 2.0)))),
    cartprod(interval(-2.0, 3.0), interval(-1.0, 2.0))))`;
  const neg = resolvedNegLogZ(src);
  assert.ok(neg != null && Math.abs(-neg - O1) < 1e-5, `logZ=${-(neg!)} vs ${O1}`);
});

test('O2b coupled+kink 2-D normalizer matches scipy oracle', () => {
  const O2 = 1.5844046542638608;
  // N(x; 1.0 - 0.5·√|y|, 0.5) as an explicit weight; box [0,1]×[-4,4]
  const src = `
D = normalize(truncate(weighted((x, y) ->
    exp(0.0 - 0.5*((x - (1.0 - 0.5*sqrt(abs(y))))/0.5)^2) / (0.5*sqrt(2.0*pi)),
    Lebesgue(cartprod(interval(0.0, 1.0), interval(-4.0, 4.0)))),
    cartprod(interval(0.0, 1.0), interval(-4.0, 4.0))))`;
  const neg = resolvedNegLogZ(src);
  assert.ok(neg != null && Math.abs(-neg - O2) < 1e-5, `logZ=${-(neg!)} vs ${O2}`);
});

test('O3 semi-infinite Cauchy normalizer matches oracle (COV, no NaN)', () => {
  const O3 = -0.6931471805599453; // log(1/2)
  const src = `
D = normalize(truncate(weighted(x -> 1.0/(pi*5.0*(1.0 + (x/5.0)^2)),
    Lebesgue(interval(0.0, inf))), interval(0.0, inf)))`;
  const neg = resolvedNegLogZ(src);
  assert.ok(neg != null && Math.abs(-neg - O3) < 1e-5, `logZ=${-(neg!)} vs ${O3}`);
});

test('over-cap 4-D cartprod truncation refuses loudly', () => {
  const src = `
D = normalize(truncate(weighted((a,b,c,d) -> 1.0,
    Lebesgue(cartprod(interval(0.0,1.0),interval(0.0,1.0),interval(0.0,1.0),interval(0.0,1.0)))),
    cartprod(interval(0.0,1.0),interval(0.0,1.0),interval(0.0,1.0),interval(0.0,1.0))))`;
  assert.throws(() => resolvedNegLogZ(src), /dimension 4 exceeds .*cap.*3/i);
});

test('Z=0 over the region refuses (everywhere-zero weight)', () => {
  const src = `
D = normalize(truncate(weighted((x,y) -> 0.0,
    Lebesgue(cartprod(interval(0.0,1.0), interval(0.0,1.0)))),
    cartprod(interval(0.0,1.0), interval(0.0,1.0))))`;
  assert.throws(() => resolvedNegLogZ(src), /Z = .*0|normalizer/i);
});

test('unbounded truncation of a non-Normal scalar measure refuses (no silent NaN)', () => {
  // Defense-in-depth for the 1-D scalar-reference fast path: interval(2, inf)
  // resolves to a semi-lo axis (numericBoundValue recognises the `inf` const),
  // and that path has no change-of-variables, so a non-Normal kernel over an
  // unbounded set must THROW rather than compute dx=inf → logweighted(NaN).
  // Built as a plain IR (no massFrom) so it reaches resolveTruncateNormalizers'
  // truncate branch directly — in a full derivation context such a node is
  // routed to the mass resolver instead, so this exercises the guard in
  // isolation (a future massFrom-absent caller is exactly the risk it covers).
  const ir = {
    kind: 'call', op: 'normalize', args: [
      { kind: 'call', op: 'truncate', args: [
        { kind: 'call', op: 'Exponential', args: [], kwargs: { rate: lit(1) } },
        { kind: 'call', op: 'interval', args: [lit(2), { kind: 'const', name: 'inf' }] },
      ] },
    ],
  };
  assert.throws(() => matDensity.resolveTruncateNormalizers(ir, {}, { bindings: new Map() }),
    /unbounded truncation|has no deterministic quadrature|NaN/i);
});
