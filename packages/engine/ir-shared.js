'use strict';

// ir-shared.js — leaf IR utilities shared across the orchestrator
// decomposition (constant folding, IR→value resolution, self-ref
// collection, set parsing). The dependency ROOT of the split: depends
// only on lower/analyzer (+ a lazy sampler require for the general
// deterministic evaluator); NOTHING here requires lift/derivations/
// signatures/profile-plan/orchestrator, so it breaks all module
// cycles. orchestrator.js (and later derivations/profile-plan)
// re-bind these names from here; the public API is unchanged.

const { lowerExpr } = require('./lower');
const { isMeasureExpr } = require('./analyzer');

/**
 * Resolve a measure-typed AST argument (the measure operand of
 * weighted / normalize / superpose, etc.) to a binding name we can
 * alias the new derivation to. Returns null when the argument isn't a
 * shape we currently support.
 *
 * Accepts:
 *   - Identifier(<name>) where <name>'s binding is a measure (per
 *     spec §sec:measure-algebra; uses isMeasureExpr to be robust to
 *     lawof bindings, alias chains, distribution constructors,
 *     weighted/normalize/superpose results, etc.)
 *   - CallExpr `lawof(<ident>)` — the spec's identity law gives
 *     `lawof(draw(m)) = m`, and our empirical-measure cache treats
 *     a variate and its underlying measure as the same atoms +
 *     weights, so we alias to the inner ident's cached measure
 *     directly. This is the spec-correct way to lift a value into a
 *     measure on the fly.
 *
 * Inline measure constructions (e.g. `weighted(0.5, Normal(0, 1))`,
 * or chains like `weighted(0.5, normalize(m))`) need anonymous
 * intermediate derivations and are deferred — the user can split
 * them into named bindings for now.
 */
function resolveMeasureBaseName(astNode, bindings) {
  if (!astNode) return null;
  if (astNode.type === 'Identifier' && bindings.has(astNode.name)) {
    return isMeasureExpr(astNode, bindings) ? astNode.name : null;
  }
  if (astNode.type === 'CallExpr'
      && astNode.callee && astNode.callee.type === 'Identifier'
      && astNode.callee.name === 'lawof'
      && Array.isArray(astNode.args) && astNode.args.length === 1) {
    const inner = astNode.args[0];
    if (inner && inner.type === 'Identifier' && bindings.has(inner.name)) {
      return inner.name;
    }
  }
  return null;
}

/**
 * Resolve an IR node to a constant numeric value, or null if it
 * doesn't reduce. Handles literal numerics, the named built-in
 * constants the evaluator knows, and refs to bindings whose RHS
 * itself reduces to a constant. Used by `weighted` / `logweighted`
 * derivations to pre-compute the log-shift at classification time
 * rather than at sample-render time. Cycle-guarded.
 */
function resolveConstant(ir, bindings, seen) {
  if (!ir) return null;
  if (ir.kind === 'lit') {
    if (typeof ir.value === 'number' && Number.isFinite(ir.value)) return ir.value;
    return null;
  }
  if (ir.kind === 'const') {
    if (ir.name === 'pi')  return Math.PI;
    if (ir.name === 'e')   return Math.E;
    if (ir.name === 'inf') return Infinity;
    return null;
  }
  if (ir.kind === 'ref' && ir.ns === 'self') {
    if (seen.has(ir.name)) return null;
    seen.add(ir.name);
    const b = bindings.get(ir.name);
    if (!b || !b.node || !b.node.value) return null;
    let bIR;
    try { bIR = lowerExpr(b.node.value); } catch (_) { return null; }
    return resolveConstant(bIR, bindings, seen);
  }
  // Constant-fold small arithmetic. Crucially, the parser lowers a
  // negative literal `-3.5` to `(call neg (lit 3.5))`, so without this
  // we'd fail to recognise plain negative numbers as constants. The
  // operator set matches EVALUABLE_OPS so the language's evaluator
  // semantics agree at this level.
  if (ir.kind === 'call' && ir.op && Array.isArray(ir.args)) {
    const args = ir.args.map(a => resolveConstant(a, bindings, seen));
    if (args.some(v => v == null)) return null;
    switch (ir.op) {
      case 'neg': return args.length === 1 ? -args[0] : null;
      case 'pos': return args.length === 1 ?  args[0] : null;
      case 'add': return args.length === 2 ? args[0] + args[1] : null;
      case 'sub': return args.length === 2 ? args[0] - args[1] : null;
      case 'mul': return args.length === 2 ? args[0] * args[1] : null;
      case 'div': return args.length === 2 ? args[0] / args[1] : null;
      default: return null;
    }
  }
  return null;
}

function isCallOp(ir, op, expectedArgCount) {
  if (!ir || ir.kind !== 'call' || ir.op !== op || !Array.isArray(ir.args)) return false;
  if (expectedArgCount !== null && ir.args.length !== expectedArgCount) return false;
  return true;
}

function isSelfRef(ir) {
  return !!ir && ir.kind === 'ref' && ir.ns === 'self';
}

/**
 * Convert a lowered IR expression to a concrete JS value (number,
 * array of values, plain object). Used by the viewer's bayesupdate /
 * logdensityof / likelihood materialisers to translate a recorded
 * `obsIR` (the AST shape of `observed_data` or `record(obs = ...)`,
 * etc.) into the JS value traceeval clamps against at sample time.
 *
 * Resolution order on `ref self <name>`:
 *   1. fixedValues, if supplied — pre-eval may have materialised a
 *      dynamically-computed value (rand result, tuple_get, etc.) we
 *      cannot reach through static IR recursion.
 *   2. Recursive walk into the referenced binding's lowered IR.
 *
 * Static fast paths (no sampler dependency, exact-shaped JS output):
 *   - { kind: 'lit', value: <number> }    → number
 *   - { kind: 'call', op: 'vector', args }→ array of resolved elements
 *   - { kind: 'call', op: 'record',
 *       fields: [{name, value}, ...] }    → plain object keyed by field
 *   - { kind: 'call', op: 'neg', args }   → negative
 *   - { kind: 'ref', ns: 'self', name }   → resolve as above
 *
 * Anything else (general deterministic expressions: `broadcast`,
 * arithmetic, `get`, nested compositions — e.g. kernel-broadcast
 * distribution parameters like `Gamma.(tau .+ 1.0, tau)`) is delegated
 * to the engine's real deterministic evaluator `sampler.evaluateExpr`,
 * with this resolver's own ref machinery (fixedValues → recursive
 * binding walk → cycle guard) supplied as the evaluator's env. This is
 * the same evaluator the orchestrator's fixed-phase pre-eval and the
 * worker use, so every value-position resolver in the engine agrees on
 * deterministic semantics. The fallback is strictly additive — inputs
 * that hit a static fast path are byte-for-byte unchanged; only inputs
 * that previously threw `unsupported op/kind` now resolve.
 *
 * Still throws on genuine errors (not an IR node, dependency cycles,
 * an expression the deterministic evaluator itself rejects). The
 * thrown message names the failure so the viewer can surface it
 * directly as a plot-time error rather than a silent failure.
 */
function resolveIRToValue(ir, bindings, fixedValues) {
  return walk(ir, new Set());
  function walk(ir, seen) {
    if (!ir || typeof ir !== 'object') {
      throw new Error('resolveIRToValue: not an IR node');
    }
    if (ir.kind === 'lit' && typeof ir.value === 'number') return ir.value;
    if (ir.kind === 'ref' && ir.ns === 'self') {
      if (fixedValues && fixedValues.has(ir.name)) return fixedValues.get(ir.name);
      if (seen.has(ir.name)) {
        throw new Error(`resolveIRToValue: cycle through '${ir.name}'`);
      }
      const b = bindings && bindings.get(ir.name);
      if (!b || !b.ir) {
        throw new Error(`resolveIRToValue: no IR for '${ir.name}'`);
      }
      const next = new Set(seen); next.add(ir.name);
      return walk(b.ir, next);
    }
    if (ir.kind === 'call') {
      if (ir.op === 'vector' && Array.isArray(ir.args)) {
        const out = new Array(ir.args.length);
        for (let i = 0; i < ir.args.length; i++) out[i] = walk(ir.args[i], seen);
        return out;
      }
      if (ir.op === 'record' && Array.isArray(ir.fields)) {
        const out = {};
        for (const f of ir.fields) out[f.name] = walk(f.value, seen);
        return out;
      }
      if (ir.op === 'neg' && Array.isArray(ir.args) && ir.args.length === 1) {
        return -walk(ir.args[0], seen);
      }
      // Fall through to the general deterministic evaluator.
    }
    // General case: delegate to the engine's real deterministic
    // evaluator. `sampler.evaluateExpr(ir, env)` understands the full
    // value language (broadcast, arithmetic, get, functionof bodies,
    // …); we only need to answer its top-level `ref self`/`%local`
    // lookups, which we route back through `walk` so fixedValues, the
    // recursive binding walk, and cycle detection above are reused
    // verbatim. A Proxy gives evaluateExpr's `name in env` / `env[name]`
    // access without eagerly snapshotting the (possibly large)
    // fixedValues map.
    const env = new Proxy(Object.create(null), {
      has(_t, name) {
        if (typeof name !== 'string') return false;
        if (fixedValues && fixedValues.has(name)) return true;
        return !!(bindings && bindings.get(name) && bindings.get(name).ir);
      },
      get(_t, name) {
        if (typeof name !== 'string') return undefined;
        return walk({ kind: 'ref', ns: 'self', name }, seen);
      },
    });
    const samplerLib = require('./sampler');
    return valueToPlain(samplerLib.evaluateExpr(ir, env));
  }
}

/**
 * Normalize a `sampler.evaluateExpr` result back to `resolveIRToValue`'s
 * documented JS contract (number | nested array | plain object). The
 * evaluator returns plain numbers/arrays for most expressions but a
 * shape-tagged Value for some vector/matrix ops; callers of
 * resolveIRToValue (obsIR clamping, MvNormal mu/cov, kernel-broadcast
 * params) expect plain JS / will re-`asValue` it, so collapse Values to
 * row-major nested arrays here. Booleans pass through as-is (the array
 * caller coerces true/false → 1/0, matching the rest of the engine).
 */
function valueToPlain(v) {
  if (v == null || typeof v === 'number' || typeof v === 'boolean') return v;
  if (Array.isArray(v)) return v.map(valueToPlain);
  // Shape-tagged Value → nested JS array (row-major), scalar → number.
  if (typeof v === 'object' && Array.isArray(v.shape)
      && v.data instanceof Float64Array) {
    const data = v.data;
    const build = (axis, offset, stride) => {
      if (axis === v.shape.length) return data[offset];
      const n = v.shape[axis];
      const inner = stride / n;
      const out = new Array(n);
      for (let i = 0; i < n; i++) out[i] = build(axis + 1, offset + i * inner, inner);
      return out;
    };
    const total = v.shape.reduce((a, b) => a * b, 1);
    return v.shape.length === 0 ? data[0] : build(0, 0, total);
  }
  return v;
}

/**
 * Collect the names of every (ref self <name>) inside an IR subtree.
 * Used by the worker / main thread to gather upstream sample arrays
 * before drawing or evaluating. Doesn't follow into reified scopes —
 * those introduce their own scope and their bodies aren't part of the
 * outer binding's data dependencies for sampling.
 */
function collectSelfRefs(ir) {
  const seen = new Set();
  walk(ir);
  return seen;
  function walk(node) {
    if (!node || typeof node !== 'object') return;
    if (node.kind === 'ref' && node.ns === 'self') seen.add(node.name);
    if (node.args)   for (const a of node.args)            walk(a);
    if (node.kwargs) for (const k in node.kwargs)          walk(node.kwargs[k]);
    // joint/record IRs use `fields: [{ name, value }, ...]` instead
    // of args/kwargs. Walk values so refs inside joint fields don't
    // get missed.
    if (Array.isArray(node.fields)) for (const f of node.fields) walk(f && f.value);
    if (node.body)                                          walk(node.body);
    // Reified-scope params/paramKwargs are name lists, not IRs.
  }
}

function lowerSafe(ast) {
  try { return lowerExpr(ast); } catch (_) { return null; }
}

const NAMED_SETS = {
  reals:           { kind: 'reals' },
  posreals:        { kind: 'posreals' },
  nonnegreals:     { kind: 'nonnegreals' },
  unitinterval:    { kind: 'interval', lo: 0, hi: 1 },
  integers:        { kind: 'integers' },
  posintegers:     { kind: 'posintegers' },
  nonnegintegers:  { kind: 'nonnegintegers' },
  booleans:        { kind: 'booleans' },
};

function parseSetIR(setIR, bindings) {
  if (!setIR) return null;
  if (setIR.kind === 'const' && NAMED_SETS[setIR.name])
    return NAMED_SETS[setIR.name];
  if (setIR.kind === 'ref' && setIR.ns === 'self' && NAMED_SETS[setIR.name])
    return NAMED_SETS[setIR.name];
  if (setIR.kind === 'call' && setIR.op === 'interval'
      && Array.isArray(setIR.args) && setIR.args.length === 2) {
    // Bounds resolve via the same constant-folder used elsewhere — so
    // `interval(-2.0, 2.0)` (which lowers `-2.0` to `(neg (lit 2.0))`),
    // `interval(0, inf)`, and `interval(LO, HI)` with constant-bound
    // refs all reduce to numeric bounds.
    const seen = new Set();
    const lo = resolveConstant(setIR.args[0], bindings || new Map(), seen);
    const hi = resolveConstant(setIR.args[1], bindings || new Map(), new Set());
    if (typeof lo === 'number' && typeof hi === 'number') {
      return { kind: 'interval', lo, hi };
    }
  }
  return null;
}

const NAMED_SET_NAMES = new Set([
  'reals', 'posreals', 'nonnegreals', 'unitinterval',
  'integers', 'posintegers', 'nonnegintegers', 'booleans',
]);

module.exports = {
  resolveMeasureBaseName,
  resolveConstant,
  isCallOp,
  isSelfRef,
  resolveIRToValue,
  valueToPlain,
  collectSelfRefs,
  lowerSafe,
  NAMED_SETS,
  parseSetIR,
  NAMED_SET_NAMES,
};
