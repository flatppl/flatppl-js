'use strict';

// ir-shared.js — leaf IR utilities shared across the orchestrator
// decomposition (constant folding, IR→value resolution, self-ref
// collection, set parsing). The dependency ROOT of the split: depends
// only on lower/analyzer (+ a lazy sampler require for the general
// deterministic evaluator); NOTHING here requires lift/derivations/
// signatures/profile-plan/orchestrator, so it breaks all module
// cycles. orchestrator.js (and later derivations/profile-plan)
// re-bind these names from here; the public API is unchanged.

import type { IRNode } from './engine-types';

const { lowerExpr } = require('./lower.ts');
const { isMeasureExpr } = require('./analyzer.ts');

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
function resolveMeasureBaseName(astNode: any, bindings: any) {
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
function resolveConstant(ir: any, bindings: any, seen: Set<any>): any {
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
    const args = ir.args.map((a: any) => resolveConstant(a, bindings, seen));
    if (args.some((v: any) => v == null)) return null;
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

function isCallOp(ir: IRNode | null | undefined, op: string, expectedArgCount: number | null) {
  if (!ir || ir.kind !== 'call' || ir.op !== op || !Array.isArray(ir.args)) return false;
  if (expectedArgCount !== null && ir.args.length !== expectedArgCount) return false;
  return true;
}

function isSelfRef(ir: IRNode | null | undefined) {
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
function resolveIRToValue(ir: any, bindings: any, fixedValues: any) {
  return walk(ir, new Set());
  function walk(ir: any, seen: Set<any>): any {
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
        const out: Record<string, any> = {};
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
    const samplerLib = require('./sampler.ts');
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
function valueToPlain(v: any): any {
  if (v == null || typeof v === 'number' || typeof v === 'boolean') return v;
  if (Array.isArray(v)) return v.map(valueToPlain);
  // Shape-tagged Value → nested JS array (row-major), scalar → number.
  if (typeof v === 'object' && Array.isArray(v.shape)
      && v.data instanceof Float64Array) {
    const data = v.data;
    const build = (axis: number, offset: number, stride: number): any => {
      if (axis === v.shape.length) return data[offset];
      const n = v.shape[axis];
      const inner = stride / n;
      const out = new Array(n);
      for (let i = 0; i < n; i++) out[i] = build(axis + 1, offset + i * inner, inner);
      return out;
    };
    const total = v.shape.reduce((a: number, b: number) => a * b, 1);
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
function collectSelfRefs(ir: IRNode | null | undefined) {
  const seen = new Set<string>();
  walk(ir);
  return seen;
  function walk(node: IRNode | null | undefined) {
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

function lowerSafe(ast: any): IRNode | null {
  try { return lowerExpr(ast); } catch (_) { return null; }
}

const NAMED_SETS: Record<string, any> = {
  reals:           { kind: 'reals' },
  posreals:        { kind: 'posreals' },
  nonnegreals:     { kind: 'nonnegreals' },
  unitinterval:    { kind: 'interval', lo: 0, hi: 1 },
  integers:        { kind: 'integers' },
  posintegers:     { kind: 'posintegers' },
  nonnegintegers:  { kind: 'nonnegintegers' },
  booleans:        { kind: 'booleans' },
};

function parseSetIR(setIR: any, bindings: any) {
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

// =====================================================================
// Static gates. Pure constant catalogues with zero dependencies — they
// live in the leaf so every split module (lift's evaluability check,
// derivations' discreteness/sampleability checks, the orchestrator
// core) shares one authority. orchestrator.js / derivations.js re-bind
// these via the facade.
// =====================================================================

// Distributions the worker's REGISTRY currently implements. Hardcoded
// here to avoid pulling sampler.js (and stdlib) into the main bundle.
// Mirrored in sampler.js's REGISTRY; if you add one there, add it here
// too. The orchestrator gates on this list — if a binding's RHS is a
// distribution we don't list, the chain comes back unsupported instead
// of failing later in the worker.
const SAMPLEABLE_DISTRIBUTIONS = new Set([
  'Normal', 'Exponential', 'Uniform', 'Logistic', 'Weibull',
  'LogNormal', 'Beta', 'Gamma', 'InverseGamma',
  'GeneralizedNormal', 'ChiSquared', 'VonMises', 'Laplace',
  'Cauchy', 'StudentT', 'Bernoulli', 'Binomial', 'Poisson',
  'Geometric', 'NegativeBinomial', 'NegativeBinomial2',
  'Categorical', 'Categorical0',
  // Dirac is degenerate (zero entropy): the sampler emits the
  // 'value' kwarg verbatim N times. Listed here so measure-alias
  // bindings like `m = Dirac(value = 5)` get classified 'skip' and
  // resolved to a sample step at the target rather than failing
  // SAMPLEABLE_DISTRIBUTIONS gate. Identity rewrite for
  // `draw(Dirac(value=e))` lives in classifyForChain — at the
  // draw site we re-route to evaluate(e) rather than sampling.
  'Dirac',
]);

// Subset of the above whose density is over the counting reference (a
// pmf, integer atoms). Used by the worker to switch between KDE and
// integer-histogram density estimation.
const DISCRETE_DISTRIBUTIONS = new Set([
  'Bernoulli', 'Binomial', 'Poisson',
  'Geometric', 'NegativeBinomial', 'NegativeBinomial2',
  'Categorical', 'Categorical0',
]);

// Deterministic builtins whose call IRs the worker's evaluateExpr knows
// how to compute. Mirrors the operator desugaring in lower.js plus a
// small catalogue of safe scalar functions. Anything else lowered as
// `(call <op> ...)` is treated as unsupported, so a stray `joint(...)`
// or `disintegrate(...)` doesn't silently get scheduled.
//
// Keep in sync with sampler.js's evaluateExpr handler set. When the
// evaluator gains a new builtin (e.g. `pow`), add it both there and
// here.
const EVALUABLE_OPS = new Set([
  // Operator desugaring (BIN_OP_MAP / UN_OP_MAP in lower.js).
  // This list mirrors sampler.js's ARITH_OPS exactly. Extend both
  // sides together when adding ops (the static gate must match the
  // worker's evaluator).
  'add', 'sub', 'mul', 'div', 'divide', 'mod', 'neg', 'pos',
  'identity',
  'abs', 'abs2', 'exp', 'log', 'log10', 'log1p', 'expm1', 'sqrt',
  'sin', 'cos', 'tan',
  'asin', 'acos', 'atan', 'atan2',
  'sinh', 'cosh', 'tanh',
  'asinh', 'acosh', 'atanh',
  'floor', 'ceil', 'round',
  'pow',
  // Complex arithmetic (spec §03 / §07): constructor + accessors.
  'complex', 'real', 'imag', 'conj', 'cis',
  // Binary min/max and gamma/loggamma/link functions (spec §07
  // Elementary functions). All scalar→scalar (or scalar,scalar→scalar)
  // and dispatch through sampler.ARITH_OPS.
  'min', 'max',
  'gamma', 'loggamma',
  'logit', 'invlogit', 'probit', 'invprobit',
  // Comparisons → boolean.
  'lt', 'le', 'gt', 'ge', 'equal', 'unequal',
  // Predicates → boolean.
  'isfinite', 'isinf', 'isnan', 'iszero',
  // Logic / conditionals.
  'land', 'lor', 'lxor', 'lnot', 'ifelse',
  // Reductions over arrays (sampler.js implements the runtime ops). The
  // static gate for these is conservative: only mark the binding
  // evaluable when the operand is a static array (kind: 'array'
  // derivation) — handled by the array-evaluable check downstream.
  // Generic ref-to-stochastic-array isn't evaluable in the per-i
  // worker model since each atom's value would itself be an array.
  // Note: 'vector' deliberately omitted — leaves like `[mu, 1.0]`
  // (with stochastic refs) must NOT classify as evaluable, so the
  // existing array-derivation special case keeps owning that path.
  'sum', 'mean', 'prod', 'lengthof', 'sizeof', 'maximum', 'minimum', 'var', 'std',
  'cumsum', 'cumprod',
  // Norms and softmax family (spec §07). All single-arg, vector
  // input; *unit / softmax / logsoftmax return vectors, the rest
  // return scalars.
  'l1norm', 'l2norm', 'l1unit', 'l2unit',
  'logsumexp', 'softmax', 'logsoftmax',
  // Engine-internal projection emitted by the analyzer's multi-LHS
  // rewriter (`a, b = rand(...)`). sampler.evaluateCall handles it.
  'tuple_get', 'tuple',
  // Field access: lowered from surface `obj.field` and from `record(
  // a=x, b=y)` constructors. Both are pure value computations the
  // evaluator handles.
  'get_field', 'record',
  // Unified element/subset/slice access (spec §07). `v[i]`, `A[i,j]`,
  // `A[:,j]`, `v[[1,3]]`, `get(r,"a")` all lower to `get` (1-based);
  // `get0` is the 0-based variant. Pure deterministic value ops —
  // sampler.evaluateCall dispatches both via a dedicated case. Added
  // so a fixed-phase expression containing indexing (e.g.
  // `Gamma(shape = tau[1] + 1.0, …)`) evaluates through the single
  // deterministic-evaluator authority instead of dead-ending.
  'get', 'get0',
  // Random-number primitives (spec §sec:random). All three are
  // ordinary value-typed functions whose phase propagates from
  // their inputs. sampler.evaluateCall dispatches each.
  'rnginit', 'rngstate', 'rand',
  // Shape functions (spec §07 Approximation functions). Pure value
  // ops; kwargs-shaped so they don't fit ARITH_OPS — sampler.evaluateCall
  // dispatches each via a dedicated case.
  'polynomial', 'bernstein', 'stepwise',
  // Binning (spec §07). bincounts produces an integer count array
  // from edges + data; kwargs-shaped, sampler.evaluateCall dispatches.
  // selectbins keeps whole bins whose interval intersects a region.
  'bincounts', 'selectbins',
  // Array generation (spec §07). All pure value ops over fixed-phase
  // arguments; dispatch through ARITH_OPS.
  'linspace', 'extlinspace', 'partition', 'reverse', 'addaxes',
  'fill', 'zeros', 'ones', 'eye', 'onehot',
  'rowstack', 'colstack', 'array',
  // Higher-order ops (spec §04 / §07). Dispatched via dedicated cases
  // in sampler.evaluateCall (not ARITH_OPS) because they evaluate a
  // referenced function's body per element. filter takes a unary
  // predicate; reduce / scan take binary accumulators; broadcast
  // takes an n-ary function and n equal-length arrays.
  'filter', 'reduce', 'scan', 'broadcast',
  // Scalar restrictors (spec §07).
  'boolean', 'integer',
  // Linear algebra (spec §07). All pure value ops on matrices /
  // vectors; dispatch through ARITH_OPS.
  'transpose', 'adjoint', 'trace', 'diagmat', 'self_outer',
  'det', 'logabsdet', 'inv', 'linsolve', 'lower_cholesky',
  'row_gram', 'col_gram',
  // Multi-axis aggregation (spec §04 §sec:aggregate). Pure
  // deterministic value op — phase joins its inputs. The materialiser
  // dispatches through a pattern table with a nested-loop interpreter
  // as fallback; both surface here through the standard `evaluate`
  // derivation.
  'aggregate',
]);

// =====================================================================
// Measure-IR canonicalisation. Pure (IR shape + binding.phase),
// zero-dependency, and shared by the orchestrator core and the
// derivation classifier — lives in the leaf so there's a single
// authority and no core<->derivations back-edge.
// =====================================================================

/**
 * Canonicalise measure-construction IRs so downstream classification,
 * sampling, and the viewer's plot dispatch all see a single normalized
 * shape per equivalence class. Pure: input IR is not mutated.
 *
 *   lawof(e)              ≡ Dirac(value = e)   ONLY when e is fixed-phase.
 *      (For deterministic e the law is a point mass at e. For
 *      stochastic e — e.g. `lawof(draw(m))` — the spec identity is
 *      lawof(draw(m)) ≡ m, NOT Dirac(value=draw_result), so we skip
 *      the rewrite. Spec §sec:variate-measure + §sec:lawof.)
 *
 *   Dirac(e)              ≡ Dirac(value = e)
 *      (Positional argument bound to the kwarg name per spec
 *      §sec:calling-convention: built-in callables accept both
 *      positional and keyword forms, with identical semantics.
 *      Purely syntactic — no phase check needed.)
 *
 * Applied at every entry point that classifies measure IRs
 * (classifyForChain, resolveMeasure, target-promotion,
 * classifyDerivation). After this point, fixed-phase lawof and
 * positional-Dirac don't appear as distinct measure surface forms —
 * only Dirac(value=...) remains, and the Dirac sampler / viewer text
 * path handles it uniformly.
 *
 * @param ir       IR node to (possibly) rewrite.
 * @param bindings Optional bindings map. When supplied, enables the
 *                 lawof rewrite by letting us check the phase of a
 *                 ref-arg. Without it, lawof passes through unchanged.
 */
function normalizeMeasureIR(ir: IRNode | null | undefined, bindings: any): IRNode | null | undefined {
  if (!ir || ir.kind !== 'call') return ir;
  if (ir.op === 'lawof'
      && Array.isArray(ir.args) && ir.args.length === 1
      && (!ir.kwargs || Object.keys(ir.kwargs).length === 0)) {
    if (isFixedPhaseValueIR(ir.args[0], bindings)) {
      return { kind: 'call', op: 'Dirac',
               kwargs: { value: ir.args[0] }, loc: ir.loc };
    }
  }
  if (ir.op === 'Dirac'
      && (!ir.kwargs || !Object.prototype.hasOwnProperty.call(ir.kwargs, 'value'))
      && Array.isArray(ir.args) && ir.args.length === 1) {
    return { kind: 'call', op: 'Dirac',
             kwargs: { value: ir.args[0] }, loc: ir.loc };
  }
  return ir;
}

// Conservative "this IR denotes a deterministic value" predicate used
// by normalizeMeasureIR's lawof rewrite. Literals and named constants
// are always fixed; refs are fixed iff they point at a binding with
// phase='fixed'. Anything else (calls, missing bindings) returns
// false — the rewrite skips them and the lawof stays in its original
// form for downstream phase-aware dispatch.
function isFixedPhaseValueIR(ir: any, bindings: any) {
  if (!ir) return false;
  if (ir.kind === 'lit' || ir.kind === 'const') return true;
  if (ir.kind === 'ref' && ir.ns === 'self' && bindings) {
    const b = bindings.get(ir.name);
    return !!(b && b.phase === 'fixed');
  }
  return false;
}

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
  normalizeMeasureIR,
  isFixedPhaseValueIR,
  SAMPLEABLE_DISTRIBUTIONS,
  DISCRETE_DISTRIBUTIONS,
  EVALUABLE_OPS,
};
