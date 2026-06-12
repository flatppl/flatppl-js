'use strict';

// FlatPIR structural type inference.
//
// Operates on the LoweredModule produced by `pir.lowerToModule(...)`.
// Walks each binding's lowered RHS, infers a type for every call,
// writes per-call meta annotations (FlatPIR `(%meta type phase)`,
// type slot only — phase is in phaseinfer.js), and sets
// `binding.inferredType` to the type of the binding's outermost
// expression for fast lookup.
//
// Diagnostics are collected into a flat array compatible with the
// analyzer's existing diagnostic stream (same {severity, message,
// loc} shape) so they merge cleanly into the editor.
//
// Why on lowered IR
// =================
// The source AST has many node kinds (BinaryExpr, UnaryExpr,
// ArrayLiteral, TupleLiteral, FieldAccess, etc.). FlatPIR collapses
// them all to calls — `add`, `mul`, `vector`, `tuple`, `get_field`,
// etc. The inference pass over the IR is therefore one switch on
// {lit, const, ref, hole, call}, with the call case dispatching by
// op name. Cleaner; fewer special cases.
//
// Polymorphism
// ============
// Built-in signatures use type variables (`weighted: (real,
// measure<T>) → measure<T>`); types.js's unify handles them.
//
// User-defined function/kernel signatures carry their result type
// directly (computed at definition time by inferring the body in the
// scope where parameters take their declared types). For now we
// don't recompute the body's type per call site — that polymorphic
// flow is in the FlatPIR spec but unused in practice for the
// visualizer's current scope. Added when needed.
//
// Scopes
// ======
// `functionof(body, kw=...)` and `kernelof(body, kw=...)` introduce
// an inner `%local` scope. Inside their bodies, parameter refs are
// `(%ref %local <name>)`. The inference pass tracks an active scope
// stack: a Map<paramName, type> for each enclosing reified callable.
// %local refs resolve against this stack; %self refs against the
// module's binding map.

import type { IRNode } from './engine-types';

const T = require('./types.ts');
const builtins = require('./builtins.ts');
const aggregateShape = require('./aggregate-shape.ts');

// =====================================================================
// Constant maps (carried over from the AST-based version)
// =====================================================================

const CONST_TYPES: Record<string, any> = {
  pi:    T.REAL,
  inf:   T.REAL,
  im:    T.COMPLEX,
  true:  T.BOOLEAN,
  false: T.BOOLEAN,
};

const SET_VALUE_TYPES: Record<string, any> = {
  reals: T.REAL, posreals: T.REAL, nonnegreals: T.REAL, unitinterval: T.REAL,
  integers: T.INTEGER, posintegers: T.INTEGER, nonnegintegers: T.INTEGER,
  booleans: T.BOOLEAN,
  complexes: T.COMPLEX,
  rngstates: T.any(),
  anything: T.any(),
};

// Op classifications for shape-polymorphic inference. Binary ops
// take two numeric operands and return a numeric of the broadcast
// shape (scalar/scalar → scalar; scalar+array → array; array+array
// of matching shape → array). Unary ops take one numeric and
// return the same shape. Comparisons take two numerics and return
// boolean of the broadcast shape (so vec_a < vec_b yields
// array<bool>; scalar_a < vec_b yields array<bool> too).
const BINARY_ARITH_OPS = new Set(['add', 'sub', 'mul', 'div', 'mod', 'pow']);
const UNARY_ARITH_OPS  = new Set([
  'neg', 'pos', 'abs', 'abs2', 'exp', 'log', 'log10', 'sqrt',
  'sin', 'cos', 'floor', 'ceil', 'round',
]);
const COMPARISON_OPS = new Set(['lt', 'le', 'gt', 'ge', 'equal', 'unequal']);

// =====================================================================
// Public entry
// =====================================================================

/**
 * Run type inference over a LoweredModule. Mutates each binding to
 * set `binding.inferredType` and writes per-call `meta.type`
 * annotations. Returns diagnostics for type mismatches; loc fields
 * point at the source AST positions captured during lowering.
 *
 * `opts.resolveFixed` (optional): a callback the inference pass
 * invokes at shape positions to fold constant expressions to
 * concrete integers — `iid(M, n)` where `n = length(data)` becomes
 * `array([N], elem)` rather than `array([%dynamic], elem)`. Caller
 * builds the resolver via `fixed-eval.makeResolver(...)`; typeinfer
 * stays unaware of the value-mode evaluator. Engine-concepts §17.4
 * "resolve, don't rewrite" — the resolver is consulted only at
 * narrowly-identified shape positions, and only the resulting
 * integer is embedded in type annotations; the source IR is left
 * intact in either case.
 */
function inferTypes(loweredModule: any, opts?: { resolveFixed?: any }) {
  const ctx = createInferenceContext(loweredModule, opts);
  for (const [name] of loweredModule.bindings) ctx.inferBinding(name);
  return ctx.diagnostics;
  // NOTE: no eager post-binding const-eval pass. The resolver is
  // demand-driven (engine-concepts §17.4) — it's invoked only from
  // shape positions and recursively descends into refs ONLY as
  // needed. Shape-observer short-circuits (length/lengthof/sizeof
  // reading from inferredType) prevent the recursion from
  // materialising expensive bindings whose value isn't actually
  // needed for any shape. This is the "query system" / lazy-eval
  // pattern from rustc/salsa, Haskell, Idris/Agda.
}

/**
 * On-demand inference at a synthetic call site: given a body
 * expression and a scope binding param names to concrete input
 * types, return the inferred result type. Used for plot-time
 * specialization of polymorphic functions — the module-level
 * inference produces a best-effort type with `any` inputs, but
 * when we plot we have specific input types and can specialize
 * exactly, the same way a real call site would.
 *
 * Assumes the module has already been processed by inferTypes
 * (each binding has its inferredType set). Lazy inferBinding
 * still works if not — it'll walk on demand.
 *
 * @param loweredModule  the module produced by `lower(ast)`
 * @param expr           an IR expression (typically the body of a
 *                        functionof / kernelof)
 * @param paramTypes     Map<paramName, type>; each %local-ref
 *                        inside `expr` resolves through this.
 * @returns inferred type of `expr` in that scope
 */
function inferExprInScope(loweredModule: any, expr: IRNode, paramTypes: any) {
  const ctx = createInferenceContext(loweredModule);
  const scopes = paramTypes ? [paramTypes] : [];
  return ctx.inferExpr(expr, scopes);
}

/**
 * Build a fresh inference context bound to a LoweredModule. Returns
 * the `diagnostics` accumulator and the same `inferBinding` /
 * `inferExpr` helpers that drive the module-level pass — exposing
 * them lets external callers run inference on synthetic
 * expressions (call sites, sub-bodies) reusing exactly the same
 * rules. Cycle detection (visiting/visited) is per-context, so
 * separate contexts don't interfere.
 */
function createInferenceContext(loweredModule: any, opts?: { resolveFixed?: any }) {
  const diagnostics: any[] = [];
  const visiting = new Set();
  const visited  = new Set();
  const resolveFixed = opts && opts.resolveFixed;

  function inferBinding(name: any): any {
    const b = loweredModule.bindings.get(name);
    if (!b)                   return T.failed('unknown binding "' + name + '"');
    if (visited.has(name))    return b.inferredType || T.deferred();
    if (visiting.has(name)) {
      const t = T.failed('cyclic type inference at "' + name + '"');
      b.inferredType = t;
      return t;
    }
    visiting.add(name);
    const t: any = inferExpr(b.rhs, []);   // [] = no enclosing scopes
    visiting.delete(name);
    visited.add(name);
    b.inferredType = t;
    return t;
  }

  // -------------------------------------------------------------------
  // Expression-level inference
  // -------------------------------------------------------------------
  // `scopes` is a stack of Map<paramName, type> for each enclosing
  // functionof/kernelof. Top of stack is the innermost scope.

  function inferExpr(expr: any, scopes: any): any {
    if (!expr) return T.failed('null expression');
    switch (expr.kind) {
      case 'lit':   return inferLit(expr);
      case 'const': return inferConst(expr);
      case 'ref':   return inferRef(expr, scopes);
      case 'hole':  return T.any();   // bound positionally inside fn(...)
      case 'call':  return inferCall(expr, scopes);
    }
    return T.deferred();
  }

  function inferLit(expr: any) {
    const v = expr.value;
    if (typeof v === 'number') {
      // Lower.js preserves the lexical-form distinction in `numType`
      // (integer literals have no decimal/exponent in source).
      // Fall back to runtime check for synthesized lits without it.
      if (expr.numType === 'integer') return T.INTEGER;
      if (expr.numType === 'real')    return T.REAL;
      return Number.isInteger(v) ? T.INTEGER : T.REAL;
    }
    if (typeof v === 'boolean') return T.BOOLEAN;
    if (typeof v === 'string')  return T.STRING;
    return T.deferred();
  }

  function inferConst(expr: any) {
    if (CONST_TYPES[expr.name])  return CONST_TYPES[expr.name];
    if (builtins.isSet(expr.name)) return setMarker(expr.name);
    return T.any();
  }

  function inferRef(expr: any, scopes: any): any {
    if (expr.ns === '%local') {
      // Look up in the scope stack from innermost outward.
      for (let i = scopes.length - 1; i >= 0; i--) {
        if (scopes[i].has(expr.name)) return scopes[i].get(expr.name);
      }
      return T.failed('unbound %local "' + expr.name + '"');
    }
    if (expr.ns === 'self') {
      if (loweredModule.bindings.has(expr.name)) return inferBinding(expr.name);
      // Some surface idents (constants, set names) lower as refs
      // rather than const — handle that gracefully here too.
      if (CONST_TYPES[expr.name])    return CONST_TYPES[expr.name];
      if (builtins.isSet(expr.name)) return setMarker(expr.name);
      return T.failed('undefined name "' + expr.name + '"');
    }
    // Cross-module ref — `(%ref mod X)` resolves through the
    // module binding's loaded definition (spec §11). For standard
    // modules (engine-provided), look the binding up in the
    // registry; for `load_module` modules, the resolver descends
    // into the loaded LoweredModule (open follow-up — see
    // TODO-flatppl-js §multi-file).
    return inferCrossModuleRef(expr);
  }

  // Resolve `(%ref <module-alias> <name>)` through the module
  // binding's loaded definition. The module-alias is itself a
  // self-namespaced binding; we read its lowered IR to determine
  // which kind of module load (standard_module or load_module) and
  // dispatch accordingly.
  function inferCrossModuleRef(expr: any): any {
    const modAlias = expr.ns;
    const modBindingLoweredBinding = loweredModule.bindings.get(modAlias);
    if (!modBindingLoweredBinding) {
      return T.failed("undefined module alias '" + modAlias + "'");
    }
    const modRhs = modBindingLoweredBinding.rhs;
    if (!modRhs || modRhs.kind !== 'call'
        || (modRhs.op !== 'standard_module' && modRhs.op !== 'load_module')) {
      return T.failed("'" + modAlias
        + "' is not a module binding (cannot resolve "
        + modAlias + "." + expr.name + ")");
    }
    if (modRhs.op === 'standard_module') {
      return resolveStandardModuleRef(modRhs, expr.name, modAlias);
    }
    // load_module path — pending the multi-file end-to-end wiring.
    return T.deferred();
  }

  function resolveStandardModuleRef(stdModCall: any, bindingName: string, modAlias: string): any {
    // `standard_module(<name-lit>, <compat-lit>)`. Both args are
    // literal strings — the analyzer validates this shape upstream.
    const args = stdModCall.args || [];
    const nameArg = args[0], compatArg = args[1];
    if (!nameArg || nameArg.kind !== 'lit' || typeof nameArg.value !== 'string'
        || !compatArg || compatArg.kind !== 'lit' || typeof compatArg.value !== 'string') {
      return T.failed(modAlias + ': standard_module() requires literal name+compat strings');
    }
    const stdName = nameArg.value;
    const stdCompat = compatArg.value;
    const stdModules = require('./standard-modules.ts');
    const entry = stdModules.lookupStandardModule(stdName, stdCompat);
    if (!entry) {
      return T.failed("standard module '" + stdName + "@" + stdCompat
        + "' is not provided by this engine");
    }
    const desc = entry.bindings.get(bindingName);
    if (!desc) {
      return T.failed("'" + bindingName + "' is not a binding of standard module '"
        + stdName + "@" + stdCompat + "'");
    }
    // Each descriptor kind maps to a different type signal:
    //   - 'function': a `funcType(...)` per the descriptor's sig
    //     (built from types.ts factories at registration time).
    //   - 'value':    the descriptor's value already carries the
    //     concrete type (e.g. a literal scalar / array).
    //   - 'distribution': a kernel constructor — treated as a
    //     function returning a measure (the dist call's domain
    //     comes from the descriptor's `domain` field).
    if (desc.kind === 'function' && desc.sig) {
      return desc.sig;
    }
    if (desc.kind === 'value' && desc.valueType) {
      return desc.valueType;
    }
    if (desc.kind === 'distribution' && desc.sig) {
      return desc.sig;
    }
    return T.deferred();
  }

  function inferCall(expr: any, scopes: any): any {
    // User-defined call: lower.js puts the callee on `target`.
    if (expr.target) return inferUserCall(expr, scopes);

    // Expression-headed user call (spec §11 %call, b070d0a): the callee
    // is an expression — an inline reification in the common case
    // (`functionof(e, p = a)(2.5)`, a lambda application) — that must
    // evaluate to a user-defined callable. Infer the callee; a
    // function / kernel type applies to its result (monomorphic, like
    // inferUserCall); anything else is the typing-condition failure.
    if (expr.callee) {
      const calleeT = inferExpr(expr.callee, scopes);
      if (calleeT && (calleeT.kind === 'function' || calleeT.kind === 'kernel')) {
        return write(calleeT.result != null ? calleeT.result : T.deferred(), expr);
      }
      if (calleeT && calleeT.kind === 'failed') {
        return write(T.failed('expression-headed call cascade'), expr);
      }
      if (calleeT && calleeT.kind === 'deferred') return write(T.deferred(), expr);
      return write(T.failed('expression-headed call: callee must evaluate to a '
        + 'user-defined callable (function or kernel), got '
        + T.show(calleeT)), expr);
    }

    // Special-cased ops whose result type depends on actuals or
    // structural shape in ways that don't fit the static signature
    // table.
    switch (expr.op) {
      case 'elementof': return write(inferElementof(expr, scopes), expr);
      case 'lawof':     return write(inferLawof(expr, scopes), expr);
      case 'record':    return write(inferRecord(expr, scopes), expr);
      case 'table':     return write(inferTable(expr, scopes), expr);
      case 'joint':     return write(inferJoint(expr, scopes), expr);
      case 'tuple':     return write(inferTuple(expr, scopes), expr);
      // tuple_get(<tuple-expr>, <slot lit>) — internal IR op emitted by
      // pir.lowerToModule for multi-LHS bindings (`a, b = rand(...)`).
      // Result type is the tuple's i-th element. Special-cased because
      // the result depends on the literal slot value, which the generic
      // signature table can't express.
      case 'tuple_get': return write(inferTupleGet(expr, scopes), expr);
      // get_field(<record-expr>, <name lit>) — record field access
      // lowered from surface `obj.field`. Same kind of special case as
      // tuple_get: the result type depends on the literal field-name
      // argument.
      case 'get_field': return write(inferGetField(expr, scopes), expr);
      // Lebesgue / Counting parametrise on a support set (spec §06).
      // The measure's domain mirrors the support's value-type, so
      // Lebesgue(support = cartpow(reals, n)) is measure(array<real,n>).
      case 'Lebesgue':  return write(inferReferenceMeasure(expr, scopes, T.REAL), expr);
      case 'Counting':  return write(inferReferenceMeasure(expr, scopes, T.INTEGER), expr);
      case 'vector':    return write(inferVector(expr, scopes), expr);
      case 'iid':       return write(inferIid(expr, scopes), expr);
      // Spec §07 table reductions: when sum / mean / var / std / prod /
      // maximum / minimum is applied to a table, the result is a record
      // whose fields are the column names and values are the per-column
      // reductions. Check the input type up front so the static SIGNATURE_
      // FACTORIES entry (which returns any() / REAL) doesn't shadow this.
      case 'sum':
      case 'mean':
      case 'var':
      case 'std':
      case 'prod':
      case 'maximum':
      case 'minimum': {
        const tbl = _maybeTableReduction(expr, scopes);
        if (tbl != null) return write(tbl, expr);
        break;
      }
      // lengthof(t) for a table returns the row count (an integer).
      // The signature factory already returns INTEGER unconditionally
      // (it works for vectors and tables uniformly); no extra case here.
      // Const-eval-driven shape inference (engine-concepts §17.4).
      // Shape-determining producers whose result rank/shape depends
      // on their dim arg(s). All of them consult the resolver via
      // resolveIntegerShape / resolveIntegerVectorShape; if the dim
      // resolves to a concrete integer (or shape vector), the result
      // type carries that; otherwise %dynamic falls through.
      case 'zeros':
      case 'ones':      return write(inferZerosOnes(expr, scopes), expr);
      case 'fill':      return write(inferFill(expr, scopes), expr);
      case 'eye':       return write(inferEye(expr, scopes), expr);
      case 'onehot':    return write(inferOnehot(expr, scopes), expr);
      // rowstack/colstack of an inline vector-of-vectors literal can
      // pin a concrete [m, n] shape — the literal carries the outer
      // and inner lengths exactly. Without this, the resolver has
      // to materialise the matrix to know its dims, defeating the
      // demand-driven design when chains go through lengthof / sizeof
      // of a rowstack output. Fall back to %dynamic for non-literal
      // shape-of-shape inputs.
      case 'rowstack':
      case 'colstack': return write(inferRowstack(expr, scopes), expr);
      // get / get0 — unified element / subset / axis-slice / singleton
      // access (spec §07). Shape inference here covers the array case
      // (rank, shape, elem) precisely; the record case redirects to
      // get_field's machinery. Static singleton-axis check (`only` /
      // `!`): when the indexed dim's length is statically known and
      // ≠ 1, emit a static error.
      case 'get':
      case 'get0':      return write(inferGet(expr, scopes), expr);
      // Multi-axis aggregation (spec §04 §sec:aggregate). Result
      // shape = [length(.axis) for axis in output_axes], element
      // type follows the expr. Axis lengths come from the first
      // get/get0 indexing position of each axis name in expr
      // (statically-resolvable when the indexed array's shape is
      // known).
      case 'aggregate': return write(inferAggregate(expr, scopes), expr);
      // Metric-aware Einstein summation (spec §04 §sec:metricsum).
      // Pre-lift type-shape: identical to aggregate (axis lengths come
      // from the body's indexings; result shape = lengths-per-output-axis,
      // element type = body element type, scalar for empty axis list).
      // Variance markers on output axes are TYPE-IRRELEVANT here — the
      // post-lift result tensor is stored in all-upper canonical layout,
      // so its rank-N shape matches a plain `aggregate(sum, ...)` over
      // the same axis lengths. The metricsum → aggregate lift pass
      // strips variance markers; typeinfer just needs to surface the
      // right shape for downstream consumers BEFORE lift runs.
      case 'metricsum':  return write(inferMetricsum(expr, scopes), expr);
      // Density-evaluation ops (spec §06 / §07): the inferred type
      // comes from the static signature, BUT we additionally run the
      // type-mode consume/rest walker over (measure-IR, variate-type)
      // to surface shape-mismatched joints / iid lengths / record
      // fields as parse-time diagnostics rather than runtime
      // exceptions. Engine-concepts §17.3.
      case 'logdensityof':
      case 'densityof':
      case 'bayesupdate':
      case 'likelihoodof': {
        const t = inferLikelihoodOps(expr, scopes);
        _checkDensityShapes(expr);
        return write(t, expr);
      }
      // kernelof and fn are lowered to functionof by lower.js (per
      // spec §sec:kernelof line 421-422 and §sec:fn line 618-628),
      // so we only see functionof here.
      case 'functionof': return write(inferReification(expr, scopes), expr);
      // fchain(f1, f2, ...) — deterministic function composition
      // (spec §04 Function composition and annotation, engine-concepts
      // §19.4). Result type is the composed function type computed by
      // the shared `inferChainComposition` helper (consume/rest at the
      // chain's input-set level — engine-concepts §17.3 extended).
      case 'fchain':    return write(inferFchain(expr, scopes), expr);
      // jointchain / kchain — dependent composition (spec §06 line
      // 192-266). Closed-first ⇒ measureType; kernel-first ⇒
      // kernelType with residual inputs (collapses to measure when
      // residual is empty). Routes through inferChainComposition's
      // kernel modes (engine-concepts §19.4).
      case 'jointchain': return write(inferJointchain(expr, scopes, 'jointchain-retain'), expr);
      case 'kchain':     return write(inferJointchain(expr, scopes, 'kchain-marginal'), expr);
      // transpose / adjoint: spec §07 — apply to vectors and matrices.
      // For vectors, return type is transposed_vector (the new spec
      // type from flatppl-design 244b0e5); for transposed_vector,
      // return type is vector (involution); for matrices, swap dims.
      case 'transpose': return write(inferTransposeAdjoint(expr, scopes), expr);
      case 'adjoint':   return write(inferTransposeAdjoint(expr, scopes), expr);
      // weighted / logweighted: per spec §06 the weight is a constant
      // OR a function of the variate (returning real). The static
      // type signature pins the scalar form; the function form is
      // accepted here as a separate code path that still rejects
      // measure / record / etc. weights (matching the existing
      // "weighted(measure, measure)" reject path).
      case 'weighted':
      case 'logweighted': return write(inferWeighted(expr, scopes), expr);
      case 'pushfwd':     return write(inferPushfwd(expr, scopes), expr);
      // A non-scalar locscale survives the analyzer pre-pass as an
      // `{op:'locscale'}` node (it routes through lift's affine-registry
      // lowering, not the scalar expansion). Its type follows the base
      // measure — an affine pushforward changes neither domain nor shape.
      case 'locscale':    return write(inferLocscale(expr, scopes), expr);
    }
    // Numeric arithmetic with shape polymorphism: both scalars,
    // both arrays of matching shape, or scalar/array broadcast.
    if (BINARY_ARITH_OPS.has(expr.op)) return write(inferArith2(expr, scopes), expr);
    if (UNARY_ARITH_OPS.has(expr.op))  return write(inferArith1(expr, scopes), expr);
    if (COMPARISON_OPS.has(expr.op))   return write(inferComparison(expr, scopes), expr);
    // broadcast(fn, A1, A2, ...) propagates the unifying shape of the
    // data args. Crucial for the viewer's matrix-heatmap dispatch:
    // a dotted op (`.^ 2`, `.*`, …) lowers to `broadcast(...)`, and
    // without this fall-through the result inferred as `deferred()`
    // (so a 3×3 result rendered as a scalar — see TODO §07 broadcast).
    if (expr.op === 'broadcast') return write(inferBroadcast(expr, scopes), expr);

    return write(inferGenericCall(expr, scopes), expr);
  }

  // Helper to attach inferred type to the call's meta slot AND
  // return the type. setMeta is from pir.js but we don't import to
  // keep this module standalone — direct write is fine.
  function write(t: any, expr: any) {
    if (!expr.meta) expr.meta = {};
    expr.meta.type = t;
    return t;
  }

  // -------------------------------------------------------------------
  // Generic call inference: signature lookup + arg unify
  // -------------------------------------------------------------------

  function inferGenericCall(expr: any, scopes: any): any {
    const op = expr.op;
    const sig = T.signatureOf(op);
    if (!sig) return T.deferred();

    let s = new Map();
    const args   = expr.args   || [];
    const kwargs = expr.kwargs || {};

    if (sig.args !== null) {
      const rawN = sig.args.length;
      const got  = args.length;
      const variadic = sig.variadic === 'positional';
      const fixedN = variadic ? rawN - 1 : rawN;
      // Pure-kwargs form is permitted (spec §05 calling conventions):
      // every ordinary built-in accepts either positional OR keyword
      // arg lists. If the caller used 0 positional args AND the
      // signature lists kwargs whose names match the positional
      // slots, skip the positional arity + type checks and rely on
      // the kwarg loop below to validate. Mixed positional+kwarg
      // calls go through both branches.
      const kwargsCoverPositional =
        got === 0 && sig.kwargs && Object.keys(sig.kwargs).length > 0;
      if (!kwargsCoverPositional) {
        if (variadic) {
          if (got < fixedN) return arityError(op, '≥' + fixedN, got, expr.loc);
        } else if (got !== rawN) {
          return arityError(op, rawN, got, expr.loc);
        }
        for (let i = 0; i < fixedN; i++) {
          const at: any = inferExpr(args[i], scopes);
          const next = T.unify(sig.args[i], at, s);
          if (next == null) {
            // Apply the substitution accumulated from earlier args so
            // the diagnostic's "expected" type is concrete, not the
            // bare type variable. E.g. `logdensityof(iid(Normal,3), data_5)`
            // reports "arg 2 expects array of real (length 3), got
            // array of real (length 5)" instead of the noisy
            // "arg 2 expects any, got …" (engine-concepts §17.4).
            return argError(op, i, T.substitute(sig.args[i], s), at, args[i].loc);
          }
          s = next;
        }
        if (variadic) {
          const tail = sig.args[rawN - 1];
          for (let i = fixedN; i < got; i++) {
            const at: any = inferExpr(args[i], scopes);
            const next = T.unify(tail, at, s);
            if (next == null) return argError(op, i, T.substitute(tail, s), at, args[i].loc);
            s = next;
          }
        }
      }
    }

    for (const k in sig.kwargs) {
      if (!(k in kwargs)) continue;   // optional/defaulted kwargs allowed missing
      const at: any = inferExpr(kwargs[k], scopes);
      const next = T.unify(sig.kwargs[k], at, s);
      if (next == null) return kwargError(op, k, T.substitute(sig.kwargs[k], s), at, kwargs[k].loc);
      s = next;
    }
    return T.substitute(sig.result, s);
  }

  // Likelihood-object ops (spec §06). `likelihoodof(K, obs)` produces a
  // first-class LIKELIHOOD object (not a measure, not callable) whose
  // parameter interface is the kernel's inputs — so its inferredType is
  // a concrete `likelihood`, not `deferred`. `densityof`/`logdensityof`
  // accept either a measure (→ existing signature path) OR a likelihood
  // (→ a real density/log-density value). `bayesupdate(L, prior)` is the
  // unnormalized posterior measure over the prior's domain.
  function inferLikelihoodOps(expr: any, scopes: any): any {
    const op = expr.op;
    const args = expr.args || [];
    if (op === 'likelihoodof') {
      const kT: any = args.length > 0 ? inferExpr(args[0], scopes) : T.deferred();
      if (args.length > 1) inferExpr(args[1], scopes);   // keep obs typed/written
      // K is a kernel; a measure (nullary kernel) has no parameter
      // interface. Carry the kernel's inputs as the likelihood's params.
      const inputs = (kT && kT.kind === 'kernel') ? kT.inputs : undefined;
      return T.likelihood(inputs);
    }
    if (op === 'densityof' || op === 'logdensityof') {
      const mT: any = args.length > 0 ? inferExpr(args[0], scopes) : T.deferred();
      if (mT && mT.kind === 'likelihood') {
        if (args.length > 1) inferExpr(args[1], scopes);   // theta (param point)
        return T.REAL;
      }
      return inferGenericCall(expr, scopes);   // measure(T) signature path
    }
    if (op === 'bayesupdate') {
      const lT: any = args.length > 0 ? inferExpr(args[0], scopes) : T.deferred();
      const pT: any = args.length > 1 ? inferExpr(args[1], scopes) : T.deferred();
      if (lT && lT.kind === 'likelihood' && T.isMeasure(pT)) return pT;
      return inferGenericCall(expr, scopes);
    }
    return inferGenericCall(expr, scopes);
  }

  // -------------------------------------------------------------------
  // Special-case op handlers
  // -------------------------------------------------------------------

  // transpose / adjoint dispatch on input shape per spec §07:
  //   vector(n)            → transposed_vector(n)
  //   transposed_vector(n) → vector(n)             (involution)
  //   matrix(m, n)         → matrix(n, m)          (dim swap)
  //   matrix(N, m, n)      → matrix(N, n, m)       (atom-batched per-slice;
  //                          rank-3 not currently expressible in array type
  //                          system but documented here for future extension)
  //   scalar               → error (transpose undefined on scalars)
  //   other                → deferred (unknown shape — let runtime handle)
  // weighted(weight, base) / logweighted(weight, base) per spec §06.
  // `weight` is a non-negative real OR a function (returning real)
  // of the variate; `base` is a measure. The result is a measure
  // with the base's domain. We unify the weight type against either
  // REAL or a function type (any inputs → real result), and reject
  // anything else (notably measures — that's a common user mistake
  // the old signature already caught).
  function inferWeighted(expr: any, scopes: any): any {
    const op = expr.op;
    const args = expr.args || [];
    if (args.length !== 2) return arityError(op, 2, args.length, expr.loc);
    const wT: any = inferExpr(args[0], scopes);
    const mT: any = inferExpr(args[1], scopes);
    // Measure check first (matches existing diagnostic style).
    if (!T.isMeasure(mT)) {
      diagnostics.push({
        severity: 'error',
        message: op + ': arg 2 expects measure, got ' + T.show(mT),
        loc: args[1].loc,
      });
      return T.failed(op + ' arg 2');
    }
    // Weight: accept REAL, INTEGER (promotes to real), or a function
    // type returning a scalar. Reject measures, records, etc. Cascades
    // (wT.kind === 'failed') are suppressed — the root error already
    // emitted a diagnostic, no need to pile another on.
    if (wT && wT.kind === 'failed') return T.failed(op + ' arg 1 (cascade)');
    const isReal = wT && wT.kind === 'scalar'
      && (wT.prim === 'real' || wT.prim === 'integer');
    const isFunction = wT && wT.kind === 'function';
    const isDeferred = wT && (wT.kind === 'deferred' || wT.kind === 'any'
      || wT.kind === 'var');
    if (!isReal && !isFunction && !isDeferred) {
      diagnostics.push({
        severity: 'error',
        message: op + ': arg 1 expects real or function, got ' + T.show(wT),
        loc: args[0].loc,
      });
      return T.failed(op + ' arg 1');
    }
    return mT;  // result is the same measure type
  }

  function inferTransposeAdjoint(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 1) return arityError(expr.op, 1, args.length, expr.loc);
    const at: any = inferExpr(args[0], scopes);
    if (!at) return T.deferred();
    switch (at.kind) {
      case 'array':
        if (at.rank === 1) {
          // vector(n) → transposed_vector(n)
          return T.tvector(at.shape[0], at.elem);
        }
        if (at.rank === 2) {
          // matrix(m, n) → matrix(n, m). Swap shape entries; elem unchanged.
          return T.array(2, [at.shape[1], at.shape[0]], at.elem);
        }
        // rank ≥ 3 — spec is silent; runtime supports swap-last-two-axes.
        // Pass through unchanged at the type level.
        return at;
      case 'tvector':
        // transposed_vector(n) → vector(n)
        return T.array(1, [at.length], at.elem);
      case 'scalar':
        return T.failed(expr.op + ': not defined on scalars');
      case 'deferred':
      case 'any':
      case 'var':
        return T.deferred();
    }
    return T.deferred();
  }

  function inferElementof(expr: any, scopes: any) {
    const args = expr.args || [];
    if (args.length !== 1) return arityError('elementof', 1, args.length, expr.loc);
    const t = setValueType(args[0], scopes);
    if (t == null) {
      const argT = inferExpr(args[0], scopes);
      if (argT && argT.kind === 'failed') return T.failed('elementof cascade');
      diagnostics.push({
        severity: 'error',
        message: 'elementof expects a set or set-constructor expression; got ' + T.show(argT),
        loc: args[0].loc,
      });
      return T.failed('elementof bad arg');
    }
    return t;
  }

  function inferLawof(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 1) return arityError('lawof', 1, args.length, expr.loc);
    const at: any = inferExpr(args[0], scopes);
    if (at && at.kind === 'failed') return T.failed('lawof cascade');
    if (T.isMeasure(at)) return at;             // identity law: lawof(measure) = measure
    if (T.isValue(at))   return T.measure(at);
    diagnostics.push({
      severity: 'error',
      message: 'lawof expects a value-typed argument, got ' + T.show(at),
      loc: args[0].loc,
    });
    return T.failed('lawof bad arg');
  }

  // Spec §04: measures, kernels, likelihood objects and functions are
  // first-class objects but "may not appear inside arrays, records, or
  // tables" — only value types (scalars, arrays, records, tables, tuples)
  // may. Without this guard a `record(a = Exponential(...), …)` infers an
  // uninhabitable record-of-measures type silently (no LSP/plot error) and
  // then mis-materialises downstream. Reject the object-layer field/element
  // types with a clear, spec-citing diagnostic. (A field like `a = draw(M)`
  // or `a = some_variate` is a VALUE — scalar-typed — and passes; the prior
  // idiom `joint(a = M, …)` is the correct way to build a measure over
  // records.) Does not return `failed` — the type still builds, so the rest
  // of inference proceeds; this is an additive diagnostic.
  function objectLayerNoun(t: any): string | null {
    if (!t) return null;
    switch (t.kind) {
      case 'measure':    return 'measure';
      case 'kernel':     return 'kernel';
      case 'function':   return 'function';
      case 'likelihood': return 'likelihood object';
      default:           return null;
    }
  }
  function checkContainerElem(t: any, loc: any, container: string, label: string) {
    const noun = objectLayerNoun(t);
    if (!noun) return;
    const art = container === 'array' ? 'an ' : 'a ';
    diagnostics.push({
      severity: 'error',
      message: container + ' ' + label + ': a ' + noun + ' may not appear inside ' + art
        + container + ' (spec §04 — measures, kernels, likelihoods and functions '
        + 'are first-class objects but cannot be stored in arrays, records, or tables)'
        + (t.kind === 'measure'
          ? '; use joint(' + (container === 'record' ? 'name = M, …' : 'M, …')
            + ') to build a measure over ' + container + 's' : ''),
      loc,
    });
  }

  function inferRecord(expr: any, scopes: any) {
    // record uses `fields` (ordered), not `kwargs`.
    const fields = expr.fields || [];
    const out: Record<string, any> = {};
    for (const f of fields) {
      const ft = inferExpr(f.value, scopes);
      checkContainerElem(ft, f.loc || expr.loc, 'record', "field '" + f.name + "'");
      out[f.name] = ft;
    }
    return T.record(out);
  }

  // Spec §07 "Table reductions": sum / mean / var / std / prod / max /
  // min applied to a table operates column-wise and returns a record
  // whose fields are the column names and values are the per-column
  // reductions. Returns null when the input isn't a table — caller
  // falls through to the signature-factory path (which handles arrays
  // and scalars).
  //
  // Per-column result type:
  //   sum, prod, mean   → same as column element type (real / complex)
  //   var, std          → real (Bessel-corrected variance / its sqrt)
  //   maximum, minimum  → same as column element type
  function _maybeTableReduction(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 1) return null;
    const t: any = inferExpr(args[0], scopes);
    if (!t || t.kind !== 'table') return null;
    const op = expr.op;
    const fields: Record<string, any> = {};
    for (const k in t.columns) {
      const cT = t.columns[k];
      if (op === 'var' || op === 'std') {
        fields[k] = T.REAL;
      } else {
        // sum, prod, mean, maximum, minimum preserve element type.
        fields[k] = cT;
      }
    }
    return T.record(fields);
  }

  // table(col1 = [...], col2 = [...]) per spec §03. Each column's
  // value must be a 1-D array; all columns must have the same length
  // (the row count). The table type carries column-element types (not
  // the array types themselves — spec §11's table shape is
  // `(%table (%columns (<name> <element-type>) ...) (%nrows <N>))`).
  function inferTable(expr: any, scopes: any) {
    const fields = expr.fields || [];
    if (fields.length === 0) return T.failed('table: at least one column required');
    const columns: Record<string, any> = {};
    let nrows: number | '%dynamic' = '%dynamic';
    let nrowsBound = false;
    for (const f of fields) {
      const ct: any = inferExpr(f.value, scopes);
      // The value should be a rank-1 array. Deferred / any flow
      // through with deferred column type — engine still produces
      // a table at runtime.
      if (ct && ct.kind === 'array' && ct.rank === 1) {
        // Spec §04: a table column may not hold measures / kernels / etc.
        checkContainerElem(ct.elem, f.loc || expr.loc, 'table', "column '" + f.name + "'");
        columns[f.name] = ct.elem;
        const dim = ct.shape[0];
        if (typeof dim === 'number') {
          if (!nrowsBound) { nrows = dim; nrowsBound = true; }
          else if (nrows !== dim) {
            diagnostics.push({
              severity: 'error',
              message: 'table: column "' + f.name + '" has length ' + dim
                + ', but earlier columns have length ' + nrows
                + ' (spec §03: all columns must have equal length)',
              loc: f.loc || expr.loc,
            });
            return T.failed('table column length mismatch');
          }
        }
      } else if (ct && (ct.kind === 'deferred' || ct.kind === 'any')) {
        columns[f.name] = T.deferred();
      } else if (ct && ct.kind === 'failed') {
        return T.failed('table column cascade');
      } else {
        diagnostics.push({
          severity: 'error',
          message: 'table: column "' + f.name + '" must be an array (vector); got '
            + T.show(ct),
          loc: f.loc || expr.loc,
        });
        return T.failed('table non-array column');
      }
    }
    return T.table(columns, nrows);
  }

  function inferJoint(expr: any, scopes: any) {
    const fields = expr.fields || [];
    const out: Record<string, any> = {};
    for (const f of fields) {
      const at = inferExpr(f.value, scopes);
      if (T.isMeasure(at)) out[f.name] = at.domain;
      else if (at.kind === 'deferred' || at.kind === 'any') out[f.name] = T.deferred();
      else if (at.kind === 'failed') return T.failed('joint cascade');
      else {
        diagnostics.push({
          severity: 'error',
          message: 'joint kwarg "' + f.name + '" expects a measure, got ' + T.show(at),
          loc: f.value.loc || expr.loc,
        });
        return T.failed('joint bad kwarg');
      }
    }
    return T.measure(T.record(out));
  }

  /**
   * fchain(f1, f2, ..., fN) — deterministic function composition per
   * spec §04. Result type is the composed funcType, computed by the
   * shared `inferChainComposition` helper (engine-concepts §19.4).
   *
   * Per-step requirement: each arg's inferred type must be `funcType`.
   * Non-function step types surface as a step-anchored diagnostic
   * from the helper.
   */
  function inferFchain(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length === 0) {
      diagnostics.push({
        severity: 'error',
        message: 'fchain requires ≥ 1 function arg (spec §04 forbids nullary callables)',
        loc: expr.loc,
      });
      return T.failed('fchain nullary');
    }
    const steps: any[] = [];
    for (const a of args) {
      // Refs carry a name; we pass it through for diagnostic clarity.
      const name = (a && a.kind === 'ref' && a.ns === 'self') ? a.name : undefined;
      steps.push({ type: inferExpr(a, scopes), loc: a && a.loc, name });
    }
    const densityPrims = require('./density-prims.ts');
    const r = densityPrims.inferChainComposition(steps, 'func');
    for (const d of r.diagnostics) diagnostics.push(d);
    return r.resultType;
  }

  /**
   * jointchain / kchain — dependent composition (spec §06 line
   * 192-266; engine-concepts §19.4). Lowering surfaces:
   *
   *   - Positional `jointchain(M, K1, K2)` → `args: [...]`
   *   - Keyword    `jointchain(name1 = M, name2 = K)` → `fields:
   *                  [{name, value}, ...]` (labels preserved for the
   *                  retained record shape).
   *
   * Routes through `inferChainComposition` with mode='kchain-marginal'
   * (kchain) or 'jointchain-retain' (jointchain). Result-kind dispatch
   * happens inside the helper: kernel-first → kernelType with residual
   * inputs; closed-first → measureType (the spec §06 kernel↔measure
   * collapse at residual-empty falls out without a special case).
   */
  function inferJointchain(expr: any, scopes: any,
      mode: 'kchain-marginal' | 'jointchain-retain'): any {
    let comps: any[];
    let labels: string[] | null = null;
    if (Array.isArray(expr.fields) && expr.fields.length > 0) {
      labels = expr.fields.map((f: any) => f.name);
      comps  = expr.fields.map((f: any) => f.value);
    } else {
      comps = expr.args || [];
    }
    const opName = (mode === 'kchain-marginal') ? 'kchain' : 'jointchain';
    if (comps.length < 2) {
      diagnostics.push({
        severity: 'error',
        message: opName + ' requires ≥ 2 components (spec §06 line 192-266)',
        loc: expr.loc,
      });
      return T.failed(opName + ' arity');
    }
    const steps: any[] = [];
    for (const a of comps) {
      const name = (a && a.kind === 'ref' && a.ns === 'self') ? a.name : undefined;
      steps.push({ type: inferExpr(a, scopes), loc: a && a.loc, name });
    }
    const densityPrims = require('./density-prims.ts');
    const r = densityPrims.inferChainComposition(steps, mode, { labels });
    for (const d of r.diagnostics) diagnostics.push(d);
    return r.resultType;
  }

  function inferTuple(expr: any, scopes: any) {
    const args = expr.args || [];
    return T.tuple(args.map((a: any) => inferExpr(a, scopes)));
  }

  // Lebesgue(support = S) / Counting(support = S). The support kwarg is
  // optional; missing → use the default scalar. When present, the support
  // is a set expression whose value-type drives the result's measure-domain
  // (cartpow → array, cartprod → record/tuple, stdsimplex → array<real,n>).
  // Falls back to the default scalar when the support shape can't be
  // statically resolved (e.g. the support is itself a binding ref).
  function inferReferenceMeasure(expr: any, scopes: any, defaultElem: any) {
    const args   = expr.args || [];
    const kwargs = expr.kwargs || {};
    let support = null;
    // Spec §06 form is `Lebesgue(support = S)`, but keep accepting a
    // single positional support per the calling-convention rule.
    if ('support' in kwargs)        support = kwargs.support;
    else if (args.length === 1)     support = args[0];
    if (!support) return T.measure(defaultElem);
    const t = setValueType(support, scopes);
    return T.measure(t || defaultElem);
  }

  function inferGetField(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 2) {
      return arityError('get_field', 2, args.length, expr.loc);
    }
    const recT: any = inferExpr(args[0], scopes);
    if (recT && recT.kind === 'failed') return T.failed('get_field cascade');
    // Tables and records share the dot-access shape `r.col` (spec §03):
    // column access on a table returns the column AS A VECTOR (array of
    // the column-element type and length nrows); field access on a
    // record returns the field value directly.
    if (recT && recT.kind === 'table' && recT.columns) {
      const nameIR = args[1];
      if (!nameIR || nameIR.kind !== 'lit' || typeof nameIR.value !== 'string') {
        return T.failed('get_field name must be a literal string');
      }
      if (!(nameIR.value in recT.columns)) {
        diagnostics.push({
          severity: 'error',
          message: `get_field: '${nameIR.value}' is not a column of ${T.show(recT)}`,
          loc: expr.loc,
        });
        return T.failed('get_field unknown column');
      }
      return T.array(1, [recT.nrows], recT.columns[nameIR.value]);
    }
    // `any` / `deferred` recipients (common in user-fn bodies where
    // the param's type isn't pinned at definition — the param defaults
    // to `any` until B5's polymorphic-at-call-site re-infers the body
    // with the actual call-site type) flow through as deferred result.
    // This lets `row -> row.col` body-infer without a spurious error
    // when row's type is unknown statically.
    if (recT && (recT.kind === 'any' || recT.kind === 'deferred')) {
      return T.deferred();
    }
    if (!recT || recT.kind !== 'record' || !recT.fields) {
      diagnostics.push({
        severity: 'error',
        message: 'get_field expects a record-typed expression; got ' + T.show(recT),
        loc: args[0].loc || expr.loc,
      });
      return T.failed('get_field bad arg');
    }
    const nameIR = args[1];
    if (!nameIR || nameIR.kind !== 'lit' || typeof nameIR.value !== 'string') {
      return T.failed('get_field name must be a literal string');
    }
    if (!(nameIR.value in recT.fields)) {
      diagnostics.push({
        severity: 'error',
        message: `get_field: '${nameIR.value}' is not a field of ${T.show(recT)}`,
        loc: expr.loc,
      });
      return T.failed('get_field unknown field');
    }
    return recT.fields[nameIR.value];
  }

  function inferTupleGet(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 2) {
      return arityError('tuple_get', 2, args.length, expr.loc);
    }
    const tupleT: any = inferExpr(args[0], scopes);
    if (tupleT && tupleT.kind === 'failed') return T.failed('tuple_get cascade');
    if (!tupleT || tupleT.kind !== 'tuple') {
      diagnostics.push({
        severity: 'error',
        message: 'tuple_get expects a tuple-typed expression; got ' + T.show(tupleT),
        loc: args[0].loc || expr.loc,
      });
      return T.failed('tuple_get bad arg');
    }
    // Slot must be a literal int — the lowering pass always emits one.
    const slotIR = args[1];
    if (!slotIR || slotIR.kind !== 'lit' || typeof slotIR.value !== 'number') {
      return T.failed('tuple_get slot must be a literal index');
    }
    const i = slotIR.value | 0;
    if (i < 0 || i >= tupleT.elems.length) {
      diagnostics.push({
        severity: 'error',
        message: `tuple_get index ${i} out of range for ${T.show(tupleT)}`,
        loc: expr.loc,
      });
      return T.failed('tuple_get out of range');
    }
    return tupleT.elems[i];
  }

  // -------------------------------------------------------------------
  // get / get0: array & record element / subset / slice access
  // -------------------------------------------------------------------
  //
  // Shape-precise inference for the array case. For each selector
  // position k we know exactly how it affects the result:
  //
  //   {const all}            — keeps dim k                (shape[k] kept)
  //   {const only}           — drops dim k; STATIC ERROR
  //                            if shape[k] is known and ≠ 1
  //   {axis name}            — drops dim k                (axis env at runtime)
  //   {lit number}           — drops dim k
  //   {call vector …}        — subset selection: keeps dim k with
  //                            new length = args.length
  //   anything else (deferred selector) — conservatively drops dim k
  //
  // Unselected tail dims (when k < rank) are kept verbatim.
  //
  // The container type may be array, record, tuple, or unknown
  // (deferred / any). The array branch is the only one that performs
  // shape inference today; the others either redirect (record →
  // inferGetField shape, tuple → inferTupleGet shape) or fall back
  // to deferred. This keeps the function focused on the singleton-
  // axis static check that motivated it without expanding the
  // typeinfer surface across all container kinds.

  function inferGet(expr: any, scopes: any) {
    const args = expr.args || [];
    if (args.length < 2) {
      return arityError(expr.op, '≥ 2', args.length, expr.loc);
    }
    const containerT: any = inferExpr(args[0], scopes);
    if (containerT && containerT.kind === 'failed') return T.failed('get cascade');

    // Tuple integer-index: redirect to tuple_get's existing inferrer
    // (same shape — single integer literal selector).
    if (containerT && containerT.kind === 'tuple'
        && args.length === 2
        && args[1].kind === 'lit'
        && typeof args[1].value === 'number') {
      return inferTupleGet(expr, scopes);
    }

    // Record single-field access: redirect to get_field's shape rule.
    // Subset selection (array of strings) → deferred for v0.1.
    if (containerT && containerT.kind === 'record') {
      if (args.length === 2 && args[1].kind === 'lit'
          && typeof args[1].value === 'string') {
        return inferGetField(expr, scopes);
      }
      return T.deferred();
    }

    // Table access (spec §03):
    //   - get(t, i) (integer literal/integer expr) → record per row.
    //   - get(t, "col") → array of column-element type, length = nrows.
    //   - get(t, ["c1","c2"]) → table sub-selection (deferred for v0.1).
    if (containerT && containerT.kind === 'table') {
      if (args.length === 2 && args[1].kind === 'lit'
          && typeof args[1].value === 'string') {
        // Column access — same shape as get_field.
        return inferGetField(expr, scopes);
      }
      if (args.length === 2 && args[1].kind === 'lit'
          && typeof args[1].value === 'number') {
        // Row access — returns a record over the same column names.
        return T.record(containerT.columns);
      }
      if (args.length === 2) {
        // Row access via expression (axis or computed int). Conservative:
        // a row is a record over the table's columns.
        const selT: any = inferExpr(args[1], scopes);
        if (selT && selT.kind === 'scalar' && selT.prim === 'integer') {
          return T.record(containerT.columns);
        }
      }
      return T.deferred();
    }

    if (containerT && containerT.kind === 'array') {
      // Nested arrays (`[[1,2],[3,4]]` lowers to vector(vector(…))) carry
      // a rank-1 outer with an array-typed elem; for indexing purposes
      // the spec treats them as a single multi-dim array (`A[i, j]` ≡
      // `A[i][j]`). Flatten before walking the selectors.
      const flat = _flattenArrayType(containerT);
      const rank = flat.rank;
      const shape = flat.shape;
      const sels = args.slice(1);
      if (sels.length > rank) {
        diagnostics.push({
          severity: 'error',
          message: `${expr.op}: too many selectors (${sels.length}) `
            + `for rank-${rank} array`,
          loc: expr.loc,
        });
        return T.failed('get over-selected');
      }
      const outShape: any[] = [];
      for (let k = 0; k < sels.length; k++) {
        const sel = sels[k];
        const dim = shape[k];
        if (sel && sel.kind === 'const' && sel.name === 'all') {
          outShape.push(dim);                        // keep dim
          continue;
        }
        if (sel && sel.kind === 'const' && sel.name === 'only') {
          // Static singleton-axis check: when the dim's length is
          // statically known and ≠ 1, this is a static error per
          // spec §07 "Singleton-axis indexing with `only`".
          if (typeof dim === 'number' && dim !== 1) {
            diagnostics.push({
              severity: 'error',
              message: `${expr.op}: 'only' selector requires the indexed `
                + `axis to have length 1, got length ${dim}`,
              loc: sel.loc || expr.loc,
            });
            return T.failed('only on non-singleton');
          }
          continue;                                   // drop dim
        }
        if (sel && sel.kind === 'axis') {
          continue;                                   // drop dim (axis env at runtime)
        }
        if (sel && sel.kind === 'lit' && typeof sel.value === 'number') {
          continue;                                   // drop dim (integer index)
        }
        if (sel && sel.kind === 'call' && sel.op === 'vector') {
          // Subset selection — new dim length = number of vector args
          // (or %dynamic if the args list is variadic at runtime).
          outShape.push(Array.isArray(sel.args) ? sel.args.length : '%dynamic');
          continue;
        }
        // Array-valued selector (array-of-indices subset selection / gather,
        // spec §07): `A[idx]` where `idx` is any rank-1 integer array — a
        // named binding (`theta[person]`), an `indicesof`, etc. The result
        // dim length is the INDEX array's length, not the indexed array's.
        // Without this the dim was dropped with no length, so a gather whose
        // index is a ref (not an inline `[...]`) lost its axis — and an
        // expression of two such gathers (`theta[person] .- b[item]`, no
        // literal-length operand to anchor it) inferred a %dynamic broadcast
        // axis, mis-sizing the downstream density variate footprint.
        {
          const selT: any = inferExpr(sel, scopes);
          if (selT && selT.kind === 'array' && Array.isArray(selT.shape)
              && selT.shape.length === 1) {
            outShape.push(selT.shape[0]);
            continue;
          }
        }
        // Unrecognised selector shape — be conservative and drop the
        // dim with a deferred length. This still keeps the rest of
        // the array's shape information.
        // (Falls through; no push.)
      }
      // Tail dims that weren't selected stay verbatim.
      for (let k = sels.length; k < rank; k++) outShape.push(shape[k]);
      if (outShape.length === 0) return flat.elem;
      return T.array(outShape.length, outShape, flat.elem);
    }

    // Deferred / any / tvector / measure — no shape inference yet.
    return T.deferred();
  }

  // Flatten nested array types for INDEX-DEPTH resolution only:
  // `array(1, [m], array(1, [n], T))` looks like
  // `array(2, [m, n], T)` for the purpose of `inferGet`'s multi-d
  // indexing sugar (spec §05/§07: `A[i, j]` ≡ `A[i][j]` is defined
  // on both flat arrays and arrays of arrays). The flattening here
  // is ONLY for resolving the element type after N selectors and
  // discovering axis lengths in aggregate bodies — it does NOT
  // collapse the spec §03 semantic distinction between a flat matrix
  // and a vector-of-vectors at any other site.
  //
  // Callers that care about the spec-§03 distinction (matrix-input
  // signature unification, metricsum's "arrays of scalars"
  // Expression restrictions, the runtime requireMatrix guard) must
  // inspect the ORIGINAL un-flattened type, not the result of this
  // helper. See e.g. `inferMetricsum` which checks `metricT.elem`
  // directly for the strict scalar-element invariant.
  function _flattenArrayType(t: any): any {
    if (!t || t.kind !== 'array') return t;
    const dims: any[] = [...t.shape];
    let elem: any = t.elem;
    while (elem && elem.kind === 'array') {
      for (const d of elem.shape) dims.push(d);
      elem = elem.elem;
    }
    return { kind: 'array', rank: dims.length, shape: dims, elem };
  }

  // -------------------------------------------------------------------
  // aggregate(f_reduction, output_axes, expr) — spec §04 §sec:aggregate
  // -------------------------------------------------------------------
  //
  // Shape inference: the output's rank = output_axes.length; each
  // axis's length comes from the first get/get0 indexing position in
  // `expr` where the container's shape is statically known. When any
  // axis length is unknown, fall back to '%dynamic' for that
  // dimension (the rank stays known).
  //
  // Element type: the result type of `expr`. For the standard
  // arithmetic-on-reals case this is REAL; for richer cases it
  // follows whatever the body's inferred type is. The seven
  // reductions are all scalar-in / scalar-out, so the body's element
  // type passes through unchanged.

  // Shared core for `aggregate` and `metricsum` shape inference. Both
  // ops carry the same `(axes_vector, body)` shape (modulo the first arg
  // — reducer ref for aggregate, metric ref for metricsum — neither of
  // which affects output shape or element type). The function returns
  // `{ axisNames, lengths, elemT }` on success or `null` to signal a
  // deferred / non-canonical IR shape. Callers add op-specific side
  // effects (aggregate's `aggregateShape.annotate` for the runtime
  // broadcast-reduce evaluator; metricsum has none — its IR gets rewritten
  // to a fresh aggregate at lift time which gets annotated lazily by
  // `aggregateShape.getCanonical`).
  //
  // Per-axis lengths come from get/get0 indexings in the body whose
  // container's shape is statically known. A repeated axis label (e.g.
  // `A[.i, .i]` — the diagonal/trace) binds ONE length across every index
  // position per spec §04 line 853; a statically-known mismatch is a hard
  // diagnostic here (formerly first-seen-wins, which then silently read
  // out of bounds at runtime). A `%dynamic` occurrence upgrades to a later
  // concrete length. Variance markers on body axes (only legal inside
  // metricsum) don't affect length discovery — `A[.mu^]` and `A[.mu_]`
  // both index the same axis-name slot on the same stored array.
  function _inferAxisAggregateShape(axesIR: any, bodyIR: any, scopes: any): {
    axisNames: string[];
    lengths: Record<string, number | '%dynamic'>;
    elemT: any;
  } | null {
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') {
      return null;
    }
    const axisNames: string[] = [];
    for (const a of axesIR.args || []) {
      if (!a || a.kind !== 'axis') return null;
      axisNames.push(a.name);
    }
    inferExpr(bodyIR, scopes);  // populate meta.type on sub-calls

    const lengths: Record<string, number | '%dynamic'> = {};
    function walk(n: any) {
      if (!n || typeof n !== 'object') return;
      // Don't descend into a nested aggregate / metricsum — their axes
      // are in a separate scope per spec §05. Both ops' axes are
      // lexically scoped to the immediately enclosing aggregation, so
      // we treat the boundary identically regardless of which one we
      // started in.
      if (n.kind === 'call' && (n.op === 'aggregate' || n.op === 'metricsum')) return;
      if (n.kind === 'call' && (n.op === 'get' || n.op === 'get0')) {
        const innerArgs = n.args || [];
        if (innerArgs.length >= 2) {
          const container = innerArgs[0];
          // `inferRef` doesn't write `meta.type` on its way back, so
          // reading from `container.meta.type` misses refs. Re-infer
          // the container directly — cheap, side-effect-free.
          let containerT: any;
          try { containerT = inferExpr(container, scopes); }
          catch (_) { containerT = null; }
          if (containerT && containerT.kind === 'array') {
            const flat = _flattenArrayType(containerT);
            const sels = innerArgs.slice(1);
            for (let k = 0; k < sels.length; k++) {
              const s = sels[k];
              if (!s || s.kind !== 'axis') continue;
              const dim = flat.shape[k];
              const thisLen: number | '%dynamic' =
                (typeof dim === 'number') ? dim : '%dynamic';
              if (!(s.name in lengths)) {
                lengths[s.name] = thisLen;
                continue;
              }
              // Repeated axis label — spec §04 line 853 binds ONE length
              // across every index position. A statically-known mismatch
              // is a hard diagnostic (formerly first-seen-wins, which then
              // silently read out of bounds at runtime). A `%dynamic`
              // occurrence upgrades to a later concrete length; both
              // dynamic stays dynamic (runtime re-resolves).
              const prev = lengths[s.name];
              if (typeof prev === 'number' && typeof thisLen === 'number'
                  && prev !== thisLen) {
                diagnostics.push({
                  severity: 'error',
                  message: `axis '.${s.name}' is indexed at conflicting lengths `
                    + `(${prev} and ${thisLen}) — spec §04 binds one length per `
                    + `axis label, so a repeated index like [.${s.name}, .${s.name}] `
                    + `requires equal-length dimensions`,
                  loc: s.loc || (n as any).loc,
                });
              } else if (prev === '%dynamic' && typeof thisLen === 'number') {
                lengths[s.name] = thisLen;
              }
            }
          }
        }
      }
      // Recurse into children — but not into the special 'op',
      // 'name', 'ns' string fields.
      for (const k of Object.keys(n)) {
        if (k === 'loc' || k === 'kind' || k === 'op'
            || k === 'name' || k === 'ns') continue;
        const v = n[k];
        if (v && typeof v === 'object') {
          if (Array.isArray(v)) v.forEach(walk);
          else walk(v);
        }
      }
    }
    walk(bodyIR);

    // Element type: best-effort. If we have a meta.type on the body and
    // it's a scalar (or array of scalars), pass that through; else
    // default to REAL (the most common contraction result type).
    const bodyT: any = bodyIR && bodyIR.meta && bodyIR.meta.type;
    const elemT = (bodyT && (bodyT.kind === 'scalar')) ? bodyT : T.REAL;
    return { axisNames, lengths, elemT };
  }

  function inferAggregate(expr: any, scopes: any) {
    const args = expr.args || [];
    if (args.length !== 3) return T.deferred();
    const shape = _inferAxisAggregateShape(args[1], args[2], scopes);
    if (!shape) return T.deferred();
    // P1 (engine-concepts §11 / TODO): bake the canonical form onto
    // the IR so the runtime broadcast-reduce evaluator (single-point
    // AND atom-batched) doesn't re-walk the body. The annotation lives
    // on `expr.meta.aggregateCanonical`; the runtime reads it through
    // aggregate-shape.getCanonical and only re-resolves axis lengths
    // that typeinfer left as `%dynamic`. Metricsum doesn't run this
    // path — its IR gets rewritten to a fresh aggregate at lift time
    // which gets annotated lazily.
    aggregateShape.annotate(expr, shape.lengths);
    const outShape = shape.axisNames.map((n) =>
      n in shape.lengths ? shape.lengths[n] : '%dynamic');
    // Empty output_axes (spec §04 §sec:aggregate: "The bracketed axis
    // list may be empty for full reduction to a scalar") returns the
    // body's scalar element type directly — rank-0 arrays aren't a
    // distinct type in FlatPIR.
    if (shape.axisNames.length === 0) return shape.elemT;
    return T.array(shape.axisNames.length, outShape, shape.elemT);
  }

  // Metricsum type inference (spec §04 §sec:metricsum). Shape-wise
  // identical to aggregate(sum, [stripped_axes], body) — the all-upper
  // canonical storage rule means the rank-N result has the same shape
  // as the sum-aggregate contraction over the same axis lengths.
  // Delegates to `_inferAxisAggregateShape` for the shared work;
  // metricsum-specific bits stay here:
  //   1. No `aggregateShape.annotate` call — the post-lift aggregate IR
  //      is a different node and gets its annotation lazily.
  //   2. Enforces spec §sec:metricsum "Expression restrictions" via
  //      type-shape inference: metric + every variance-marked-axis-
  //      indexed container must be arrays of scalars; body must produce
  //      a scalar value. These checks are TYPE-AWARE — the engine has
  //      full shape inference at typeinfer time, so a non-scalar body
  //      or a tensor-of-tensors metric surfaces here as a parse-time
  //      diagnostic rather than a silent miscomputation at runtime.
  function inferMetricsum(expr: any, scopes: any) {
    const args = expr.args || [];
    if (args.length !== 3) return T.deferred();
    const shape = _inferAxisAggregateShape(args[1], args[2], scopes);
    if (!shape) return T.deferred();

    // ─── Static check: spec §sec:metricsum "Expression restrictions" ───
    // (a) The metric argument must be a rank-2 array of scalars. Spec
    //     §sec:metricsum: "It must be a square, symmetric, and
    //     invertible rank-2 array." Per spec §03, a nested vec-of-vec
    //     `array(1, …, array(1, …, scalar))` is NOT a matrix; the user
    //     must wrap with `rowstack(...)` to make it one. Check the
    //     UN-FLATTENED type so the vec-of-vec form fails here with a
    //     diagnostic pointing the user at the explicit lift.
    const metricT: any = inferExpr(args[0], scopes);
    if (metricT && metricT.kind !== 'failed') {
      const isDeferred = (metricT.kind === 'deferred'
                        || metricT.kind === 'any' || metricT.kind === 'var');
      if (!isDeferred) {
        if (metricT.kind !== 'array') {
          diagnostics.push({
            severity: 'error',
            message: 'metricsum: metric must be a rank-2 array of scalars, got '
              + T.show(metricT),
            loc: args[0].loc,
          });
          return T.failed('metricsum metric not array');
        }
        // Strict: metric.elem must be a scalar (NOT an array — that
        // would be a vector-of-vectors per spec §03).
        const metricElem = metricT.elem;
        const metricElemDeferred = metricElem && (metricElem.kind === 'deferred'
          || metricElem.kind === 'any' || metricElem.kind === 'var');
        if (metricElem && metricElem.kind === 'array') {
          diagnostics.push({
            severity: 'error',
            message: 'metricsum: metric is a vector-of-vectors '
              + '(' + T.show(metricT) + ') per spec §03; metricsum requires '
              + 'a rank-2 array of scalars. Wrap with `rowstack(...)` (rows = '
              + 'inner vectors) or `colstack(...)` (columns = inner vectors) '
              + 'to commit the storage-order interpretation.',
            loc: args[0].loc,
          });
          return T.failed('metricsum metric is vec-of-vec');
        }
        if (metricElem && metricElem.kind !== 'scalar' && !metricElemDeferred) {
          diagnostics.push({
            severity: 'error',
            message: 'metricsum: metric must be an array of scalars, got '
              + T.show(metricT),
            loc: args[0].loc,
          });
          return T.failed('metricsum metric not array-of-scalars');
        }
      }
    }

    // (b) Each container indexed by a variance-marked axis in the body
    // must be an array of scalars. We walk the body looking for
    // get/get0 calls whose selector list contains an axis with a
    // variance marker — those are the metricsum-specific accesses.
    let varianceContainerOK = true;
    const bodyIR = args[2];
    function checkVarianceContainers(n: any) {
      if (!n || typeof n !== 'object' || !varianceContainerOK) return;
      if (Array.isArray(n)) { for (const c of n) checkVarianceContainers(c); return; }
      // Stop at nested aggregate / metricsum (separate scope).
      if (n.kind === 'call' && (n.op === 'aggregate' || n.op === 'metricsum')) return;
      if (n.kind === 'call' && (n.op === 'get' || n.op === 'get0')) {
        const innerArgs = n.args || [];
        if (innerArgs.length >= 2) {
          const sels = innerArgs.slice(1);
          let hasVarianceAxis = false;
          for (const s of sels) {
            if (s && s.kind === 'axis' && s.variance) { hasVarianceAxis = true; break; }
          }
          if (hasVarianceAxis) {
            const container = innerArgs[0];
            let containerT: any;
            try { containerT = inferExpr(container, scopes); }
            catch (_) { containerT = null; }
            if (containerT && containerT.kind !== 'failed') {
              const cDeferred = (containerT.kind === 'deferred'
                || containerT.kind === 'any' || containerT.kind === 'var');
              if (!cDeferred && containerT.kind === 'array') {
                // Spec §sec:metricsum "Expression restrictions":
                // "metric itself and all arrays indexed with
                // co-/contravariant axis names in `expr` must be
                // arrays of scalars." Per spec §03, a vec-of-vec
                // (array-of-array) is NOT an array of scalars; check
                // the UN-FLATTENED elem directly.
                const elem = containerT.elem;
                const elemDeferred = elem && (elem.kind === 'deferred'
                  || elem.kind === 'any' || elem.kind === 'var');
                if (elem && elem.kind === 'array') {
                  diagnostics.push({
                    severity: 'error',
                    message: 'metricsum: variance-marked-axis container is a '
                      + 'vector-of-vectors (' + T.show(containerT) + ') per '
                      + 'spec §03; metricsum body containers must be flat '
                      + 'arrays of scalars. Wrap with `rowstack(...)` (rows = '
                      + 'inner vectors) or `colstack(...)` (columns = inner '
                      + 'vectors) to commit the storage-order interpretation.',
                    loc: (container && container.loc) || n.loc,
                  });
                  varianceContainerOK = false;
                  return;
                }
                if (elem && elem.kind !== 'scalar' && !elemDeferred) {
                  diagnostics.push({
                    severity: 'error',
                    message: 'metricsum: arrays indexed by a variance-marked '
                      + 'axis must have scalar elements, got ' + T.show(containerT),
                    loc: (container && container.loc) || n.loc,
                  });
                  varianceContainerOK = false;
                  return;
                }
              }
            }
          }
        }
      }
      for (const k of Object.keys(n)) {
        if (k === 'loc' || k === 'kind' || k === 'op'
            || k === 'name' || k === 'ns') continue;
        const v = n[k];
        if (v && typeof v === 'object') {
          if (Array.isArray(v)) v.forEach(checkVarianceContainers);
          else checkVarianceContainers(v);
        }
      }
    }
    checkVarianceContainers(bodyIR);
    if (!varianceContainerOK) return T.failed('metricsum tensor not array-of-scalars');

    // (c) The body must produce scalar values for all combinations of
    // axis indices. `_inferAxisAggregateShape` already called inferExpr
    // on bodyIR, so its meta.type is set. If it resolved to non-scalar
    // (record, array, measure, tuple, …), emit a diagnostic. Deferred /
    // any / failed are silently passed through (already-failed cases
    // suppressed for cascade hygiene).
    const bodyT: any = bodyIR && bodyIR.meta && bodyIR.meta.type;
    if (bodyT && bodyT.kind !== 'failed') {
      const bodyDeferred = (bodyT.kind === 'deferred'
        || bodyT.kind === 'any' || bodyT.kind === 'var');
      if (!bodyDeferred && bodyT.kind !== 'scalar') {
        diagnostics.push({
          severity: 'error',
          message: 'metricsum: body must produce a scalar value, got ' + T.show(bodyT),
          loc: bodyIR.loc,
        });
        return T.failed('metricsum non-scalar body');
      }
    }

    const outShape = shape.axisNames.map((n) =>
      n in shape.lengths ? shape.lengths[n] : '%dynamic');
    if (shape.axisNames.length === 0) return shape.elemT;
    return T.array(shape.axisNames.length, outShape, shape.elemT);
  }

  function inferVector(expr: any, scopes: any) {
    // `(call vector e1 e2 …)` — the array's length is the number of
    // arguments (statically known); the element type is the unifying
    // type of the elements. Empty vectors get an %any element type.
    //
    // Elements unify under SCALAR PROMOTION (spec §03's `booleans ⊂
    // integers ⊂ reals ⊂ complexes` lattice): `[1, 2.0]` is a real
    // vector, `[1, -3, 28]` stays integer. Strict `T.unify` would
    // refuse the first case because integer ≠ real as type constants;
    // `T.unifyArith` is the scalar-promoting variant the binary-arith
    // path already uses for `add` / `sub` / etc. Same rule across all
    // numeric contexts.
    const args = expr.args || [];
    if (args.length === 0) return T.array(1, [0], T.any());
    const elemTypes = args.map((a: any) => inferExpr(a, scopes));
    let s = new Map();
    let elem = elemTypes[0];
    for (let i = 1; i < elemTypes.length; i++) {
      // Try strict structural unify first (records / measures / etc.);
      // fall back to scalar-promoting `unifyArith` so `[1, 2.0]` lifts
      // to real and `[1, -3, 28]` stays integer. `T.unify` returns the
      // new subst Map; `T.unifyArith` returns {result, subst}.
      let nextSubst: any = T.unify(elem, elemTypes[i], s);
      let nextElem: any = elem;
      if (nextSubst == null) {
        const r: any = T.unifyArith(elem, elemTypes[i], s);
        if (r != null) {
          nextSubst = r.subst;
          nextElem = r.result;
        }
      }
      if (nextSubst == null) {
        diagnostics.push({
          severity: 'error',
          message: 'array element type mismatch: '
            + T.show(elem) + ' vs ' + T.show(elemTypes[i]),
          loc: args[i].loc || expr.loc,
        });
        return T.failed('array element mismatch');
      }
      s = nextSubst;
      elem = T.substitute(nextElem, s);
    }
    const elemT = T.substitute(elem, s);
    // Spec §04: no measure / kernel / likelihood / function inside an array.
    checkContainerElem(elemT, expr.loc, 'array', 'element');
    return T.array(1, [args.length], elemT);
  }

  // -------------------------------------------------------------------
  // Reification: functionof / kernelof / fn
  // -------------------------------------------------------------------
  //
  // Per spec §sec:functionof and §sec:kernelof:
  //   * functionof(body, kw=...) reifies body into a callable. If body
  //     is value-typed, the result is a function; if measure-typed, a
  //     kernel.
  //   * kernelof(body, kw=...) ≡ functionof(lawof(body), kw=...) —
  //     always produces a kernel; the body must be value-typed.
  //   * fn(body) lowers to functionof with placeholder parameters
  //     extracted from the body's holes.
  //
  // The function's parameters carry the type of their boundary. For a
  // placeholder boundary (`par = _par_`), the parameter's type is %any
  // (the placeholder is `elementof(anything)` per spec). For an
  // elementof-bound boundary (`par = _some_elementof`), it's the value
  // type of that elementof's set. For a stochastic-bound boundary
  // (`theta1 = theta1`), the parameter type is the boundary expression's
  // structural type — the spec says boundaries are substituted with
  // `elementof(valueset(boundary))` whose value type follows the
  // boundary's domain.

  // ---- Numeric arithmetic with shape polymorphism -------------------
  //
  // Binary: scalar+scalar → scalar; matching-shape arrays elementwise;
  // scalar/array broadcast. Unary: shape-preserving. Comparisons
  // produce boolean of the broadcast shape.

  // broadcast(fn, A1, A2, …) — propagates the broadcast shape of the
  // data args AND the function's per-cell result type. The function
  // arg supplies the elementwise op (value-fn or kernel constructor;
  // see spec §04 higher-order).
  //
  // Per-arg classification distinguishes the spec's two array kinds:
  //
  //   - Flat tensor: `array(rank=k, shape=[...], elem=scalar)`.
  //     All k axes are loop axes; per cell the callable sees a
  //     scalar. `rowstack(...)` matrices, literal `[1, 2, 3]` flat
  //     vectors, broadcast results.
  //
  //   - Nested vector: `array(rank=1, shape=[N], elem=array(...))`.
  //     The outer rank-1 axis is the loop axis; per cell the
  //     callable receives the inner array WHOLE. This is the
  //     Ref-wrap idiom in the type system: `[C]` where `C` is a
  //     vector lifts to `array(1, [1], array(1, [k], real))`.
  //     §03 spec: "Vectors of vectors are not interpreted as
  //     matrices implicitly" — the two types are genuinely
  //     distinct.
  //
  // Outer-shape unification follows the spec rule: all collection
  // args must have the same OUTER rank; per axis sizes must be
  // equal or 1 (singleton broadcast).
  //
  // Two callable shapes the lowerer produces:
  //   - synthetic `functionof` (from dotted operators / `fn(...)`):
  //     re-infer the body with per-cell types in scope.
  //   - bare ref to a user-defined callable (`f.(args)` →
  //     `broadcast(f, args...)` keeps the ref): use the callable's
  //     declared result type, monomorphic-at-definition (the same
  //     simplification `inferUserCall` makes for scalar calls).
  // Anything else falls back to deferred.
  // For a BARE measure-producing builtin head under `broadcast`
  // (e.g. `Normal.(means, sigmas)`, `Binomial.(n, p)`), the per-cell
  // result is the head applied to the per-cell arg types. A bare
  // distribution constructor IS a (Markov) kernel (spec §06 uniform
  // kernel extension, §04 functionof-of-measure; engine-concepts §19),
  // so broadcasting it yields an array-valued measure exactly like a
  // user kernel does. Returns the per-cell measure type, or null if
  // `opName` is not a measure-producing builtin. Best-effort tvar
  // binding (e.g. `Dirac(value=T)`); shape/type mismatches in the cell
  // args are not diagnosed here (the result type is all we need).
  function inferMeasureHeadCellResult(opName: any, cellTypes: any): any {
    const sig: any = T.signatureOf(opName);
    if (!sig || !T.isMeasure(sig.result)) return null;
    let s = new Map();
    // Distributions are kwargs-only (sig.args === null) but accept
    // positional binding (spec §05); map the per-cell arg types onto the
    // declared params in order.
    const params: any[] = (sig.args && sig.args.length)
      ? sig.args
      : (sig.kwargs ? Object.keys(sig.kwargs).map((k) => sig.kwargs[k]) : []);
    const m = Math.min(params.length, cellTypes.length);
    for (let i = 0; i < m; i++) {
      const next = T.unify(params[i], cellTypes[i], s);
      if (next != null) s = next;
    }
    return T.substitute(sig.result, s);
  }

  function inferBroadcast(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length < 2) return T.deferred();

    // Phase 1: infer each data arg's type.
    const dataTypes: any[] = [];
    for (let i = 1; i < args.length; i++) {
      const t = inferExpr(args[i], scopes);
      if (t && t.kind === 'failed') return T.failed('broadcast cascade');
      dataTypes.push(t);
    }

    // Phase 2: per-arg classification — collection (outer shape +
    // cell type) or held-constant.
    function classifyArg(t: any): any {
      if (t && t.kind === 'array') {
        if (t.elem && t.elem.kind === 'array') {
          // Nested vector: outer rank-1 axis is the loop axis; the
          // inner array is what the callable sees per cell.
          return { collection: true, outerShape: [t.shape[0]], cellType: t.elem };
        }
        // Flat tensor: all axes are loop axes; cell type = scalar elem.
        return { collection: true, outerShape: t.shape.slice(), cellType: t.elem };
      }
      if (t && t.kind === 'table') {
        // Spec §03: "When a table is passed to broadcast, it is
        // traversed row-wise and each row treated as a record passed
        // to the function used in the broadcast." Outer axis = nrows;
        // per cell the callable sees a record over the table's
        // columns.
        return {
          collection: true,
          outerShape: [t.nrows],
          cellType: T.record(t.columns),
        };
      }
      if (t && t.kind === 'scalar') {
        return { collection: false, cellType: t };
      }
      if (t && (t.kind === 'any' || t.kind === 'deferred')) {
        // Common inside a `functionof` body where a boundary has no
        // declared type — the param is `any`, and downstream broadcasts
        // over it still need to flow some cell type to the callable's
        // body inference (signature lookups can tighten the result).
        // Treat as held-constant; cellType propagates through.
        return { collection: false, cellType: t };
      }
      // measure / failed / tuple / record (records are explicitly
      // disallowed as broadcast inputs per spec §04) / etc.
      return null;
    }
    const argClassif: any[] = [];
    for (const t of dataTypes) {
      const c = classifyArg(t);
      if (c == null) return T.deferred();
      argClassif.push(c);
    }

    // Phase 3: outer-shape unification across collection args.
    // Spec §04: same OUTER rank required; per-axis sizes must be
    // equal or 1 (singleton broadcast).
    let outerShape: any[] | null = null;
    let hasCollection = false;
    for (const c of argClassif) {
      if (!c.collection) continue;
      hasCollection = true;
      if (outerShape === null) {
        outerShape = c.outerShape.slice();
        continue;
      }
      if (c.outerShape.length !== outerShape.length) {
        // Rank mismatch — spec violation. Stay deferred so the
        // runtime evaluator surfaces the precise error at the
        // broadcast call site.
        return T.deferred();
      }
      const merged: any[] = [];
      for (let a = 0; a < outerShape.length; a++) {
        const ai = outerShape[a], bi = c.outerShape[a];
        // Singleton expansion takes precedence over %dynamic: a size-1
        // axis expands to the OTHER axis's length, even when that length
        // is dynamic. (Checking %dynamic first would wrongly collapse
        // `%dynamic ∧ 1` to 1 — e.g. `transport.(xs, [pars])` with xs of
        // length n=%dynamic and a singleton [pars] must stay %dynamic,
        // not become length 1.)
        if (ai === 1) merged.push(bi);
        else if (bi === 1) merged.push(ai);
        else if (ai === '%dynamic') merged.push(bi);
        else if (bi === '%dynamic') merged.push(ai);
        else if (ai === bi) merged.push(ai);
        else return T.deferred();
      }
      outerShape = merged;
    }

    // Phase 4: resolve cell-result type via the callable.
    const cellTypes = argClassif.map((c) => c.cellType);
    const fn = args[0];
    let elem: any = T.deferred();
    if (fn && fn.kind === 'call' && fn.op === 'functionof'
        && fn.body && Array.isArray(fn.params)) {
      const localScope = new Map<string, any>();
      for (let i = 0; i < fn.params.length && i < cellTypes.length; i++) {
        localScope.set(fn.params[i], cellTypes[i]);
      }
      elem = inferExpr(fn.body, scopes.concat([localScope]));
    } else if (fn && fn.kind === 'ref' && fn.ns === 'self') {
      const calleeType: any = inferBinding(fn.name);
      if (T.isCallable(calleeType)) {
        elem = calleeType.result;
      } else {
        // Not a user-defined callable binding. A bare builtin
        // measure-producing head (`Normal`, `Binomial`, …) shadows to
        // `failed`/non-callable here; treat it as the kernel it is and
        // infer the head applied to the per-cell arg types, so the
        // measure-wrap below tightens `Normal.(…)` to an array-valued
        // measure (spec §04/§06; engine-concepts §19). This is also what
        // lets a `(n,p) -> Binomial.(n,p)` lambda reify to a kernel:
        // its body's broadcast now types as a measure, so inferReification
        // makes the lambda a kernelType rather than a function.
        const headResult = inferMeasureHeadCellResult(fn.name, cellTypes);
        if (headResult) elem = headResult;
        else return T.deferred();
      }
    } else {
      // Computed/dynamic head, or anything not matching the forms above
      // — conservative defer (downstream consumers accept a deferred
      // broadcast result via the passthrough).
      return T.deferred();
    }
    if (elem && elem.kind === 'failed') return T.failed('broadcast cascade');

    // Kernel-broadcast result type (spec §04 "broadcast(kernel, ...)
    // returns an array-valued measure: the independent product measure
    // of the kernel applications at each array position"). At the type
    // level: a measure over an array whose shape is the broadcast outer
    // shape and whose element type is the kernel's per-cell variate
    // type. Mirrors how iid is typed (see inferIid).
    //
    // P2 — populate the three-shape decomposition. The outer broadcast
    // shape is `batchShape` (conditionally-independent cells, NOT iid
    // replicates per Pyro/TFP convention); the inner kernel's
    // `eventShape` carries through; sampleShape stays empty.
    if (elem && elem.kind === 'measure') {
      if (!hasCollection) return elem;   // no outer axis ⇒ single call
      const batchShape = outerShape!.slice();
      const eventShape = Array.isArray(elem.eventShape) ? elem.eventShape.slice() : [];
      const sampleShape = Array.isArray(elem.sampleShape) ? elem.sampleShape.slice() : [];
      return T.measure(
        T.array(outerShape!.length, outerShape!.slice(), elem.domain),
        { sampleShape, batchShape, eventShape });
    }

    // Phase 5: combine for the value-broadcast (non-kernel) case.
    const concrete = elem && elem.kind !== 'deferred' && elem.kind !== 'any';
    if (!hasCollection) {
      // No collection args ⇒ single call; result = cell-result.
      return concrete ? elem : T.deferred();
    }
    // Default to real for unknown cell types — broadcast is by design
    // numeric, and downstream consumers (viewer plot-routing, materialise
    // shape checks) need a concrete elem to dispatch on. This matches the
    // assumption every other broadcast path in the engine already makes.
    const elemFinal = concrete ? elem : T.REAL;
    return T.array(outerShape!.length, outerShape!.slice(), elemFinal);
  }

  function inferArith2(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 2) return arityError(expr.op, 2, args.length, expr.loc);
    const aT: any = inferExpr(args[0], scopes);
    const bT: any = inferExpr(args[1], scopes);
    if (aT.kind === 'failed' || bT.kind === 'failed') return T.failed(expr.op + ' cascade');
    const r: any = T.unifyArith(aT, bT, new Map());
    if (r == null) {
      diagnostics.push({
        severity: 'error',
        message: expr.op + ': operand types ' + T.show(aT) + ' and '
          + T.show(bT) + ' are not numerically compatible',
        loc: expr.loc,
      });
      return T.failed(expr.op + ' shape mismatch');
    }
    return r.result;
  }

  function inferArith1(expr: any, scopes: any) {
    const args = expr.args || [];
    if (args.length !== 1) return arityError(expr.op, 1, args.length, expr.loc);
    const aT = inferExpr(args[0], scopes);
    if (aT.kind === 'failed') return T.failed(expr.op + ' cascade');
    // Scalar in → scalar out; array in → array out (shape preserved).
    // Per-op element-type rule:
    //   - INT_CAST  (floor/ceil/round): real → integer.
    //   - PRESERVING (neg/pos/abs/abs2): preserve integer/real
    //     unchanged; complex → real for abs/abs2 (modulus is real-
    //     valued), preserve for neg/pos (negation/identity of a
    //     complex is complex). Spec §03's scalar hierarchy
    //     (booleans ⊂ integers ⊂ reals): unary negation of an
    //     integer is mathematically an integer. Widening to real
    //     here (the pre-2026-05-29 behaviour) made `-3` mismatch
    //     `28` in an array-literal element unification — see
    //     commit 5c889a5's literal-fold workaround. With the
    //     element-type preserved we no longer need that workaround
    //     to clear the unification.
    //   - REAL_ONLY (exp/log/log10/sqrt/sin/cos): always real.
    const isIntCast    = (expr.op === 'floor' || expr.op === 'ceil' || expr.op === 'round');
    const isPreserving = (expr.op === 'neg' || expr.op === 'pos'
                          || expr.op === 'abs' || expr.op === 'abs2');
    function elemTypeFor(scalarPrim: string): any {
      if (isIntCast) return T.INTEGER;
      if (isPreserving) {
        if (scalarPrim === 'integer') return T.INTEGER;
        if (scalarPrim === 'complex') {
          // abs/abs2 of complex → real; neg/pos of complex → complex.
          return (expr.op === 'abs' || expr.op === 'abs2') ? T.REAL : T.COMPLEX;
        }
        return T.REAL;
      }
      return T.REAL;
    }
    function liftElemwise(t: any): any {
      if (t.kind === 'scalar') return elemTypeFor(t.prim);
      if (t.kind === 'array')  return T.array(t.rank, t.shape.slice(),
                                              liftElemwise(t.elem));
      if (t.kind === 'deferred' || t.kind === 'any') return t;
      return null;
    }
    const out = liftElemwise(aT);
    if (out == null) {
      diagnostics.push({
        severity: 'error',
        message: expr.op + ': operand must be numeric (scalar or array of scalars), got ' + T.show(aT),
        loc: args[0].loc,
      });
      return T.failed(expr.op + ' bad operand');
    }
    return out;
  }

  function inferComparison(expr: any, scopes: any): any {
    // Comparisons unify operand shapes via unifyArith and return
    // boolean of that shape. `equal(scalar, array)` would broadcast;
    // `equal(scalar, scalar)` → boolean.
    const args = expr.args || [];
    if (args.length !== 2) return arityError(expr.op, 2, args.length, expr.loc);
    const aT: any = inferExpr(args[0], scopes);
    const bT: any = inferExpr(args[1], scopes);
    if (aT.kind === 'failed' || bT.kind === 'failed') return T.failed(expr.op + ' cascade');
    const r: any = T.unifyArith(aT, bT, new Map());
    if (r == null) {
      diagnostics.push({
        severity: 'error',
        message: expr.op + ': operand types ' + T.show(aT) + ' and '
          + T.show(bT) + ' are not comparable',
        loc: expr.loc,
      });
      return T.failed(expr.op + ' shape mismatch');
    }
    // Result is boolean with the same shape as r.result.
    if (r.result.kind === 'array') {
      return T.array(r.result.rank, r.result.shape.slice(), T.BOOLEAN);
    }
    return T.BOOLEAN;
  }

  // ---- iid: shape-aware `iid(M, n)` ---------------------------------
  //
  // Per spec §sec:iid: produces measure<array<1, [n], M.domain>> when
  // n is statically known; %dynamic otherwise. We resolve n via
  // literal and binding-ref folding so common cases (`n = 10;
  // iid(M, n)`) yield concrete shapes for downstream shape checks.
  // pushfwd(f, M) — pushforward of measure / kernel M through fn f
  // (spec §06). Result type tracks arg 2:
  //   - M a measure → result is a measure (over f's codomain).
  //   - M a kernel  → result is a kernel (same kernel inputs, new
  //                   codomain measure).
  //   - otherwise   → defer (no specific check; downstream
  //                   classifier validates structural shape).
  //
  // The variate type of f's codomain isn't statically tracked
  // here — defaults to real, which covers the common case (most
  // pushfwd uses produce real-valued measures: log-Cauchy, Pareto-
  // via-exp, affine-transformed Normals). Per-call bijection
  // annotations could later refine this; the classifier validates
  // the f / M shapes structurally at routing time.
  // A surviving non-scalar `locscale(base, shift, scale)` node (the
  // analyzer pre-pass leaves vector/matrix forms unexpanded so they reach
  // lift's affine-registry routing). An affine pushforward preserves the
  // base measure's type (domain + shape), so the result type is the base
  // measure's type. Modeled on inferPushfwd.
  function inferLocscale(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 3) return arityError('locscale', '3', args.length, expr.loc);
    const baseT = inferExpr(args[0], scopes);  // base measure
    inferExpr(args[1], scopes);                // shift — infer so refs resolve
    inferExpr(args[2], scopes);                // scale — infer so refs resolve
    // An affine pushforward preserves the base measure's type (domain +
    // shape), so the result type IS the base measure's type. When the base
    // doesn't infer to a clean measure (e.g. `iid(Normal, D)` with a bare
    // `Normal` distribution-symbol infers `failed`, exactly as it does when
    // used directly — see inferIid), DEFER rather than fabricating a scalar
    // `measure(real)`: a scalar default would wrongly drive the density
    // shape check to expect a real point for a vector-variate locscale.
    if (T.isMeasure(baseT)) return baseT;
    return T.deferred();
  }

  function inferPushfwd(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 2) return arityError('pushfwd', '2', args.length, expr.loc);
    // arg 0 is the function; we don't statically check its type
    // here (callable-type tracking is the orchestrator's job).
    inferExpr(args[0], scopes);
    const m2 = inferExpr(args[1], scopes);
    if (T.isMeasure(m2)) return T.measure(T.REAL);
    if (m2 && m2.kind === 'kernel') {
      // Preserve the kernel's input signature; only the output
      // measure's variate type changes.
      return { kind: 'kernel', inputs: m2.inputs || {}, output: T.measure(T.REAL) };
    }
    if (m2 && m2.kind === 'failed') return T.failed('pushfwd cascade');
    // Permissive default — pushfwd OUTSIDE measure/kernel context
    // (e.g. inside a `fn(...)` body whose arg-types haven't been
    // resolved yet) defers rather than erroring.
    return T.deferred();
  }

  function inferIid(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length < 2) return arityError('iid', '≥2', args.length, expr.loc);
    const measureT: any = inferExpr(args[0], scopes);
    if (!T.isMeasure(measureT)) {
      if (measureT && measureT.kind === 'failed') return T.failed('iid cascade');
      diagnostics.push({
        severity: 'error',
        message: 'iid: arg 1 expects a measure, got ' + T.show(measureT),
        loc: args[0].loc,
      });
      return T.failed('iid bad measure');
    }
    // Resolve dimensions: walk each n arg and try to fold to an
    // integer literal. Refs to integer-typed bindings remain
    // %dynamic for now (full constant folding via resolveConstant
    // could be promoted from orchestrator if we want).
    //
    // New spec: arg 1 may be a single positive int OR a vector of
    // positive ints. The vector-literal form (lowered as `vector(...)`)
    // is unpacked into per-axis dims here so downstream shape checks
    // see concrete dims. A non-literal vector falls through to a
    // single dynamic-rank axis (best we can do statically).
    const dims: any[] = [];
    let dimArgs = args.slice(1);
    if (dimArgs.length === 1
      && dimArgs[0].kind === 'call'
      && dimArgs[0].op === 'vector'
      && Array.isArray(dimArgs[0].args)) {
      dimArgs = dimArgs[0].args;
    }
    for (let i = 0; i < dimArgs.length; i++) {
      const arg = dimArgs[i];
      const dT = inferExpr(arg, scopes);
      // Type-check: each dim must be integer-promotable. Allow an
      // integer-array second arg too (the non-literal vector case).
      const s = T.unify(T.INTEGER, dT, new Map());
      if (s == null) {
        // Permit `iid(M, sizes)` where `sizes` is an integer array.
        if (dimArgs.length === 1 && dT && dT.kind === 'array') {
          dims.push('%dynamic');
          continue;
        }
        diagnostics.push({
          severity: 'error',
          message: 'iid: dim ' + (i + 1) + ' expects integer, got ' + T.show(dT),
          loc: arg.loc,
        });
        return T.failed('iid bad dim');
      }
      const resolved = resolveIntegerShape(arg);
      dims.push(resolved != null ? resolved : '%dynamic');
    }
    const rank = dims.length;
    // P2 — populate the sample-shape decomposition. iid prepends its
    // own dims to the inner measure's sampleShape (so nested iid
    // stacks: `iid(iid(M, 5), 3)` carries sampleShape = [3, 5]).
    // batchShape and eventShape pass through from the inner measure.
    const innerSample = Array.isArray(measureT.sampleShape) ? measureT.sampleShape : [];
    const sampleShape = dims.concat(innerSample);
    const batchShape  = Array.isArray(measureT.batchShape)  ? measureT.batchShape.slice()  : [];
    const eventShape  = Array.isArray(measureT.eventShape)  ? measureT.eventShape.slice()  : [];
    return T.measure(
      T.array(rank, dims, measureT.domain),
      { sampleShape, batchShape, eventShape });
  }

  // Resolve a shape-position IR expression to a non-negative integer
  // VECTOR if possible. Used by `fill`/`zeros`/`ones` whose single
  // arg may be an integer (rank-1 result) or an integer vector
  // (rank-N result, one dim per element). Tries literal forms first;
  // falls back to the resolver. Returns null if neither works.
  // Demand-driven fixed-value boundary (engine-concepts §17.4): a shape
  // position asked the resolver for a value and every input resolved (so
  // the computation is fixed-phase and computable in principle), but an
  // operation in it isn't implemented in simple-eval mode — so the shape
  // can't be folded. Per the demand-driven contract this is a hard error
  // (the value is genuinely needed for inference and should be
  // computable), NOT a silent %dynamic. A value that's merely
  // not-statically-known (external / elementof / draw / load_data) comes
  // back as `undefined`, never as UNSUPPORTED, and legitimately stays
  // %dynamic.
  function _shapeValueUncomputable(ir: any): null {
    const where = (ir && ir.kind === 'ref' && typeof ir.name === 'string')
      ? `the value of '${ir.name}'`
      : 'a value';
    diagnostics.push({
      severity: 'error',
      message: `could not compute ${where}, needed here for type/shape `
        + `inference — it uses an operation not supported in fixed-phase `
        + `(simple) evaluation`,
      loc: (ir && ir.loc) || undefined,
    });
    return null;
  }

  function resolveIntegerVectorShape(ir: any): number[] | null {
    // Literal integer → rank-1 of that length.
    const litInt = literalIntFromIR(ir);
    if (litInt != null && litInt >= 0) return [litInt];
    // Literal vector of integers → use as multi-dim shape directly.
    if (ir && ir.kind === 'call' && ir.op === 'vector'
        && Array.isArray(ir.args)) {
      const dims: number[] = [];
      for (const a of ir.args) {
        const d = literalIntFromIR(a);
        if (d == null || d < 0) return null;
        dims.push(d);
      }
      return dims;
    }
    // Resolver fallback. Result is either a number (rank-1 result)
    // or a shape-explicit Value carrying the dims (rank-N).
    if (!resolveFixed) return null;
    const v = resolveFixed(ir);
    if (v === (resolveFixed as any).UNSUPPORTED) { _shapeValueUncomputable(ir); return null; }
    if (typeof v === 'number' && Number.isInteger(v) && v >= 0) return [v];
    if (v && typeof v === 'object'
        && Array.isArray(v.shape) && v.shape.length === 1
        && v.data && typeof v.data.length === 'number') {
      const dims: number[] = [];
      for (let i = 0; i < v.data.length; i++) {
        const d = v.data[i] | 0;
        if (d < 0) return null;
        dims.push(d);
      }
      return dims;
    }
    return null;
  }

  // fill(value, shape) — result element type matches value's type;
  // result shape from the shape arg (integer → rank-1; vector → rank-N).
  function inferFill(expr: any, scopes: any) {
    const args = expr.args || [];
    if (args.length !== 2) return arityError('fill', 2, args.length, expr.loc);
    const vT = inferExpr(args[0], scopes);
    const dims = resolveIntegerVectorShape(args[1]);
    const elem = (vT && vT.kind === 'scalar') ? vT : T.REAL;
    if (dims == null) return T.deferred();
    return T.array(dims.length, dims, elem);
  }

  // zeros(shape) / ones(shape) — real-valued; same shape rule as fill.
  function inferZerosOnes(expr: any, scopes: any) {
    const args = expr.args || [];
    if (args.length !== 1) return arityError(expr.op, 1, args.length, expr.loc);
    const dims = resolveIntegerVectorShape(args[0]);
    if (dims == null) return T.deferred();
    return T.array(dims.length, dims, T.REAL);
  }

  // eye(n) / eye(n=n) — n×n identity matrix.
  function inferEye(expr: any, scopes: any) {
    const args = expr.args || [];
    const kwargs = expr.kwargs || {};
    const sizeIR = (args.length > 0) ? args[0]
      : (kwargs.n != null ? kwargs.n : null);
    if (!sizeIR) return arityError('eye', 1, args.length, expr.loc);
    const n = resolveIntegerShape(sizeIR);
    if (n == null) return T.array(2, ['%dynamic', '%dynamic'], T.REAL);
    return T.array(2, [n, n], T.REAL);
  }

  // Density-op shape check: run the type-mode consume/rest walker
  // over (measure-IR, variate-type) for logdensityof / densityof /
  // bayesupdate / likelihoodof calls. Surfaces shape mismatches as
  // parse-time diagnostics. Engine-concepts §17.3 — the type-mode
  // counterpart to density.ts's runtime empty-rest invariant.
  //
  // Op shapes:
  //   logdensityof(M, x)        — args[0]=M, args[1]=x
  //   densityof(M, x)            — args[0]=M, args[1]=x
  //   likelihoodof(K, x)         — args[0]=K (kernel), args[1]=x;
  //                                 we treat K's signature like M
  //   bayesupdate(L, prior)      — args[0]=L (likelihood object),
  //                                 args[1]=prior. The posterior's
  //                                 variate is just the prior's
  //                                 variate; the (kernel, observation)
  //                                 shape compatibility is checked at
  //                                 the INNER `likelihoodof(K, x)`
  //                                 call site (when L is built that
  //                                 way), so bayesupdate adds nothing
  //                                 new to the static check.
  function _checkDensityShapes(expr: any): void {
    if (expr.op === 'bayesupdate') return;   // see comment above
    const args = expr.args || [];
    if (args.length < 2) return;
    const mIR = args[0];
    const xT = inferExpr(args[1], []);
    if (!xT || xT.kind === 'failed' || xT.kind === 'deferred' || xT.kind === 'any') {
      return;   // can't statically determine variate type
    }
    // Resolve measure IR. If args[0] is a self-ref, look up the
    // binding's RHS (the actual measure expression). Otherwise use
    // as-is.
    const measureIR = _resolveMeasureIR(mIR);
    if (!measureIR) return;   // unresolved ref; defer to runtime
    const densityPrims = require('./density-prims.ts');
    const err = densityPrims.staticDensityShapeCheck(measureIR, xT);
    if (err) {
      diagnostics.push({
        severity: 'error',
        message: expr.op + ': ' + err,
        loc: expr.loc,
      });
    }
  }

  // For a self-ref to a binding, return that binding's RHS IR (the
  // measure expression). For an inline measure expression, return
  // it as-is. Returns null when the ref can't be resolved.
  function _resolveMeasureIR(ir: any): any {
    if (!ir) return null;
    if (ir.kind === 'ref' && ir.ns === 'self') {
      const b = loweredModule.bindings.get(ir.name);
      return (b && b.rhs) || null;
    }
    return ir;
  }

  // rowstack(vector_of_vectors) / colstack(vector_of_vectors). When
  // the input is an inline vector(vector(...), vector(...), ...)
  // literal, both dims are static — emit the precise [m, n] shape.
  // (`colstack` swaps which dim is rows vs cols but the *type-level*
  // shape is the same since both axes are dense real.)
  function inferRowstack(expr: any, scopes: any) {
    const args = expr.args || [];
    if (args.length !== 1) return T.deferred();
    const outerIR = args[0];
    // Inline literal? Read m = outer length, n = inner length(0).
    if (outerIR && outerIR.kind === 'call' && outerIR.op === 'vector'
        && Array.isArray(outerIR.args)) {
      const m = outerIR.args.length;
      if (m === 0) return T.array(2, [0, 0], T.REAL);
      const first = outerIR.args[0];
      if (first && first.kind === 'call' && first.op === 'vector'
          && Array.isArray(first.args)) {
        const n = first.args.length;
        // Confirm every row has the same length statically.
        for (let i = 1; i < m; i++) {
          const row = outerIR.args[i];
          if (!row || row.kind !== 'call' || row.op !== 'vector'
              || !Array.isArray(row.args) || row.args.length !== n) {
            return T.array(2, ['%dynamic', '%dynamic'], T.REAL);
          }
        }
        return T.array(2, [m, n], T.REAL);
      }
    }
    // Non-literal outer: fall back to %dynamic (the existing
    // SIGNATURE_FACTORIES behaviour).
    return T.array(2, ['%dynamic', '%dynamic'], T.REAL);
  }

  // onehot(i, n) — length-n vector with a 1 at position i, 0
  // elsewhere. The result shape's length comes from n; the value of
  // i doesn't affect the type.
  function inferOnehot(expr: any, scopes: any) {
    const args = expr.args || [];
    const kwargs = expr.kwargs || {};
    const nIR = (args.length >= 2) ? args[1]
      : (kwargs.n != null ? kwargs.n : null);
    if (!nIR) return arityError('onehot', 2, args.length, expr.loc);
    const n = resolveIntegerShape(nIR);
    if (n == null) return T.array(1, ['%dynamic'], T.REAL);
    return T.array(1, [n], T.REAL);
  }

  function literalIntFromIR(ir: any) {
    if (!ir) return null;
    if (ir.kind === 'lit' && ir.numType === 'integer') return ir.value;
    if (ir.kind === 'lit' && Number.isInteger(ir.value)) return ir.value;
    return null;
  }

  // Resolve a shape-position IR expression to a non-negative integer
  // if possible. Tries the cheap literal-extract first; falls back to
  // the resolver callback (set up via fixed-eval.makeResolver) when
  // const-eval is enabled by the caller. Per engine-concepts §17.4:
  // invoked ONLY at known shape positions, never at general
  // sub-expressions — and the source IR is left intact regardless.
  //
  // Includes the shape-only short-circuit for `length(x)` / `sizeof(x)`:
  // when the argument's inferred type already carries a literal shape,
  // we read it directly without invoking the resolver (avoids
  // materialising large arrays just to ask their length).
  function resolveIntegerShape(ir: any): number | null {
    const lit = literalIntFromIR(ir);
    if (lit != null) return lit;
    // Shape-only short-circuit: `length(x)` / `lengthof(x)` reads
    // the leading-axis length from x's inferred type. Works on ANY
    // x — ref, inline call, anything inferExpr can give a type for.
    // The principle the engine-concepts §17.4 design was built for
    // — most common shape-determining expression in real models, and
    // the only safe way to chain through expensive intermediates.
    // (`sizeof(x)` returns the shape *vector* and is handled by
    // resolveIntegerVectorShape, not here.)
    if (ir && ir.kind === 'call'
        && (ir.op === 'length' || ir.op === 'lengthof')
        && Array.isArray(ir.args) && ir.args.length === 1) {
      const argT: any = inferExpr(ir.args[0], []);
      if (argT && argT.kind === 'array'
          && Array.isArray(argT.shape) && argT.shape.length > 0
          && typeof argT.shape[0] === 'number') {
        return argT.shape[0];
      }
    }
    // Fall back to the caller-supplied resolver (typically wired by
    // analyzer.ts via fixed-eval.makeResolver). Without a resolver
    // we conservatively report "unknown" and the type stays
    // %dynamic — same behaviour as before the const-eval pass.
    if (!resolveFixed) return null;
    const v = resolveFixed(ir);
    if (v === (resolveFixed as any).UNSUPPORTED) return _shapeValueUncomputable(ir);
    if (typeof v === 'number' && Number.isInteger(v) && v >= 0) return v;
    return null;
  }

  function inferReification(expr: any, scopes: any): any {
    // Only `functionof` reaches here — kernelof and fn are lowered
    // to functionof by lower.js. The kernelof spec rule "x must not
    // be a measure" emerges naturally from the lawof inside the
    // lowered form (lawof requires a value-typed argument); we don't
    // need a special case here.
    const params      = expr.params      || [];   // scope-local names
    const paramKwargs = expr.paramKwargs || [];   // surface kwarg names
    const newScope = new Map();
    for (let i = 0; i < params.length; i++) {
      let paramType = T.any();
      const kwName = paramKwargs[i];
      if (kwName && expr.kwargs && expr.kwargs[kwName]) {
        const boundaryT = inferExpr(expr.kwargs[kwName], scopes);
        if (T.isMeasure(boundaryT)) {
          diagnostics.push({
            severity: 'error',
            message: 'functionof boundary "' + kwName
              + '" must be a value, got ' + T.show(boundaryT),
            loc: expr.kwargs[kwName].loc || expr.loc,
          });
          paramType = T.failed('functionof boundary type');
        } else if (T.isValue(boundaryT)) {
          paramType = boundaryT;
        }
      }
      newScope.set(params[i], paramType);
    }

    const innerScopes = scopes.concat([newScope]);
    const bodyT: any = expr.body ? inferExpr(expr.body, innerScopes) : T.deferred();
    // Inputs use the *surface* keyword name from paramKwargs — that's
    // what call-site kwargs bind to. Types come from the scope.
    const inputs = params.map((p: any, i: any) => ({
      name: paramKwargs[i] || p,
      type: newScope.get(p),
    }));

    // Per spec §sec:functionof-measure: a functionof with a measure
    // body produces a kernel; with a value body, a function.
    if (T.isMeasure(bodyT))      return T.kernelType(inputs, bodyT);
    if (T.isValue(bodyT))        return T.funcType(inputs,   bodyT);
    if (bodyT.kind === 'failed') return T.failed('functionof cascade');
    return T.deferred();
  }

  // -------------------------------------------------------------------
  // User-defined call: callee is a (%ref self <fn-name>)
  // -------------------------------------------------------------------

  function inferUserCall(expr: any, scopes: any): any {
    const head = expr.target;
    if (!head || head.ns !== 'self') {
      // Cross-module user calls — not yet implemented.
      return write(T.deferred(), expr);
    }

    // `broadcasted(f)` wrapper recognition. The lift rewrites
    // `bc(args)` (where bc = broadcasted(f)) to `broadcast(f, args)`
    // at runtime materialisation time — but that lift runs AFTER
    // typeinfer, so the call's IR still says `target=bc, args=...`.
    // Without this hook, the call types as deferred (callee's
    // inferred type is deferred — broadcasted(f) has no signature
    // and shouldn't, since the callable is fully polymorphic).
    //
    // Route the call through `inferBroadcast` directly. The wrapper
    // is just sugar at the type level: `broadcasted(f)(args)` ≡
    // `broadcast(f, args)` per spec §04.
    //
    // Direct form `broadcasted(f)(args)` (no via-binding) is NOT
    // handled here — it fails to lower (non-Identifier callee in
    // lower.ts) and stores as a lit-null placeholder. Use the
    // via-binding form `bc = broadcasted(f); bc(args)` to get
    // type-level routing.
    const b = loweredModule.bindings.get(head.name);
    if (b && b.rhs && b.rhs.kind === 'call' && b.rhs.op === 'broadcasted'
        && b.rhs.args && b.rhs.args.length === 1) {
      const f = b.rhs.args[0];
      const broadcastIR: any = {
        kind: 'call', op: 'broadcast',
        args: [f].concat(expr.args || []),
        kwargs: expr.kwargs,
        loc: expr.loc,
      };
      return write(inferBroadcast(broadcastIR, scopes), expr);
    }

    const calleeType: any = inferBinding(head.name);
    if (!T.isCallable(calleeType)) {
      // Cascade silently when the callee already failed or is still
      // deferred (couldn't infer its type — e.g. unknown built-in,
      // standard module function not yet typed). Only error when we
      // positively know it's a non-callable (scalar / measure / etc.).
      if (calleeType && (calleeType.kind === 'failed' || calleeType.kind === 'deferred'
                         || calleeType.kind === 'any')) {
        return write(T.deferred(), expr);
      }
      diagnostics.push({
        severity: 'error',
        message: '"' + head.name + '" is not callable (got ' + T.show(calleeType) + ')',
        loc: expr.loc,
      });
      return write(T.failed('not callable'), expr);
    }

    // For now: take the callee's `result` directly. This is the
    // "monomorphic-at-definition" simplification. Once we add full
    // polymorphism, we'd traverse the callee's body with the call
    // site's actual argument types.
    //
    // We DO type-check the call args against the callee's input
    // types — that catches passing wrong-typed values to functions.
    const inputs = calleeType.inputs;
    let args   = expr.args   || [];
    let kwargs = expr.kwargs || {};

    // Auto-splatting (spec §sec:calling-convention lines 99-102):
    // a single positional record argument is equivalent to passing
    // each field as a kwarg. We detect this when the call has
    // exactly one positional arg of record type whose field names
    // are a subset of the callee's input names; the splat replaces
    // the positional list with a kwarg map for the type-check below.
    // This lets the spec's `f(record(a=x, b=y))` and `f(some_record_
    // value)` typecheck against `f` declared with a/b kwargs.
    if (args.length === 1 && Object.keys(kwargs).length === 0 && inputs.length > 0) {
      const splatType = inferExpr(args[0], scopes);
      if (splatType && splatType.kind === 'record' && splatType.fields) {
        const inputNames = new Set(inputs.map((i: any) => i.name));
        let allMatch = true;
        for (const k in splatType.fields) {
          if (!inputNames.has(k)) { allMatch = false; break; }
        }
        if (allMatch) {
          // Synthesize per-field exprs by typing through `splatType`
          // — no AST rewrite here, just record the per-field types
          // for the unify loop. We re-key the call by field name.
          const splatKwargs: Record<string, any> = {};
          for (const k in splatType.fields) splatKwargs[k] = { __splatType: splatType.fields[k] };
          args = [];
          kwargs = splatKwargs;
        }
      }
    }

    // Positional first, then keyword. Spec allows both calling
    // conventions for user-defined callables with explicit boundaries.
    for (let i = 0; i < inputs.length; i++) {
      const inp = inputs[i];
      let actual: any = null, actualLoc = expr.loc;
      if (i < args.length) {
        actual = inferExpr(args[i], scopes);
        actualLoc = args[i].loc;
      } else if (inp.name in kwargs) {
        const kw = kwargs[inp.name];
        if (kw && kw.__splatType) {
          // Came from auto-splat — type already resolved against the
          // record's field. No AST node to re-infer / locate.
          actual = kw.__splatType;
        } else {
          actual = inferExpr(kw, scopes);
          actualLoc = kw.loc;
        }
      } else {
        // Missing argument. Diagnostic but don't bail — the result
        // type doesn't depend on which inputs were supplied (we use
        // the function's declared result).
        diagnostics.push({
          severity: 'error',
          message: 'call to "' + head.name + '" missing argument "' + inp.name + '"',
          loc: expr.loc,
        });
        continue;
      }
      if (actual && actual.kind !== 'failed') {
        const s = T.unify(inp.type, actual, new Map());
        if (s == null) {
          diagnostics.push({
            severity: 'error',
            message: head.name + ': arg "' + inp.name + '" expects ' + T.show(inp.type)
              + ', got ' + T.show(actual),
            loc: actualLoc,
          });
        }
      }
    }

    // Polymorphic-at-call-site (spec §sec:functionof). The
    // monomorphic `calleeType.result` was computed with each param
    // typed as `any` at the callee's definition site, which under-
    // specifies in general. Re-infer the body with the call-site's
    // actual arg types in scope; that tightens results like
    // `f([1.0, 2.0]) → array(...)` where the monomorphic path gave
    // `any`. Falls back to the monomorphic result when re-inference
    // doesn't sharpen (e.g. recursive calls — body inference re-enters
    // and bails to deferred via the visiting set — or when the
    // binding's IR isn't a functionof shape we can walk).
    const callee = loweredModule.bindings.get(head.name);
    const calleeIR = callee && callee.rhs;
    if (calleeIR && calleeIR.kind === 'call' && calleeIR.op === 'functionof'
        && calleeIR.body && Array.isArray(calleeIR.params)
        && !visiting.has(head.name)) {
      const params = calleeIR.params;
      const paramKwargs = calleeIR.paramKwargs || [];
      const newScope = new Map<string, any>();
      for (let i = 0; i < params.length; i++) {
        let argT: any = T.any();
        if (i < args.length) {
          argT = inferExpr(args[i], scopes);
        } else {
          const kwName = paramKwargs[i] || params[i];
          if (kwName in kwargs) {
            const kw = kwargs[kwName];
            argT = (kw && kw.__splatType) ? kw.__splatType : inferExpr(kw, scopes);
          }
        }
        newScope.set(params[i], argT);
      }
      const polymorphic: any = inferExpr(calleeIR.body, [newScope]);
      // Only use the polymorphic result when it sharpens — i.e. is
      // concrete and not failed. If the body re-inference produces
      // failed / deferred / any, keep the monomorphic `calleeType.result`
      // so legacy behaviour (the diagnostic-free default-to-deferred
      // path) is preserved.
      if (polymorphic && polymorphic.kind !== 'failed'
          && polymorphic.kind !== 'deferred' && polymorphic.kind !== 'any') {
        return write(polymorphic, expr);
      }
    }
    return write(calleeType.result, expr);
  }

  // -------------------------------------------------------------------
  // Set-expression value-type resolution (used by elementof)
  // -------------------------------------------------------------------

  function setValueType(expr: any, scopes: any): any {
    if (!expr) return null;
    if (expr.kind === 'const' && SET_VALUE_TYPES[expr.name] !== undefined) {
      return SET_VALUE_TYPES[expr.name];
    }
    if (expr.kind === 'ref' && expr.ns === 'self' && SET_VALUE_TYPES[expr.name] !== undefined) {
      return SET_VALUE_TYPES[expr.name];
    }
    if (expr.kind !== 'call') return null;
    switch (expr.op) {
      case 'interval':   return T.REAL;
      case 'stdsimplex': {
        const n = expr.args && expr.args[0] && expr.args[0].kind === 'lit'
          && Number.isInteger(expr.args[0].value) ? expr.args[0].value : '%dynamic';
        return T.array(1, [n], T.REAL);
      }
      case 'cartpow': {
        const inner: any = setValueType(expr.args[0], scopes);
        if (inner == null) return null;
        // Per spec §03 sets: cartpow(S, size) where size is a
        // positive integer (1-D) or a vector of positive integers
        // (multi-axis shape). Vector form lowers to `vector(...)`
        // — unpack it the same way `iid` does, so the result type
        // carries one dim per axis rather than collapsing into a
        // single rank-1 dynamic shape.
        let dimArgs: any[] = (expr.args || []).slice(1);
        if (dimArgs.length === 1
            && dimArgs[0].kind === 'call'
            && dimArgs[0].op === 'vector'
            && Array.isArray(dimArgs[0].args)) {
          dimArgs = dimArgs[0].args;
        }
        // Engine-concepts §17.4 — const-eval-driven shape resolution.
        // Same pattern as `iid`: try literal first, then the resolver
        // callback (which short-circuits length/lengthof/sizeof through
        // the inferred type). Falls back to %dynamic when const-eval
        // can't prove a concrete integer.
        const dims = dimArgs.map((a: any) => {
          const v = resolveIntegerShape(a);
          return v != null ? v : '%dynamic';
        });
        return T.array(dims.length, dims, inner);
      }
      case 'cartprod': {
        const fields = expr.fields || null;
        if (fields && fields.length > 0) {
          const out: Record<string, any> = {};
          for (const f of fields) {
            const t = setValueType(f.value, scopes);
            if (t == null) return null;
            out[f.name] = t;
          }
          return T.record(out);
        }
        const elems = (expr.args || []).map((a: any) => setValueType(a, scopes));
        if (elems.some((e: any) => e == null)) return null;
        return elems.length === 1 ? elems[0] : T.tuple(elems);
      }
    }
    return null;
  }

  // -------------------------------------------------------------------
  // Diagnostics helpers (suppress cascades when inputs already failed)
  // -------------------------------------------------------------------

  function arityError(op: any, expected: any, got: any, loc: any) {
    diagnostics.push({
      severity: 'error',
      message: op + ' expects ' + expected + ' positional argument(s), got ' + got,
      loc,
    });
    return T.failed(op + ' arity');
  }
  // _rowstackHint(expected, got) — when a matrix-input signature site
  // is fed a vector-of-vectors, append a §03-citing hint that points
  // the user at the explicit `rowstack(...)` / `colstack(...)` lift.
  // Returns the trailing hint string (possibly empty).
  function _rowstackHint(expected: any, got: any): string {
    // Expected: array(2, [...], scalar). Got: array(1, [...], array(...)).
    // Detect the "matrix-wanted, vec-of-vec-given" pattern at the
    // outermost level — the engine's matrix-input signatures all use
    // rank=2 with scalar elem, and vec-of-vec lowers to rank=1 with
    // array elem.
    if (!expected || expected.kind !== 'array' || expected.rank !== 2) return '';
    if (!expected.elem || expected.elem.kind !== 'scalar') return '';
    if (!got || got.kind !== 'array' || got.rank !== 1) return '';
    if (!got.elem || got.elem.kind !== 'array') return '';
    return ' — this is a vector-of-vectors per spec §03, not a matrix; '
      + 'wrap with `rowstack(...)` (rows = inner vectors) or `colstack(...)` '
      + '(columns = inner vectors) to commit the storage-order interpretation.';
  }

  function argError(op: any, i: any, expected: any, got: any, loc: any) {
    if (got && got.kind === 'failed') return T.failed(op + ' arg type (cascade)');
    diagnostics.push({
      severity: 'error',
      message: op + ': arg ' + (i + 1) + ' expects ' + T.show(expected)
        + ', got ' + T.show(got) + _rowstackHint(expected, got),
      loc,
    });
    return T.failed(op + ' arg type');
  }
  function kwargError(op: any, k: any, expected: any, got: any, loc: any) {
    if (got && got.kind === 'failed') return T.failed(op + ' kwarg type (cascade)');
    diagnostics.push({
      severity: 'error',
      message: op + ': kwarg "' + k + '" expects ' + T.show(expected)
        + ', got ' + T.show(got) + _rowstackHint(expected, got),
      loc,
    });
    return T.failed(op + ' kwarg type');
  }

  return { diagnostics, inferBinding, inferExpr };
}

// Internal "set" marker — not a user-facing type. elementof handles it.
function setMarker(name: any) { return { kind: 'set', name }; }

module.exports = { inferTypes, inferExprInScope };
