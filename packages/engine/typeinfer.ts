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
const SC = require('./shape-contract.ts');
const builtins = require('./builtins.ts');
const aggregateShape = require('./aggregate-shape.ts');
const vsLib = require('./value-set.ts');

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
// The four FlatPDL transports (spec §07) — undefined on discrete kernels.
const TRANSPORT_OPS = new Set([
  'builtin_touniform', 'builtin_fromuniform',
  'builtin_tonormal', 'builtin_fromnormal',
]);

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
 * stays unaware of the value-mode evaluator. Engine-concepts §17.1
 * "resolve, don't rewrite" — the resolver is consulted only at
 * narrowly-identified shape positions, and only the resulting
 * integer is embedded in type annotations; the source IR is left
 * intact in either case.
 */
function inferTypes(loweredModule: any, opts?: { resolveFixed?: any; modules?: any }) {
  const ctx = createInferenceContext(loweredModule, opts);
  for (const [name] of loweredModule.bindings) ctx.inferBinding(name);
  // Refinement domains over the type-inferred module (engine-concepts
  // §17.3), in dependency order: valueset (the third `%meta` slot) THEN
  // normalization (the `%mass` class — its Lebesgue/Counting rule reads
  // set boundedness). The mass pass also raises the
  // normalize-of-infinite/null static error.
  ctx.fillValuesets();
  ctx.fillMasses();
  // Consumer of the valueset domain: flag distribution parameters whose
  // value set is PROVABLY outside the parameter's required domain (spec
  // §08), e.g. `Normal(sigma = -1.0)`. Reads the valuesets filled above.
  ctx.checkDomainContracts();
  // Spec §06 "Known-bijection registry": a domain-restricted pushfwd
  // forward (log/log10/sqrt/log1p/logit/probit) additionally requires the
  // base measure's support to lie within that domain — refuse (error
  // diagnostic) rather than silently score a sub-probability measure
  // (#260 (c)).
  ctx.checkPushfwdDomainContracts();
  // Spec §04 "Reification and module scope": functionof/kernelof may not
  // take a cross-module parameterized value as an input. Needs the dep
  // registry (cross-module phases) — a no-op without a bundle.
  ctx.checkCrossModuleReification();
  return ctx.diagnostics;
  // NOTE: no eager post-binding const-eval pass. The resolver is
  // demand-driven (engine-concepts §17.1) — it's invoked only from
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
function createInferenceContext(loweredModule: any, opts?: { resolveFixed?: any; modules?: any }) {
  const diagnostics: any[] = [];
  const visiting = new Set();
  const visited  = new Set();
  const resolveFixed = opts && opts.resolveFixed;
  // Registry of compiled sibling modules (resolved-path → compiled
  // module), for cross-module `load_module` ref resolution (spec §04/§11).
  // Absent for a standalone single-file compile or an on-demand
  // inferExprInScope call — then `mod.x` stays `deferred`.
  const modules = opts && opts.modules;

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
    return resolveUserModuleRef(expr, modAlias);
  }

  // Resolve `(%ref <alias> <name>)` through a user `load_module(...)`
  // dependency (spec §04 Module composition + §11 cross-module
  // inference). The dependency was compiled (and type-inferred) before
  // this module, so its binding's `inferredType` is already available in
  // the `modules` registry. Enforces the spec §04 access rules — only
  // PUBLIC, fixed/parameterized bindings cross the module boundary
  // (stochastic boundary) — emitting a diagnostic on the offending ref.
  function resolveUserModuleRef(expr: any, modAlias: string): any {
    if (!modules) return T.deferred();          // no compile ctx (e.g. inferExprInScope)
    const reg = loweredModule.moduleRegistry && loweredModule.moduleRegistry[modAlias];
    const path = reg && reg.path;
    if (!path) return T.deferred();             // unresolved path — already diagnosed by the bundle compiler
    const dep = modules.get(path);
    if (!dep) return T.deferred();              // missing source — already diagnosed
    const bindingName = expr.name;
    const depBinding = dep.loweredModule.bindings.get(bindingName);
    if (!depBinding) {
      diagnostics.push({ severity: 'error',
        message: "'" + bindingName + "' is not a binding of module '" + modAlias
          + "' (loaded from '" + path + "')",
        loc: expr.loc });
      return T.failed("'" + bindingName + "' not in module '" + modAlias + "'");
    }
    // Spec §04 "Binding names": only public bindings form the module's
    // interface; underscore-private bindings are not accessible.
    if (!dep.loweredModule.publicSet.has(bindingName)) {
      diagnostics.push({ severity: 'error',
        message: "'" + bindingName + "' is private to module '" + modAlias
          + "' and not accessible across the module boundary",
        loc: expr.loc });
      return T.failed("'" + bindingName + "' is private to '" + modAlias + "'");
    }
    // Spec §04 "Stochastic boundary": stochastic bindings (direct draws
    // or unreified draw-descendants) are invisible to the loading module.
    if (depBinding.phase === 'stochastic') {
      diagnostics.push({ severity: 'error',
        message: "'" + bindingName + "' is a stochastic binding of module '"
          + modAlias + "' — only fixed or parameterized bindings cross the "
          + 'module boundary (spec §04). Reify it via lawof / kernelof first.',
        loc: expr.loc });
      return T.failed("'" + bindingName + "' is stochastic (module boundary)");
    }
    return depBinding.inferredType || T.deferred();
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
      // Module loads (spec §11 %module): the binding is a namespace
      // handle, not a value. Cross-module member access `mod.x` is typed
      // separately in `inferCrossModuleRef` (it reads the loaded
      // module's binding type), not here.
      case 'load_module':     return write(T.moduleType(), expr);
      case 'standard_module': return write(T.moduleType(), expr);
      case 'elementof': return write(inferElementof(expr, scopes), expr);
      case 'lawof':     return write(inferLawof(expr, scopes), expr);
      case 'record':    return write(inferRecord(expr, scopes), expr);
      case 'table':     return write(inferTable(expr, scopes), expr);
      case 'cat':       return write(inferCat(expr, scopes), expr);
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
      // Normalization functions (spec §07): vector → vector, LENGTH-
      // PRESERVING. The static signature returns `array(1, %dynamic,
      // real)`; refine the result to the input's concrete length so a
      // downstream `aggregate`/`broadcast` sees a known dim (and the
      // %meta type slot matches the value-set's concrete dim).
      case 'softmax':
      case 'logsoftmax':
      case 'l1unit':
      case 'l2unit': {
        const a = expr.args || [];
        if (a.length === 1) {
          const at = inferExpr(a[0], scopes);
          if (at && at.kind === 'array' && at.rank === 1
              && Array.isArray(at.shape) && at.shape.length === 1
              && at.shape[0] !== '%dynamic'
              && at.elem && at.elem.kind === 'scalar' && at.elem.prim === 'real') {
            return write(T.array(1, [at.shape[0]], T.REAL), expr);
          }
        }
        break;   // unknown / non-concrete length → generic signature
      }
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
      // Const-eval-driven shape inference (engine-concepts §17.1).
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
      // §19). Result type is the composed function type computed by
      // the shared `inferChainComposition` helper (consume/rest at the
      // chain's input-set level — engine-concepts §17.3 extended).
      case 'fchain':    return write(inferFchain(expr, scopes), expr);
      // jointchain / kchain — dependent composition (spec §06 line
      // 192-266). Closed-first ⇒ measureType; kernel-first ⇒
      // kernelType with residual inputs (collapses to measure when
      // residual is empty). Routes through inferChainComposition's
      // kernel modes (engine-concepts §19).
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
      // relabel(X, names) — output-side axis renaming (spec §04). Kind-
      // transparent: a relabel'd measure stays a measure (so iid /
      // truncate / normalize / likelihoodof accept it), a relabel'd
      // function/kernel stays callable, a relabel'd value becomes a
      // record. Without this rule relabel infers `deferred` and iid's
      // measure guard rejects `iid(relabel(M), n)`.
      case 'relabel':     return write(inferRelabel(expr, scopes), expr);
      // A non-scalar locscale survives the analyzer pre-pass as an
      // `{op:'locscale'}` node (it routes through lift's affine-registry
      // lowering, not the scalar expansion). Its type follows the base
      // measure — an affine pushforward changes neither domain nor shape.
      case 'locscale':    return write(inferLocscale(expr, scopes), expr);
      case 'checked':     return write(inferChecked(expr, scopes), expr);
    }
    // Static refusal: the four FlatPDL transports (touniform / fromuniform
    // / tonormal / fromnormal) are undefined on a DISCRETE kernel — there
    // is no continuous CDF/quantile to map through (spec §07). Lift the
    // runtime `density-prims._rejectDiscreteTransport` refusal to inference
    // time so the diagnostic points at the source. Type is unchanged
    // (the generic `any`-result signature still applies below).
    if (TRANSPORT_OPS.has(expr.op)) checkTransportKernelContinuous(expr);
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
            // "arg 2 expects any, got …" (engine-concepts §17.1).
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
    const result = T.substitute(sig.result, s);
    // Stamp the exact vector length onto the measure-domain of the three
    // multivariate VECTOR distributions whose atom length is statically known
    // (mirrors flatppl-rust ops.rs `param_dim`; the valueset path already
    // derives the same length via `_paramDim`, so this keeps the type-domain
    // in step — `Dirichlet(alpha=[1,1,1])` ⇒ `array[3]`, not `array[%dynamic]`).
    // The signature can only yield `%dynamic` (array shapes aren't captured
    // through unify). Matrix dists + PoissonProcess stay dynamic by design
    // (Rust's `dynmat` is also dynamic; a point process is genuinely ragged).
    const lenSpec = (MULTIVARIATE_VECTOR_DIST_LEN as any)[op];
    if (lenSpec && result && result.kind === 'measure'
        && result.domain && result.domain.kind === 'array' && result.domain.rank === 1) {
      const n = _paramDim(expr, lenSpec[0], lenSpec[1]);
      if (n !== '%dynamic') result.domain.shape = [n];   // freshly substituted ⇒ safe to set
    }
    return result;
  }
  // Length-defining parameter (kwarg name, positional index) for the
  // statically-sized multivariate vector distributions.
  const MULTIVARIATE_VECTOR_DIST_LEN: Record<string, [string, number]> = {
    MvNormal:    ['mu', 0],
    Dirichlet:   ['alpha', 0],
    Multinomial: ['p', 1],
  };

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
  // checked(value, condition) — spec §07 value-preserving assertion.
  // Result type/phase is IDENTICAL to `value` (it threads `value`
  // through unchanged), so we return value's inferred type directly. The
  // `condition` must be a boolean (the type half of the contract; the
  // fixed-phase half is enforced in the analyzer, the phase authority).
  // Accepts the canonical kwarg form `checked(value = ..., condition =
  // ...)`, the positional value `checked(value_expr, condition = ...)`,
  // and the fully-positional `checked(value_expr, condition_expr)`.
  // The kernel name of a FlatPDL transport / eval-prim — args[0] is a
  // bare distribution name in one of three shapes (mirrors sampler's
  // `_resolveKernelName`): a string literal, a kernel call (read its op),
  // or a bare identifier ref. Returns null when it isn't statically a
  // plain name (e.g. a user binding the check can't resolve — skip it).
  function _transportKernelName(expr: any): string | null {
    const k = (expr.args || [])[0];
    if (!k) return null;
    if (k.kind === 'lit' && typeof k.value === 'string') return k.value;
    if (k.kind === 'call' && typeof k.op === 'string')   return k.op;
    if (k.kind === 'ref'  && typeof k.name === 'string')  return k.name;
    return null;
  }

  function checkTransportKernelContinuous(expr: any): void {
    const name = _transportKernelName(expr);
    if (name == null) return;
    const irShared = require('./ir-shared.ts');
    if (irShared.DISCRETE_DISTRIBUTIONS && irShared.DISCRETE_DISTRIBUTIONS.has(name)) {
      diagnostics.push({
        severity: 'error',
        message: `${expr.op}: '${name}' is a discrete kernel; the four FlatPDL `
          + `transports (touniform / fromuniform / tonormal / fromnormal) are `
          + `defined only on continuous kernels (spec §07 — no continuous `
          + `CDF/quantile to map through).`,
        loc: ((expr.args || [])[0] && (expr.args || [])[0].loc) || expr.loc,
      });
    }
  }

  function inferChecked(expr: any, scopes: any): any {
    const args = expr.args || [];
    const kwargs = expr.kwargs || {};
    const valueExpr = ('value' in kwargs) ? kwargs.value : args[0];
    const condExpr  = ('condition' in kwargs) ? kwargs.condition : args[1];
    if (valueExpr == null) {
      diagnostics.push({
        severity: 'error',
        message: 'checked(): requires a value argument',
        loc: expr.loc,
      });
      return T.failed('checked: no value');
    }
    const vt: any = inferExpr(valueExpr, scopes);
    if (condExpr != null) {
      const ct: any = inferExpr(condExpr, scopes);
      // Cascades (already-failed condition) don't pile on a second error.
      const isCascade = ct && ct.kind === 'failed';
      if (!isCascade && T.unify(T.BOOLEAN, ct, new Map()) == null) {
        diagnostics.push({
          severity: 'error',
          message: 'checked(): condition must be a boolean, got ' + T.show(ct),
          loc: (condExpr.loc || expr.loc),
        });
      }
    }
    // Value passes through with identical type (and phase, set upstream).
    return vt;
  }

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
  // Object-layer kinds and tuples may not be stored in a value container
  // (array / record / table). Per spec §04 these are first-class OBJECTS,
  // not values: measures/kernels/functions/likelihoods are barred outright,
  // and a tuple may appear only inside another tuple (so tuple construction
  // never calls this). Records nest in records and tables carry table
  // columns — checked elsewhere. Arrays remain numeric per §03, but
  // array-of-record / table (a "vector of records" that arguably belongs in
  // a table) is a separate, pre-existing question the generative executor
  // currently relies on, and is deliberately NOT enforced here.
  // Returns the offending noun, or null when `t` is allowed in `container`.
  function disallowedContainerNoun(t: any): string | null {
    const obj = objectLayerNoun(t);
    if (obj) return obj;
    if (t && t.kind === 'tuple') return 'tuple';
    // A table is barred from a value container: arrays are numeric (§03),
    // record fields are scalar/array/record (§03). A table column lives in a
    // table — handled by inferTable's own table-column branch, which never
    // calls this guard, so legitimate nesting is unaffected.
    if (t && t.kind === 'table') return 'table';
    return null;
  }
  function checkContainerElem(t: any, loc: any, container: string, label: string) {
    const noun = disallowedContainerNoun(t);
    if (!noun) return;
    const art = container === 'array' ? 'an ' : 'a ';
    const why = (noun === 'tuple')
      ? 'spec §04 — tuples are objects and nest only inside other tuples, '
        + 'not inside an array, record, or table'
      : (noun === 'table')
      ? 'spec §03 — arrays are numeric and record fields are scalars, arrays, '
        + 'or records; a table belongs in a table column'
      : 'spec §04 — measures, kernels, likelihoods and functions are '
        + 'first-class objects but cannot be stored in arrays, records, or tables';
    diagnostics.push({
      severity: 'error',
      message: container + ' ' + label + ': a ' + noun + ' may not appear inside ' + art
        + container + ' (' + why + ')'
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
    return _reduceTableType(t, expr.op);
  }
  // The result type of a column-wise reduction over a table: a record of
  // per-column results. A table-valued column reduces to a NESTED record,
  // recursively (mirrors the runtime `_tableReduce`, which dispatches the
  // reduction op into the sub-table).
  function _reduceTableType(t: any, op: string): any {
    const fields: Record<string, any> = {};
    for (const k in t.columns) {
      const cT = t.columns[k];
      if (cT && cT.kind === 'table') {
        fields[k] = _reduceTableType(cT, op);
      } else if (op === 'var' || op === 'std') {
        // var/std are real-valued but reduce over the ROW axis only, so a
        // vector-per-entry column keeps its cell shape with a real leaf.
        fields[k] = _realLeafType(cT);
      } else {
        // sum, prod, mean, maximum, minimum preserve element type.
        fields[k] = cT;
      }
    }
    return T.record(fields);
  }
  // Replace the scalar leaf of a (possibly nested-array) column type with
  // `real`, preserving any array shape — the result type of var/std.
  function _realLeafType(cT: any): any {
    if (cT && cT.kind === 'array') return T.array(cT.rank, cT.shape, _realLeafType(cT.elem));
    return T.REAL;
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
    // Record one column's row count, enforcing equal length across columns
    // (spec §03). A non-numeric length ('%dynamic') is unconstrained.
    // Returns false (after pushing a diagnostic) on a mismatch.
    const noteRowCount = (len: any, fname: string, loc: any): boolean => {
      if (typeof len !== 'number') return true;
      if (!nrowsBound) { nrows = len; nrowsBound = true; return true; }
      if (nrows !== len) {
        diagnostics.push({
          severity: 'error',
          message: 'table: column "' + fname + '" has length ' + len
            + ', but earlier columns have length ' + nrows
            + ' (spec §03: all columns must have equal length)',
          loc,
        });
        return false;
      }
      return true;
    };
    for (const f of fields) {
      const ct: any = inferExpr(f.value, scopes);
      const loc = f.loc || expr.loc;
      // A column is a vector (rank-1 array) or a table (spec §03). Deferred /
      // any flow through with a deferred column type — the engine still
      // produces a table at runtime.
      if (ct && ct.kind === 'array' && ct.rank === 1) {
        // Spec §04: a table column may not hold measures / kernels / etc.
        checkContainerElem(ct.elem, loc, 'table', "column '" + f.name + "'");
        columns[f.name] = ct.elem;
        if (!noteRowCount(ct.shape[0], f.name, loc)) return T.failed('table column length mismatch');
      } else if (ct && ct.kind === 'table') {
        // A table-valued column: store the sub-table type; its row count
        // must match the other columns. The sub-table's own columns were
        // already validated when it was constructed.
        columns[f.name] = ct;
        if (!noteRowCount(ct.nrows, f.name, loc)) return T.failed('table column length mismatch');
      } else if (ct && (ct.kind === 'deferred' || ct.kind === 'any')) {
        columns[f.name] = T.deferred();
      } else if (ct && ct.kind === 'failed') {
        return T.failed('table column cascade');
      } else {
        diagnostics.push({
          severity: 'error',
          message: 'table: column "' + f.name + '" must be a vector or a table; got '
            + T.show(ct),
          loc,
        });
        return T.failed('table non-vector column');
      }
    }
    return T.table(columns, nrows);
  }

  // cat(x, y, …) — structural concatenation (spec §07), via the shared
  // catShapeType rule: all scalars → a vector, all vectors → a concatenated
  // vector, all records → a merged record; mixing kinds is a static error.
  function inferCat(expr: any, scopes: any) {
    const parts = (expr.args || []).map((a: any) => inferExpr(a, scopes));
    if (parts.some((t: any) => t && t.kind === 'failed')) return T.failed('cat cascade');
    if (parts.length === 0) return T.failed('cat: requires at least one argument');
    const shaped = catShapeType(parts);
    if (shaped == null) {
      diagnostics.push({
        severity: 'error',
        message: 'cat: arguments must be all scalars, all vectors, or all records '
          + 'with distinct fields — concatenating a mix of value kinds is not '
          + 'permitted (spec §07)',
        loc: expr.loc,
      });
      return T.failed('cat mixed kinds');
    }
    return shaped;
  }

  function inferJoint(expr: any, scopes: any) {
    const fields = expr.fields || [];
    const args = expr.args || [];
    // Positional joint (spec §06): the variate is the `cat` of the component
    // variates — all scalars → a vector, all vectors → a concatenated vector,
    // all records → a merged record; mixing shape classes (or a duplicate
    // variate name across record components) is a static error. The shape is
    // owned by shape-contract.catShape (shared with `cat` §07 and positional
    // `cartprod` §03 so the three can't drift). The density side already
    // consumes the flat/record variate correctly via consume/rest — before this
    // the type alone was wrong (an empty `record({})` for every non-all-scalar
    // case). Mixed positional+keyword args fall through to the keyword (record)
    // path below.
    if (args.length > 0 && fields.length === 0) {
      const domains: any[] = [];
      for (const a of args) {
        const at = inferExpr(a, scopes);
        if (T.isMeasure(at)) {
          domains.push(at.domain);
        } else if (at.kind === 'deferred' || at.kind === 'any') {
          domains.push(T.deferred());
        } else if (at.kind === 'failed') {
          return T.failed('joint cascade');
        } else {
          diagnostics.push({
            severity: 'error',
            message: 'joint expects measures as components, got ' + T.show(at),
            loc: a.loc || expr.loc,
          });
          return T.failed('joint bad component');
        }
      }
      const spec = SC.catShape(domains);
      if (spec == null) {
        diagnostics.push({
          severity: 'error',
          message: 'joint: positional components must be all scalar, all vector, '
            + 'or all record measures with distinct variate names — mixing shape '
            + 'classes is not permitted (spec §06); use the keyword form '
            + 'joint(a = …, b = …) to name components',
          loc: expr.loc,
        });
        return T.failed('joint mixed shape classes');
      }
      return T.measure(SC.typeOfShape(spec));
    }
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
   * shared `inferChainComposition` helper (engine-concepts §19).
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
   * 192-266; engine-concepts §19). Lowering surfaces:
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
      // A table-valued column accesses as the sub-table itself; a vector
      // column accesses as a vector of its element type (spec §03).
      const colT = recT.columns[nameIR.value];
      return (colT && colT.kind === 'table') ? colT : T.array(1, [recT.nrows], colT);
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

  // `oneBased` distinguishes the two callers that share this inferrer:
  //   - the engine-internal `tuple_get` op (multi-LHS decomposition) carries
  //     a 0-BASED slot — call with oneBased=false (the default);
  //   - surface `t[i]` lowers to `get(t, i)` with a 1-BASED index (spec §04)
  //     and is redirected here from `inferGet` with oneBased=true.
  // The runtime mirrors this: `tuple_get` indexes the JS array directly
  // (0-based) while `get` on a tuple subtracts one (1-based). Conflating the
  // two silently shifted heterogeneous/nested tuple element types by one.
  function inferTupleGet(expr: any, scopes: any, oneBased: boolean = false): any {
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
    const raw = slotIR.value | 0;          // as written (1-based for `get`)
    const i = oneBased ? raw - 1 : raw;     // resolved 0-based array index
    if (i < 0 || i >= tupleT.elems.length) {
      diagnostics.push({
        severity: 'error',
        message: `tuple index ${raw} out of range for ${T.show(tupleT)}`,
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

    // Tuple integer-index: redirect to tuple_get's existing inferrer.
    // `get` (surface `t[i]`) is 1-based (spec §04/§07); `get0` is the
    // zero-based variant (spec §07 "get0 = zero-based variant of get"), so
    // resolve the slot base off the op — mirrors the runtime's
    // `oneBased = (op === 'get')` in sampler.evaluateCall. (Hardcoding 1-based
    // here rejected `get0(t, 0)` — the determiniser's `(value, rngstate)`
    // sample-tuple projection — as out-of-range.)
    if (containerT && containerT.kind === 'tuple'
        && args.length === 2
        && args[1].kind === 'lit'
        && typeof args[1].value === 'number') {
      return inferTupleGet(expr, scopes, expr.op === 'get');
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
    //   - get(t, all) (whole row axis) → array of the row record, length nrows.
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
        // Row access — a record over the column names (table columns
        // become nested records; spec §03).
        return T.rowRecordType(containerT);
      }
      if (args.length === 2 && args[1].kind === 'const' && args[1].name === 'all') {
        // Whole row axis (`t[:]`) → array of the row record, length = nrows
        // (the runtime returns exactly this; sampler get(t, all)).
        return T.array(1, [containerT.nrows], T.rowRecordType(containerT));
      }
      if (args.length === 2) {
        // Row access via expression (axis or computed int). Conservative:
        // a row is a record over the table's columns.
        const selT: any = inferExpr(args[1], scopes);
        if (selT && selT.kind === 'scalar' && selT.prim === 'integer') {
          return T.rowRecordType(containerT);
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
          cellType: T.rowRecordType(t),
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
    // `mod` and `div` are integer-domain (spec §07: `mod(a, b) = a − b·⌊a/b⌋`,
    // `div(a, b) = ⌊a/b⌋`, both over `integers` with `b ≠ 0`). The general
    // arith ladder (unifyArith) admits reals, so enforce the integer
    // restriction here: a real (or complex) operand is a static error rather
    // than a silent fractional result. Real division is the separate `divide`
    // op, which keeps its `(real, real) → real` signature and is NOT checked.
    // Booleans embed into integers (spec §03 `booleans ⊂ integers ⊂ reals`),
    // so they pass; deferred/any/type-var operands are left to runtime (can't
    // disprove integer statically). The `b ≠ 0` precondition is NOT enforced
    // statically — a zero divisor yields a non-finite IEEE result at runtime.
    if (expr.op === 'mod' || expr.op === 'div') {
      const elemPrim = (t: any): string | null => {
        if (!t) return null;
        if (t.kind === 'scalar') return t.prim;
        if (t.kind === 'array') return elemPrim(t.elem);
        return null;
      };
      const operands = [aT, bT];
      for (let i = 0; i < operands.length; i++) {
        const p = elemPrim(operands[i]);
        if (p === 'real' || p === 'complex') {
          diagnostics.push({
            severity: 'error',
            message: expr.op + ': operands must be integer (spec §07 integer '
              + 'domain), but argument ' + (i + 1) + ' has type '
              + T.show(operands[i]) + (expr.op === 'div'
                ? ' — use `divide` for real division' : ''),
            loc: (args[i] && args[i].loc) || expr.loc,
          });
          return T.failed(expr.op + ' non-integer operand');
        }
      }
    }
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

  // The codomain of `pushfwd(f, M)` — the type f produces when applied to a
  // value of M's variate type. `f.result` is the *definition-time* type (the
  // reified param bound to `any`), so it does NOT reflect the base's shape:
  // `fn(2.0 .* _).result` is a scalar even for a vector base. Re-infer f's
  // single-param body with the param bound to `inputType` — the call-site
  // specialization the FlatPIR polymorphic path prescribes (mirrors
  // `inferReification`'s body walk, with a concrete input type instead of
  // `any`). Returns the value-typed codomain, or `null` → the caller keeps its
  // scalar default for the cases this can't specialize (a bare builtin symbol,
  // a multi-param callable, an unresolved base). The density side already
  // evaluates f's body at the drawn value, so this makes the TYPE agree.
  function _pushfwdCodomain(fExpr: any, inputType: any, scopes: any): any {
    if (!inputType || inputType.kind === 'failed' || inputType.kind === 'deferred'
        || inputType.kind === 'any') return null;
    // Resolve fExpr to a single-param reification node: an inline `fn(...)` /
    // `functionof(...)`, or a self-ref binding to one.
    let reif: any = null;
    if (fExpr && fExpr.op === 'functionof') reif = fExpr;
    else if (fExpr && fExpr.kind === 'ref' && fExpr.ns === 'self') {
      const b = loweredModule.bindings.get(fExpr.name);
      if (b && b.rhs && b.rhs.op === 'functionof') reif = b.rhs;
    }
    if (!reif || !Array.isArray(reif.params) || reif.params.length !== 1 || !reif.body) {
      return null;
    }
    const scope = new Map();
    scope.set(reif.params[0], inputType);
    // Speculative: infer the body purely to READ its codomain. Any diagnostics
    // it raises (e.g. `exp` applied to a record base — an ill-typed pushfwd we
    // don't type-check here) must NOT leak into the module's stream, or they'd
    // change unrelated results (a spurious error surfaced through this query
    // broke the disintegrate spec-coverage contract). Drop anything the walk
    // appended; when the body doesn't resolve to a value the caller falls back
    // to its scalar default, unchanged from before S2.
    const savedLen = diagnostics.length;
    const bodyT: any = inferExpr(reif.body, scopes.concat([scope]));
    if (diagnostics.length > savedLen) diagnostics.length = savedLen;
    return T.isValue(bodyT) ? bodyT : null;
  }

  function inferPushfwd(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 2) return arityError('pushfwd', '2', args.length, expr.loc);
    // arg 0 is the function; we don't statically check its type
    // here (callable-type tracking is the orchestrator's job).
    inferExpr(args[0], scopes);
    const m2 = inferExpr(args[1], scopes);
    if (T.isMeasure(m2)) {
      const cod = _pushfwdCodomain(args[0], m2.domain, scopes);
      return T.measure(cod || T.REAL);
    }
    if (m2 && m2.kind === 'kernel') {
      // pushfwd acts on the kernel's OUTPUT measures (spec §06); preserve the
      // input signature, specialize the output variate to f's codomain.
      const outMeasure = m2.output;
      const baseDomain = (outMeasure && T.isMeasure(outMeasure)) ? outMeasure.domain : null;
      const cod = baseDomain ? _pushfwdCodomain(args[0], baseDomain, scopes) : null;
      return { kind: 'kernel', inputs: m2.inputs || {}, output: T.measure(cod || T.REAL) };
    }
    if (m2 && m2.kind === 'failed') return T.failed('pushfwd cascade');
    // Permissive default — pushfwd OUTSIDE measure/kernel context
    // (e.g. inside a `fn(...)` body whose arg-types haven't been
    // resolved yet) defers rather than erroring.
    return T.deferred();
  }

  // relabel(X, names) — output-side axis renaming (spec §04 "Interface
  // adaptation"). It is KIND-TRANSPARENT: the result has the same type
  // KIND as X (only axis labels change), so the spec equivalence
  // `named_M = relabel(M, names) ≡ pushfwd(fn(relabel(_, names)), M)`
  // holds and relabel'd measures flow into iid / truncate / normalize /
  // likelihoodof exactly like a bare measure. Cases:
  //   - measure  → measure (preserve domain + sampleShape/batch/event;
  //                the rename touches labels only, not shape).
  //   - function → function, kernel → kernel (callable lifts).
  //   - value    → record. Per spec lines 482-507 relabel of a value is
  //                a record construction (array→record by position,
  //                scalar→single-field record). lift.inlineRelabel rewrites
  //                the statically-resolvable value cases to `record(...)`
  //                before typeinfer, so we usually see relabel only over a
  //                measure/kernel; for any value-typed arg that survives
  //                (dynamic names, etc.) we still report `record` so the
  //                kind is correct and never regresses to a measure.
  //   - deferred/failed → propagate (don't fabricate a measure).
  function inferRelabel(expr: any, scopes: any): any {
    const args = expr.args || [];
    if (args.length !== 2) return arityError('relabel', '2', args.length, expr.loc);
    const baseT: any = inferExpr(args[0], scopes);
    inferExpr(args[1], scopes);  // names — infer so any refs resolve
    if (T.isMeasure(baseT)) return baseT;      // labels only — shape unchanged
    if (baseT && baseT.kind === 'function')    return baseT;
    if (baseT && baseT.kind === 'kernel')      return baseT;
    if (baseT && baseT.kind === 'failed')      return T.failed('relabel cascade');
    if (baseT && (baseT.kind === 'deferred' || baseT.kind === 'any')) return T.deferred();
    // value → record (spec §04 lines 482-507). When the names arg is a
    // literal [n1, n2, ...] (lowered as `vector(<string lit>, …)`) build a
    // field-typed record so downstream `get_field` sees the actual fields:
    //   - array base   → field per name, each the array element type;
    //   - record base  → rename existing field types by position;
    //   - scalar base  → single field carrying the scalar type.
    // Without literal names (dynamic), DEFER (permissive — matches the
    // pre-rule behaviour) rather than fabricating an empty / wrong record.
    if (T.isValue(baseT)) {
      const names = literalStringVector(args[1]);
      if (names == null || names.length === 0) return T.deferred();
      const fields: Record<string, any> = {};
      if (baseT.kind === 'array') {
        const elemT = baseT.elem || T.deferred();
        for (const n of names) fields[n] = elemT;
        return T.record(fields);
      }
      if (baseT.kind === 'record') {
        const oldVals = Object.values(baseT.fields || {});
        if (oldVals.length !== names.length) return T.deferred();
        names.forEach((n: string, i: number) => { fields[n] = oldVals[i]; });
        return T.record(fields);
      }
      // scalar (or other single value) → single-field record.
      if (names.length === 1) { fields[names[0]] = baseT; return T.record(fields); }
      return T.deferred();
    }
    return T.deferred();
  }

  // Read a names argument that is a literal string vector — the lowered
  // form `vector(<string lit>, …)` (or a bare array IR). The `call`/`vector`
  // shape is the lowerer's output for a string ArrayLiteral (e.g. `["x","y"]`).
  // Returns the list of names, or null when any element isn't a static string literal.
  function literalStringVector(arg: any): string[] | null {
    if (!arg || arg.kind !== 'call' || arg.op !== 'vector' || !Array.isArray(arg.args)) {
      return null;
    }
    const out: string[] = [];
    for (const el of arg.args) {
      if (!el || el.kind !== 'lit' || typeof el.value !== 'string') return null;
      out.push(el.value);
    }
    return out;
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
  // Demand-driven fixed-value boundary (engine-concepts §17.1): a shape
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
    // `lengthof(x)` / `length(x)` is a scalar count → a rank-1 shape
    // `[n]`. Delegate to resolveIntegerShape so the array-of-arrays and
    // table row-count short-circuits apply here too (resolveFixed's
    // shape-observer reads only array types, so a table count would
    // otherwise fall through to %dynamic — e.g. `zeros(lengthof(t))`).
    if (ir && ir.kind === 'call' && (ir.op === 'length' || ir.op === 'lengthof')) {
      const n = resolveIntegerShape(ir);
      if (n != null) return [n];
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
  // const-eval is enabled by the caller. Per engine-concepts §17.1:
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
    // The principle the engine-concepts §17.1 design was built for
    // — most common shape-determining expression in real models, and
    // the only safe way to chain through expensive intermediates.
    // (`sizeof(x)` returns the shape *vector* and is handled by
    // resolveIntegerVectorShape, not here.)
    if (ir && ir.kind === 'call'
        && (ir.op === 'length' || ir.op === 'lengthof')
        && Array.isArray(ir.args) && ir.args.length === 1) {
      const argT: any = inferExpr(ir.args[0], []);
      // Vector / array-of-arrays: the leading-axis length (rows of a
      // matrix, outer count of a vec-of-vec — spec §07 "number of
      // elements"). A rank-≥2 array's shape[0] is its outer length.
      if (argT && argT.kind === 'array'
          && Array.isArray(argT.shape) && argT.shape.length > 0
          && typeof argT.shape[0] === 'number') {
        return argT.shape[0];
      }
      // Table: the row count (spec §07 "rows (table)"), carried on the
      // table type as `nrows`. Without this, `lengthof(<table>)` in a
      // shape position (e.g. `zeros(lengthof(t))`) couldn't const-fold
      // — the table type isn't an `array`, so it fell through to the
      // resolver and the dependent shape went %dynamic / deferred.
      if (argT && argT.kind === 'table' && typeof argT.nrows === 'number') {
        return argT.nrows;
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

  // The cat-shape rule (spec §07 `cat` / §06 positional `joint` / §03
  // positional `cartprod`) now lives in `shape-contract.ts` as the single
  // owner — reachable from the materialiser / density layers too, which must
  // not import typeinfer. `catShapeType` is its TYPE projection
  // (`typeOfShape ∘ catShape`): `null` → mixed-kind / unsupported component
  // (the caller diagnoses), a deferred type → under-resolved, else the
  // array/record type. Behaviour-identical to the former inline helper (pinned
  // by test/shape-contract.test.ts + the cat/joint/cartprod type suites).
  function catShapeType(parts: any[]): any {
    return SC.typeOfShape(SC.catShape(parts));
  }

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
        // Engine-concepts §17.1 — const-eval-driven shape resolution.
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
        if (elems.length === 1) return elems[0];
        // Positional cartprod (spec §03): the set of arrays formed by `cat`-ing
        // one element from each component — NOT a tuple. Per the §07 cat rule,
        // components must share a structural kind (all scalar / all vector / all
        // record); mixing is not permitted. (Per-position membership lives in
        // the value-set layer — %unknown for cartprod today.)
        const shaped = catShapeType(elems);
        if (shaped == null) {
          diagnostics.push({
            severity: 'error',
            message: 'cartprod: components must be all scalar sets, all vector '
              + 'sets, or all record sets with distinct fields — mixing structural '
              + 'kinds is not permitted (spec §07 cat)',
            loc: expr.loc,
          });
          return T.failed('cartprod mixed component kinds');
        }
        return shaped;
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

  // ===================================================================
  // Value-set inference (spec §11 third `%meta` slot; engine-concepts
  // §17.3 valueset domain). Classifies every node's value into the
  // strongest statically known §03 set (a measure node's set is its
  // support), via the producer catalogue + a natural-extent fallback
  // (every value-typed node's set is at least its type's extent).
  // Mirrors flatppl-rust `flatppl-infer::call_valueset`; the vocabulary
  // + lattice live in `value-set.ts`. Runs BEFORE the mass pass — the
  // Lebesgue/Counting mass rule consumes set boundedness (§17.3
  // dependency order: valueset < normalization).
  // ===================================================================
  const vsetCache = new Map<string, any>();
  const vsetInProgress = new Set<string>();

  // resolveDim for stdsimplex(n) / cartpow(S, n): try the literal, then
  // the const-eval resolver (length/lengthof short-circuits included).
  const _resolveDim = (ir: any): any => {
    const v = resolveIntegerShape(ir);
    return v != null ? v : null;
  };

  // The §08 Domain/Support column → a value set. Keys on the op name, so
  // it works even where the measure TYPE is still `deferred`.
  // The static length of a rank-1-array node — from its inferred type
  // if annotated, else structurally (an inline `vector(...)` literal, or
  // a ref's binding type). typeinfer doesn't write `meta.type` on every
  // kwarg arg node, so the structural fallbacks are load-bearing.
  function _arrayLenOf(node: any): any {
    if (!node) return '%dynamic';
    const t = node.meta && node.meta.type;
    if (t && t.kind === 'array' && Array.isArray(t.shape) && t.shape.length === 1) return t.shape[0];
    if (node.kind === 'call' && node.op === 'vector' && Array.isArray(node.args)) return node.args.length;
    if (node.kind === 'ref' && loweredModule.bindings.has(node.name)) {
      const bt = loweredModule.bindings.get(node.name).inferredType;
      if (bt && bt.kind === 'array' && Array.isArray(bt.shape) && bt.shape.length === 1) return bt.shape[0];
    }
    return '%dynamic';
  }

  // The static length of a vector-typed parameter (named kwarg or
  // positional), for simplex / cartpow sizes.
  function _paramDim(ir: any, kw: string, posIdx: number): any {
    let arg = ir.kwargs && ir.kwargs[kw];
    if (!arg && Array.isArray(ir.args)) arg = ir.args[posIdx];
    return _arrayLenOf(arg);
  }

  function _vectorDimOf(ir: any): any { return _arrayLenOf((ir.args || [])[0]); }

  // Widen heterogeneous element sets to the strongest named set
  // containing all of them (mirrors Rust `join_scalar_sets`): so a
  // literal weight vector `[0.7, 0.3]` (singleton-interval elements)
  // widens to `nonnegreals`, letting `l1unit`'s simplex guard fire.
  const _VECTOR_JOIN_CANDIDATES = [
    vsLib.POSINTEGERS, vsLib.NONNEGINTEGERS, vsLib.INTEGERS, vsLib.UNITINTERVAL,
    vsLib.POSREALS, vsLib.NONNEGREALS, vsLib.REALS, vsLib.BOOLEANS, vsLib.COMPLEXES,
  ];
  function _joinScalarSets(sets: any[]): any {
    if (sets.length === 0) return null;
    const first = sets[0];
    if (first !== vsLib.UNKNOWN && sets.every((s) => vsLib.equal(s, first))) return first;
    for (const cand of _VECTOR_JOIN_CANDIDATES) {
      if (sets.every((s) => vsLib.subsetOf(s, cand))) return cand;
    }
    return null;
  }

  // Value set of a node, attaching `meta.valueset` on calls. Memoised on
  // the node (and via `vsetOfBinding` for refs). Producer rules +
  // natural-extent fallback.
  function valuesetOfExpr(ir: any): any {
    if (ir && ir.kind === 'call' && ir.meta && ir.meta.valueset !== undefined) {
      return ir.meta.valueset;
    }
    let vs = _computeValueset(ir);
    // Natural-extent fallback: a value-typed node's set is at least its
    // type's extent (spec §11 total discipline).
    if (vs === vsLib.UNKNOWN && ir && ir.kind === 'call' && ir.meta && ir.meta.type) {
      const nat = vsLib.naturalOf(ir.meta.type);
      if (nat !== vsLib.UNKNOWN) vs = nat;
    }
    if (ir && ir.kind === 'call' && ir.meta) ir.meta.valueset = vs;
    return vs;
  }

  function _computeValueset(ir: any): any {
    if (!ir) return vsLib.UNKNOWN;
    if (ir.kind === 'lit') return vsLib.literalValueset(ir);
    if (ir.kind === 'const') return vsLib.constValueset(ir.name);
    if (ir.kind === 'ref') {
      return loweredModule.bindings.has(ir.name) ? vsetOfBinding(ir.name) : vsLib.UNKNOWN;
    }
    if (ir.kind !== 'call') return vsLib.UNKNOWN;
    const op = ir.op;
    const args = ir.args || [];
    switch (op) {
      // Parameters / loaded sets, and reference-measure supports.
      case 'elementof': case 'external':
        return vsLib.setExprValueset(args[0], _resolveDim);
      case 'Lebesgue': case 'Counting':
        return vsLib.setExprValueset(supportArgOf(ir), _resolveDim);
      // Drawing yields a value in the measure's support.
      case 'draw': return valuesetOfExpr(args[0]);
      case 'lawof': return valuesetOfExpr(args[0]);
      // Reweighting / truncation never grows the support.
      case 'normalize': case 'bayesupdate': return valuesetOfExpr(args[0]);
      case 'weighted': case 'logweighted': return valuesetOfExpr(args[1]);
      case 'truncate': {
        const s = vsLib.setExprValueset(args[1], _resolveDim);
        return s === vsLib.UNKNOWN ? valuesetOfExpr(args[0]) : s;
      }
      case 'iid': {
        const inner = valuesetOfExpr(args[0]);
        const t = ir.meta && ir.meta.type;
        if (inner !== vsLib.UNKNOWN && t && t.kind === 'measure'
            && t.domain && t.domain.kind === 'array'
            && Array.isArray(t.domain.shape) && t.domain.shape.length === 1) {
          return vsLib.cartpow(inner, t.domain.shape[0]);
        }
        return vsLib.UNKNOWN;
      }
      // Normalization functions (spec §07).
      case 'softmax': return vsLib.stdsimplex(_vectorDimOf(ir));
      case 'l1unit': {
        const argSet = valuesetOfExpr(args[0]);
        return vsLib.subsetOf(argSet, vsLib.cartpow(vsLib.NONNEGREALS, '%dynamic'))
          ? vsLib.stdsimplex(_vectorDimOf(ir)) : vsLib.UNKNOWN;
      }
      case 'exp': return vsLib.POSREALS;
      case 'abs': case 'abs2': case 'sqrt': return vsLib.NONNEGREALS;
      case 'invlogit': case 'invprobit': return vsLib.UNITINTERVAL;
      case 'vector': {
        const sets = args.map(valuesetOfExpr);
        const elem = _joinScalarSets(sets);
        return elem == null ? vsLib.UNKNOWN
          : vsLib.cartpow(elem, args.length);
      }
      default:
        if (builtins.DISTRIBUTIONS.has(op)) {
          return distributionSupport(ir, { paramDim: _paramDim, resolveDim: _resolveDim });
        }
        return vsLib.UNKNOWN;
    }
  }

  function vsetOfBinding(name: string): any {
    if (vsetCache.has(name)) return vsetCache.get(name);
    if (vsetInProgress.has(name)) return vsLib.UNKNOWN;
    const b = loweredModule.bindings.get(name);
    if (!b || !b.rhs) { vsetCache.set(name, vsLib.UNKNOWN); return vsLib.UNKNOWN; }
    vsetInProgress.add(name);
    const vs = valuesetOfExpr(b.rhs);
    vsetInProgress.delete(name);
    vsetCache.set(name, vs);
    return vs;
  }

  // Fill `meta.valueset` on every call node. Run before fillMasses.
  function fillValuesets() {
    const irWalk = require('./ir-walk.ts');
    const visit = (ir: any) => {
      if (!ir || typeof ir !== 'object') return;
      if (ir.kind === 'call' && ir.meta && ir.meta.valueset === undefined) {
        valuesetOfExpr(ir);
      }
      irWalk.forEachIRChild(ir, visit);
    };
    for (const [, b] of loweredModule.bindings) visit(b.rhs);
  }

  // ===================================================================
  // Domain-contract checks (spec §08 parameter domains; engine-concepts
  // §17.3 — a CONSUMER of the valueset domain). On top of the landed
  // valueset inference: a distribution parameter whose value set is
  // PROVABLY disjoint from the parameter's required domain is a static
  // error (e.g. `Normal(sigma = -1.0)`, `Beta(alpha = 0.0)`). Strictly
  // conservative — fires only on a proven violation (a non-positive real
  // literal, or a binding whose interval lies wholly outside), never on
  // a maybe (a bare `reals`/`unknown`/`deferred` parameter passes).
  //
  // Scalar params only. A vector param's element signs are not provable
  // here — a literal weight vector's value set widens to a named set
  // (`reals`) that has lost per-element negativity, and an array-typed
  // ref carries only its natural `cartpow(reals, …)` extent — so
  // Categorical/Dirichlet/Multinomial simplex contracts wait for a
  // per-element value set (TODO §11). Param names/positions mirror
  // `sampler.REGISTRY` (verified, not guessed).
  const _DOMAIN_CONTRACTS: Record<string, any[]> = {
    Normal:       [{ name: 'sigma', pos: 1, domain: vsLib.POSREALS, label: 'positive (a standard deviation)' }],
    LogNormal:    [{ name: 'sigma', pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    Exponential:  [{ name: 'rate',  pos: 0, domain: vsLib.POSREALS, label: 'positive (a rate)' }],
    Poisson:      [{ name: 'rate',  pos: 0, domain: vsLib.POSREALS, label: 'positive (a rate)' }],
    Gamma:        [{ name: 'shape', pos: 0, domain: vsLib.POSREALS, label: 'positive' },
                   { name: 'rate',  pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    InverseGamma: [{ name: 'shape', pos: 0, domain: vsLib.POSREALS, label: 'positive' },
                   { name: 'scale', pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    Beta:         [{ name: 'alpha', pos: 0, domain: vsLib.POSREALS, label: 'positive' },
                   { name: 'beta',  pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    Weibull:      [{ name: 'shape', pos: 0, domain: vsLib.POSREALS, label: 'positive' },
                   { name: 'scale', pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    Pareto:       [{ name: 'shape', pos: 0, domain: vsLib.POSREALS, label: 'positive' },
                   { name: 'scale', pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    Cauchy:       [{ name: 'scale', pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    Laplace:      [{ name: 'scale', pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    Logistic:     [{ name: 's',     pos: 1, domain: vsLib.POSREALS, label: 'positive (a scale)' }],
    StudentT:     [{ name: 'nu',    pos: 0, domain: vsLib.POSREALS, label: 'positive (degrees of freedom)' }],
    ChiSquared:   [{ name: 'k',     pos: 0, domain: vsLib.POSREALS, label: 'positive (degrees of freedom)' }],
    GeneralizedNormal: [{ name: 'alpha', pos: 1, domain: vsLib.POSREALS, label: 'positive' },
                        { name: 'beta',  pos: 2, domain: vsLib.POSREALS, label: 'positive' }],
    NegativeBinomial:  [{ name: 'alpha', pos: 0, domain: vsLib.POSREALS, label: 'positive' },
                        { name: 'beta',  pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    NegativeBinomial2: [{ name: 'psi',   pos: 1, domain: vsLib.POSREALS, label: 'positive' }],
    Bernoulli:    [{ name: 'p', pos: 0, domain: vsLib.UNITINTERVAL, label: 'in the unit interval [0, 1]' }],
    Geometric:    [{ name: 'p', pos: 0, domain: vsLib.UNITINTERVAL, label: 'in the unit interval [0, 1]' }],
    Binomial:     [{ name: 'p', pos: 1, domain: vsLib.UNITINTERVAL, label: 'in the unit interval [0, 1]' }],
  };

  // A value set PROVABLY disjoint from `domain`. Only an interval value
  // set (the form a real literal `-1.0` → `interval(-1,-1)` takes, or a
  // bounded binding) can prove it — every named set straddles 0. Posreals
  // excludes 0 (so `0.0` violates a `positive` param); nonnegreals admits
  // it; unitinterval admits `[0,1]`.
  function _provablyDisjoint(vs: any, domain: any): boolean {
    if (!vs || typeof vs !== 'object' || vs.vs !== 'interval') return false;
    const lo = vs.lo, hi = vs.hi;
    if (domain === vsLib.POSREALS)     return hi <= 0;
    if (domain === vsLib.NONNEGREALS)  return hi < 0;
    if (domain === vsLib.UNITINTERVAL) return hi < 0 || lo > 1;
    return false;
  }

  function checkDomainContracts() {
    const irWalk = require('./ir-walk.ts');
    const visit = (ir: any) => {
      if (!ir || typeof ir !== 'object') return;
      if (ir.kind === 'call' && _DOMAIN_CONTRACTS[ir.op]) {
        for (const c of _DOMAIN_CONTRACTS[ir.op]) {
          let arg = ir.kwargs && ir.kwargs[c.name];
          if (arg == null && Array.isArray(ir.args)) arg = ir.args[c.pos];
          if (arg == null) continue;
          const vs = valuesetOfExpr(arg);
          if (_provablyDisjoint(vs, c.domain)) {
            diagnostics.push({
              severity: 'error',
              message: `${ir.op}: parameter '${c.name}' must be ${c.label}, but its `
                + `value is provably outside that domain (value set `
                + `${vsLib.toSexpr(vs)}).`,
              loc: (arg.loc || ir.loc),
            });
          }
        }
      }
      irWalk.forEachIRChild(ir, visit);
    };
    for (const [, b] of loweredModule.bindings) visit(b.rhs);
  }

  // ===================================================================
  // Pushfwd domain-restriction guard (spec §06 "Known-bijection registry":
  // "A domain-restricted forward — log/log10 on posreals, sqrt (and pow)
  // on nonnegreals, log1p on interval(-1, inf), logit/probit on
  // interval(0,1) — additionally requires the base measure's support to
  // lie within that domain; where it does not, density evaluation is
  // REFUSED rather than yielding a silently sub-probability measure."
  // #260 (c).
  //
  // The per-op invert/LADJ rules in bijection-registry.ts's
  // ELEMENTARY_BIJECTIONS are domain-AGNOSTIC path-inversion (mirroring
  // the already-shipped `log`/`pow` entries) — nothing there stops
  // `pushfwd(fn(log(_)), Normal(0,1))` from synthesizing a bijection and
  // silently scoring a finite-but-wrong density on the non-positive half
  // of `Normal(0,1)`'s support. This pass is the separate domain check
  // that catches that: for every `pushfwd(f, M)`, collect the
  // domain-restricted ops on `f`'s FREE-VARIABLE inversion path (the ops
  // `invertExpr` actually traverses from the free ref to the output — see
  // `_collectPathOps`; NOT every op in the body, so an op applied to a
  // frozen sub-expression such as `sqrt(5)` in `add(_, sqrt(5))` does not
  // trigger it), and flag an error when `M`'s inferred support is PROVABLY
  // outside a path op's domain. Mirrors invert.rs `derive_chain`'s guard,
  // which is scoped to `flatten_chain`'s output (the linear chain applied
  // to the placeholder), not the whole body.
  //
  // Conservative — mirrors `_provablyDisjoint`'s posture, NOT
  // invert.rs's `is_positive_domain` (which refuses unless support is
  // PROVEN within the domain, treating unknown support as a refusal
  // too): here an UNKNOWN/DEFERRED/ANYTHING support does NOT refuse —
  // only a support POSITIVELY KNOWN to violate the domain does. This
  // keeps the check a strict addition (never spurious on a merely
  // unresolved base) at the cost of not catching every truly-invalid
  // case statically; the corresponding annotation-free lowering already
  // requires a resolvable base measure, so this is not a new gap.
  //
  // `PUSHFWD_DOMAIN_GUARDS` + the `isWithin*` predicates (why `pow` is
  // deliberately excluded, etc.) are module-level, defined + exported below
  // `inferTypes` so `derivations.ts`'s record-field pushforward recognition
  // (#260 d) reuses the SAME table rather than re-deriving it.

  // Resolve a pushfwd forward-arg expression to its single-param
  // `functionof` reification node — an inline `fn(...)`/`functionof(...)`,
  // or a self-ref to one. Mirrors `_pushfwdCodomain`'s identical
  // resolution (this pass runs independently, so it re-resolves rather
  // than threading state through).
  function _resolveForwardReif(fExpr: any): any {
    if (fExpr && fExpr.op === 'functionof') return fExpr;
    if (fExpr && fExpr.kind === 'ref' && fExpr.ns === 'self') {
      const b = loweredModule.bindings.get(fExpr.name);
      if (b && b.rhs && b.rhs.op === 'functionof') return b.rhs;
    }
    return null;
  }

  // Does `ir`'s sub-IR (including `ir` itself) contain a ref to the free
  // variable named `freeName`? (ns-agnostic, matching invertExpr's
  // default free-ref match in bijection-registry.ts.)
  function _irContainsName(ir: any, freeName: string, irWalk: any): boolean {
    if (!ir || typeof ir !== 'object') return false;
    if (ir.kind === 'ref' && ir.name === freeName) return true;
    let found = false;
    irWalk.forEachIRChild(ir, (c: any) => { if (!found) found = _irContainsName(c, freeName, irWalk); });
    return found;
  }

  // The builtin ops on the FREE VARIABLE's inversion path — the ops
  // invertExpr actually traverses from the output down to the free ref
  // (NOT every op in the body). An op applied to a FROZEN sub-expression
  // (off the inversion path, e.g. `sqrt(5)` inside `add(_, sqrt(5))`)
  // never touches the variate, so it must NOT trigger the domain guard.
  // Mirrors invert.rs's guard scoping — `derive_chain` guards `flatten_chain`'s
  // output (the linear chain applied to the placeholder), not the whole body.
  //
  // Walks output→leaf following the single arg that (transitively)
  // contains the free ref, collecting each call op's name. Bails (leaving
  // `into` as-collected-so-far) on a non-straight-line shape — the free
  // var in >1 arg, or a non-call/non-free-ref leaf — exactly the shapes
  // where invertExpr itself returns null (so bijMeta is null and the
  // pushfwd refuses via the case-3 generic path anyway; not guarding a
  // non-invertible shape is safe).
  function _collectPathOps(body: any, freeName: string, into: Set<string>, irWalk: any): void {
    let cur = body;
    while (!(cur && cur.kind === 'ref' && cur.name === freeName)) {
      if (!cur || cur.kind !== 'call' || !Array.isArray(cur.args)) return;
      let onPathIdx = -1;
      for (let i = 0; i < cur.args.length; i++) {
        if (_irContainsName(cur.args[i], freeName, irWalk)) {
          if (onPathIdx !== -1) return;   // free var in >1 arg — not straight-line
          onPathIdx = i;
        }
      }
      if (onPathIdx === -1) return;       // free var absent below here (shouldn't happen once entered)
      if (typeof cur.op === 'string') into.add(cur.op);
      cur = cur.args[onPathIdx];
    }
  }

  function checkPushfwdDomainContracts() {
    const irWalk = require('./ir-walk.ts');
    const visit = (ir: any) => {
      if (!ir || typeof ir !== 'object') return;
      if (ir.kind === 'call' && ir.op === 'pushfwd' && Array.isArray(ir.args) && ir.args.length === 2) {
        _checkOnePushfwdDomain(ir, irWalk);
      }
      irWalk.forEachIRChild(ir, visit);
    };
    for (const [, b] of loweredModule.bindings) visit(b.rhs);
  }

  function _checkOnePushfwdDomain(ir: any, irWalk: any) {
    const reif = _resolveForwardReif(ir.args[0]);
    if (!reif || !Array.isArray(reif.params) || reif.params.length !== 1 || !reif.body) return;
    const ops = new Set<string>();
    _collectPathOps(reif.body, reif.params[0], ops, irWalk);
    const mIR = ir.args[1];
    let support: any = null;   // lazily computed — only needed if a guarded op is present
    for (const op of ops) {
      const guard = PUSHFWD_DOMAIN_GUARDS[op];
      if (!guard) continue;
      if (support == null) support = valuesetOfExpr(mIR);
      if (support === vsLib.UNKNOWN || support === vsLib.DEFERRED || support === vsLib.ANYTHING) continue;
      if (guard.isWithin(support)) continue;
      diagnostics.push({
        severity: 'error',
        message: `pushfwd: forward function uses '${op}', which requires the base measure's `
          + `support to be ${guard.label}, but the base's support is provably `
          + `${vsLib.toSexpr(support)} — refuse rather than silently produce a `
          + `sub-probability measure.`,
        loc: (reif.body.loc || ir.loc),
      });
    }
  }

  // ===================================================================
  // Cross-module reification scope (spec §04 "Reification and module
  // scope"). `functionof` / `kernelof` reify within the CURRENT module
  // only: a parameterized value reached through a loaded-module reference
  // cannot become a reified input — neither by the automatic trace nor as
  // an explicit boundary node (directly or via a pure alias) — so such a
  // reification is a static error. A loaded module's CALLABLES and FIXED
  // values may still be used in the reified DAG (applied, or closed over);
  // only taking a cross-module PARAMETERIZED value as an input is barred.
  // `lawof` is unrestricted (a measure has no input list), so it is not
  // checked here — only the functionof/kernelof input surface is.
  //
  // The phase of a cross-module binding is read from the dependency's
  // already-inferred `LoweredModule` (deps compile before this module —
  // mirrors `resolveUserModuleRef`'s stochastic-boundary check). Only
  // `load_module` deps carry parameterized values; standard-module members
  // are engine-provided callables (always fixed), so they are never
  // flagged. Without a bundle the dep is unresolved → nothing is flagged
  // (editor-lint tolerance, matching the load_module rule).
  // ===================================================================
  function checkCrossModuleReification() {
    if (!modules) return;                       // no compile ctx — can't resolve deps
    const irWalk = require('./ir-walk.ts');

    // Phase of a `(%ref <alias> <name>)` cross-module value, via its
    // load_module dependency — or null if not a resolvable cross-module ref.
    function xmodRefPhase(node: any): string | null {
      if (!node || node.kind !== 'ref') return null;
      const ns = node.ns;
      if (!ns || ns === 'self' || ns === 'base' || ns === '%local') return null;
      const reg = loweredModule.moduleRegistry && loweredModule.moduleRegistry[ns];
      if (!reg || reg.kind !== 'load_module' || !reg.path) return null;
      const dep = modules.get(reg.path);
      if (!dep) return null;
      const db = dep.loweredModule.bindings.get(node.name);
      return db ? (db.phase || null) : null;
    }

    // Follow a node through PURE alias bindings (RHS is a bare ref) to its
    // terminal — so a boundary `p = a` with `a = m.theta` resolves to
    // `m.theta`, while `p = y` with `y = m.theta * 2` stops at the derived
    // current-module node `y` (which correctly CUTS the cross-module dep).
    function resolvePureAlias(node: any): any {
      const seen = new Set<string>();
      let cur = node;
      while (cur && cur.kind === 'ref' && cur.ns === 'self' && !seen.has(cur.name)) {
        seen.add(cur.name);
        const b = loweredModule.bindings.get(cur.name);
        if (!b || !b.rhs || b.rhs.kind !== 'ref') break;
        cur = b.rhs;
      }
      return cur;
    }

    // The cross-module parameterized value a node designates (after pure
    // aliasing), or null. Callables (fixed) and fixed values are excluded
    // by the phase test — exactly the spec's "callables and fixed values
    // may be used" allowance.
    function xmodParamTarget(node: any): any {
      const t = resolvePureAlias(node);
      return xmodRefPhase(t) === 'parameterized' ? t : null;
    }

    function reportXmod(t: any, loc: any, how: string) {
      diagnostics.push({
        severity: 'error',
        message: "'" + t.ns + '.' + t.name + "' is a parameterized value from a loaded "
          + 'module and cannot become a reified input (' + how + '). `functionof` / '
          + '`kernelof` reify within the current module only — a loaded module’s '
          + 'callables and fixed values may be used in a reified DAG, but not its '
          + 'parameterized values (spec §04 "Reification and module scope").',
        loc,
      });
    }

    // Automatic-trace clause (no explicit boundary): the trace back to
    // parametric leaves would make any cross-module parameterized value an
    // input. Walk the reified body's ancestor closure WITHIN this module —
    // follow self-refs into their RHS (which also follows aliases), stop at
    // cross-module refs / placeholders, and do NOT descend into a nested
    // reification's body (its own scope, checked when its node is visited).
    function walkAutoTrace(body: any, reifLoc: any) {
      const seenBindings = new Set<string>();
      const reported = new Set<string>();
      const stack: any[] = [body];
      while (stack.length) {
        const node = stack.pop();
        if (!node || typeof node !== 'object') continue;
        if (node.kind === 'ref') {
          if (node.ns === 'self') {
            if (seenBindings.has(node.name)) continue;
            seenBindings.add(node.name);
            const b = loweredModule.bindings.get(node.name);
            if (b && b.rhs) stack.push(b.rhs);
            continue;
          }
          const t = xmodParamTarget(node);       // cross-module parametric leaf?
          if (t) {
            const key = t.ns + '.' + t.name;
            if (!reported.has(key)) {
              reported.add(key);
              reportXmod(t, reifLoc, 'reached by the automatic trace');
            }
          }
          continue;                              // %local / base / cross-module: leaf
        }
        if (node.kind === 'call' && node.op === 'functionof') continue;  // nested scope
        irWalk.forEachIRChild(node, (c: any) => stack.push(c));
      }
    }

    const visit = (ir: any) => {
      if (!ir || typeof ir !== 'object') return;
      if (ir.kind === 'call' && ir.op === 'functionof') {
        const hasBoundaries = Array.isArray(ir.params) && ir.params.length > 0;
        if (hasBoundaries) {
          // Explicit-boundary clause. A boundary's origin is `paramSources`
          // (a bare FieldAccess boundary like `p = m.theta` is rejected at
          // lowering, so a cross-module boundary is always a local-binding
          // source that pure-aliases the cross-module value, e.g.
          // `a = m.theta; functionof(…, p = a)`). A `placeholder` source is
          // a fresh formal — never cross-module. A `binding` source whose
          // pure-alias chain lands on a cross-module parameterized value
          // names it as an input → error; a DERIVED local binding
          // (`y = m.theta * 2`) stops the alias resolution at `y`, correctly
          // cutting the cross-module dependency.
          const srcs = ir.paramSources || [];
          const kws = ir.paramKwargs || [];
          for (let i = 0; i < srcs.length; i++) {
            const src = srcs[i];
            if (!src || src.kind !== 'binding') continue;
            const t = xmodParamTarget({ kind: 'ref', ns: 'self', name: src.name });
            if (t) reportXmod(t, ir.loc,
              "named as the explicit boundary input '" + (kws[i] || src.name) + "'");
          }
        } else {
          walkAutoTrace(ir.body, ir.loc);
        }
        // fall through: descend to find NESTED reifications (own scope/check)
      }
      irWalk.forEachIRChild(ir, visit);
    };
    for (const [, b] of loweredModule.bindings) visit(b.rhs);
  }

  // ===================================================================
  // Mass-class inference (spec §11 "Total-mass classes"; engine-concepts
  // §17.3 normalization domain). A second pass after the type walk: for
  // every measure-typed binding, classify its total mass by composing
  // the per-op rules over the measure expression, and emit the spec-§06
  // static error when `normalize` is applied to a measure whose mass is
  // null or (locally-)infinite. Mirrors flatppl-rust's `flatppl-infer`
  // `fill_mass`; the structural recursion parallels the numeric
  // `derivations.closedFormLogTotalmass` (two domains over the same §17
  // walk — a class here, a log-number there; a property test pins their
  // agreement). Conservative throughout: an unrecognised shape is
  // MASS_UNKNOWN, never a guessed class, so the normalize diagnostic
  // cannot false-positive.
  // ===================================================================
  const massCache = new Map<string, any>();
  const massInProgress = new Set<string>();

  // Boundedness of a set expression (spec §03), for the Lebesgue /
  // Counting mass rule — read through the valueset layer (§17.3: mass
  // CONSUMES set facts; the set vocabulary + boundedness lattice have
  // ONE owner, `value-set.ts`). true / false / null (unknown).
  function setBounded(setIR: any): boolean | null {
    return vsLib.isBounded(vsLib.setExprValueset(setIR, _resolveDim));
  }

  // A fixed-phase scalar weight rescales by a constant — preserving the
  // mass class (modulo normalized→finite). A parameterised / stochastic
  // / function / inline weight does not qualify (spec §06 + Rust rule).
  function isFixedScalarWeight(weightIR: any): boolean {
    if (!weightIR) return false;
    if (weightIR.kind === 'lit' && typeof weightIR.value === 'number') return true;
    if (weightIR.kind === 'ref' && loweredModule.bindings.has(weightIR.name)) {
      const b = loweredModule.bindings.get(weightIR.name);
      return !!(b && b.phase === 'fixed'
        && b.inferredType && b.inferredType.kind === 'scalar');
    }
    return false;
  }

  // Mass class of a measure expression, composing the per-op rules.
  // Wrapper: memoise on the node (so a measure node is classified — and
  // its normalize diagnostic fired — exactly once across both fill
  // passes) and stamp the class onto the call's `meta.type` for %meta
  // emission.
  function massOfExpr(ir: any): any {
    if (ir && ir.kind === 'call' && ir.meta && ir.meta.type
        && ir.meta.type.kind === 'measure' && ir.meta.type.mass !== undefined) {
      return ir.meta.type.mass;
    }
    const m = _computeMass(ir);
    if (ir && ir.kind === 'call' && ir.meta && ir.meta.type
        && ir.meta.type.kind === 'measure') {
      ir.meta.type.mass = m;
    }
    return m;
  }

  function _computeMass(ir: any): any {
    if (!ir) return T.MASS_UNKNOWN;
    if (ir.kind === 'ref' && loweredModule.bindings.has(ir.name)) {
      return massOfBinding(ir.name);
    }
    if (ir.kind !== 'call') return T.MASS_UNKNOWN;
    const op = ir.op;
    const args = ir.args || [];

    // Reference measures: finite on a bounded support, locally finite
    // on an unbounded one.
    if (op === 'Lebesgue' || op === 'Counting') {
      const b = setBounded(supportArgOf(ir));
      return b === true ? T.MASS_FINITE
        : b === false ? T.MASS_LOCALLY_FINITE
        : T.MASS_UNKNOWN;
    }
    // Every other distribution constructor — incl. Dirac, the named §08
    // distributions, and the Poisson processes (proper probability
    // measures over point configurations) — is normalized.
    if (builtins.DISTRIBUTIONS.has(op)) return T.MASS_NORMALIZED;

    switch (op) {
      // `lawof(x)` reifies the total law of a variate: a probability measure.
      case 'lawof': return T.MASS_NORMALIZED;
      case 'normalize': {
        const base = massOfExpr(args[0]);
        if (base === T.MASS_NULL) {
          diagnostics.push({ severity: 'error',
            message: 'normalize: a measure with zero total mass cannot be normalized (spec §06)',
            loc: ir.loc });
        } else if (base === T.MASS_LOCALLY_FINITE) {
          diagnostics.push({ severity: 'error',
            message: 'normalize: a measure with infinite total mass cannot be normalized (spec §06)',
            loc: ir.loc });
        }
        return T.MASS_NORMALIZED;
      }
      // iid is a homomorphism on the mass class (Nᵗʰ power preserves it).
      case 'iid': {
        const base = massOfExpr(args[0]);
        return (base === T.MASS_NORMALIZED || base === T.MASS_NULL
          || base === T.MASS_FINITE || base === T.MASS_LOCALLY_FINITE)
          ? base : T.MASS_UNKNOWN;
      }
      // Pushforward is mass-preserving: (f∗M)(whole) = M(whole). Measure
      // is arg 1 (`pushfwd(f, M)`); `locscale(m, …)` carries it at arg 0.
      case 'pushfwd':  return massOfExpr(args[1]);
      case 'locscale': return massOfExpr(args[0]);
      // weighted(w, M) / logweighted(g, M): a fixed scalar weight keeps
      // the class (normalized demotes to finite — the scale is no longer
      // one); any other weight is unknown.
      case 'weighted':
      case 'logweighted': {
        const base = massOfExpr(args[1]);
        if (base === T.MASS_NULL) return T.MASS_NULL;
        if (isFixedScalarWeight(args[0])) {
          if (base === T.MASS_NORMALIZED || base === T.MASS_FINITE) return T.MASS_FINITE;
          if (base === T.MASS_LOCALLY_FINITE) return T.MASS_LOCALLY_FINITE;
          return T.MASS_UNKNOWN;
        }
        return T.MASS_UNKNOWN;
      }
      // truncate(M, S) demotes a finite/normalized measure to finite;
      // a locally finite measure becomes finite only on a bounded S.
      case 'truncate': {
        const base = massOfExpr(args[0]);
        if (base === T.MASS_NULL) return T.MASS_NULL;
        if (base === T.MASS_NORMALIZED || base === T.MASS_FINITE) return T.MASS_FINITE;
        if (base === T.MASS_LOCALLY_FINITE) {
          return setBounded(args[1]) === true ? T.MASS_FINITE : T.MASS_UNKNOWN;
        }
        return T.MASS_UNKNOWN;
      }
      // superpose is measure addition; select is its discrete-mixture
      // sibling — both combine additively.
      case 'superpose': return additiveMass(args.map(massOfExpr));
      case 'select': {
        const br = ir.branches || [];
        const masses = br.map((b: any) =>
          massOfExpr(b && b.ir ? b.ir : (b && b.kind ? b
            : (b && b.ref ? { kind: 'ref', ns: 'self', name: b.ref } : null))));
        return additiveMass(masses);
      }
      // Independent product (joint / record-of-measures).
      case 'joint':
      case 'record': {
        const comps = Array.isArray(ir.fields)
          ? ir.fields.map((f: any) => f.value) : args;
        return productMass(comps.map(massOfExpr));
      }
      // bayesupdate's mass is the evidence integral — statically unknown.
      // Dependent chains: conservative (the kernel arg is kernel-typed,
      // so its output mass isn't reachable here without unwrapping —
      // matches Rust's effective behaviour; refine when a consumer needs
      // it).
      case 'bayesupdate':
      case 'jointchain':
      case 'kchain':
        return T.MASS_UNKNOWN;
    }
    return T.MASS_UNKNOWN;
  }

  // Mass of a measure binding (memoised, cycle-guarded — refs resolve here).
  function massOfBinding(name: string): any {
    if (massCache.has(name)) return massCache.get(name);
    if (massInProgress.has(name)) return T.MASS_UNKNOWN;     // cycle → conservative
    const b = loweredModule.bindings.get(name);
    if (!b || !b.inferredType || b.inferredType.kind !== 'measure') {
      massCache.set(name, T.MASS_UNKNOWN);
      return T.MASS_UNKNOWN;
    }
    massInProgress.add(name);
    const m = massOfExpr(b.rhs);
    massInProgress.delete(name);
    massCache.set(name, m);
    return m;
  }

  // Fill the `mass` slot on every measure node and run the normalize
  // diagnostic. Idempotent; run once after the type walk.
  function fillMasses() {
    const irWalk = require('./ir-walk.ts');
    // Pass 1 — measure / kernel bindings. A measure binding's class
    // fills its top `inferredType`; a kernel's reified body (whose
    // `meta.type` aliases the kernel's `result` measure) fills the
    // kernel's output-mass class.
    for (const [name, b] of loweredModule.bindings) {
      if (b.inferredType && b.inferredType.kind === 'measure') {
        b.inferredType.mass = massOfBinding(name);
      } else if (b.inferredType && b.inferredType.kind === 'kernel'
          && b.inferredType.result && b.inferredType.result.kind === 'measure'
          && b.rhs && b.rhs.body) {
        massOfExpr(b.rhs.body);
      }
    }
    // Pass 2 — every remaining inner measure node (the distribution
    // inside a `draw`, a likelihood / bayesupdate arg, a kernel body's
    // sub-measures) for complete %meta emission, and so a normalize
    // nested in a value binding still raises its diagnostic. The node
    // memo in `massOfExpr` keeps this idempotent with pass 1.
    const visit = (ir: any) => {
      if (!ir || typeof ir !== 'object') return;
      if (ir.kind === 'call' && ir.meta && ir.meta.type
          && ir.meta.type.kind === 'measure' && ir.meta.type.mass === undefined) {
        massOfExpr(ir);
      }
      irWalk.forEachIRChild(ir, visit);
    };
    for (const [, b] of loweredModule.bindings) visit(b.rhs);
  }

  return { diagnostics, inferBinding, inferExpr, fillValuesets, fillMasses, checkDomainContracts,
    checkPushfwdDomainContracts, checkCrossModuleReification };
}

// Mass of an independent product of components (spec §11; Rust
// `product_mass`). Null taints; all-normalized stays normalized; a
// finite factor demotes to finite; a locally-finite factor to locally
// finite; anything unknown is unknown.
function productMass(masses: any[]): any {
  if (masses.some((m) => m === T.MASS_NULL)) return T.MASS_NULL;
  if (masses.every((m) => m === T.MASS_NORMALIZED)) return T.MASS_NORMALIZED;
  if (masses.every((m) => m === T.MASS_NORMALIZED || m === T.MASS_FINITE)) return T.MASS_FINITE;
  if (masses.every((m) =>
    m === T.MASS_NORMALIZED || m === T.MASS_FINITE || m === T.MASS_LOCALLY_FINITE)) {
    return T.MASS_LOCALLY_FINITE;
  }
  return T.MASS_UNKNOWN;
}

// Mass of a measure SUM (superpose / select). Unlike the product, a sum
// of unit masses is k (finite, not normalized): any unknown → unknown;
// any locally-finite → locally finite; all null → null; otherwise finite.
function additiveMass(masses: any[]): any {
  if (masses.length === 0) return T.MASS_NULL;
  if (masses.some((m) => m === T.MASS_UNKNOWN || m === undefined)) return T.MASS_UNKNOWN;
  if (masses.some((m) => m === T.MASS_LOCALLY_FINITE)) return T.MASS_LOCALLY_FINITE;
  if (masses.every((m) => m === T.MASS_NULL)) return T.MASS_NULL;
  return T.MASS_FINITE;
}

// Internal "set" marker — not a user-facing type. elementof handles it.
function setMarker(name: any) { return { kind: 'set', name }; }

// =====================================================================
// Module-level reuse surface (#260 d): the per-distribution scalar
// support table + the pushfwd domain-restriction guards, EXPORTED so
// `derivations.ts`'s record-field diagonal-pushforward recognition
// reuses the identical catalogue `checkPushfwdDomainContracts` uses for
// an explicit `pushfwd(...)` node, rather than re-deriving distribution
// supports or domain rules for the implicit (record-field) case. Zero
// closure dependency beyond `vsLib` (module-level, top of file), so
// these live outside `inferTypes` and are called from inside it too
// (one call site each — `_computeValueset`'s default branch,
// `_checkOnePushfwdDomain`) with no behaviour change.
// =====================================================================

// Lebesgue/Counting take their support as a `support` kwarg or a lone
// positional arg (spec §06). Shared by the valueset + mass rules.
function supportArgOf(ir: any): any {
  if (ir.kwargs && ir.kwargs.support) return ir.kwargs.support;
  return (ir.args || [])[0] || null;
}

// Per-distribution support (spec §08). `opts.paramDim`/`opts.resolveDim`
// are optional loweredModule-bound resolvers (array length / non-literal
// dimension lookups) that only `typeinfer`'s own call site can supply;
// a caller without a `LoweredModule` in scope (derivations.ts querying a
// record field's SCALAR stochastic ancestor) omits them — the Uniform /
// MvNormal / Dirichlet / Multinomial branches then degrade to whatever
// `vsLib.setExprValueset`'s literal-only fallback resolves (Uniform with
// literal bounds still resolves fine) or `vsLib.UNKNOWN` (vector dists —
// never a scalar bijection's ancestor in practice, so inert there).
function distributionSupport(
  ir: any, opts?: { paramDim?: (ir: any, kw: string, posIdx: number) => any, resolveDim?: (ir: any) => any },
): any {
  const op = ir.op;
  switch (op) {
    case 'Uniform': return vsLib.setExprValueset(supportArgOf(ir), opts && opts.resolveDim);
    case 'Normal': case 'GeneralizedNormal': case 'Cauchy': case 'StudentT':
    case 'Logistic': case 'VonMises': case 'Laplace': return vsLib.REALS;
    case 'LogNormal': case 'InverseGamma':
    case 'Pareto': return vsLib.POSREALS;
    // Gamma/ChiSquared density is nonzero at x=0 (Gamma(1,β)=Exponential;
    // shape≤1 is finite/diverges there), so 0 is in the §08 support —
    // nonnegreals, like Exponential/Weibull. This is the DENSITY support
    // only; the HMC unconstraining transform still treats them as the
    // positive half-line (transforms.ts SUPPORT_BY_DIST), the x=0 boundary
    // being measure-zero — mirroring how Exponential is already handled.
    case 'Exponential': case 'Weibull':
    case 'Gamma': case 'ChiSquared': return vsLib.NONNEGREALS;
    case 'Beta': return vsLib.UNITINTERVAL;
    case 'Bernoulli': return vsLib.BOOLEANS;
    case 'Categorical': return vsLib.POSINTEGERS;
    case 'Categorical0': case 'Binomial': case 'Geometric':
    case 'NegativeBinomial': case 'NegativeBinomial2': case 'Poisson':
      return vsLib.NONNEGINTEGERS;
    case 'MvNormal':
      return (opts && opts.paramDim) ? vsLib.cartpow(vsLib.REALS, opts.paramDim(ir, 'mu', 0)) : vsLib.UNKNOWN;
    case 'Dirichlet':
      return (opts && opts.paramDim) ? vsLib.stdsimplex(opts.paramDim(ir, 'alpha', 0)) : vsLib.UNKNOWN;
    case 'Multinomial':
      return (opts && opts.paramDim) ? vsLib.cartpow(vsLib.NONNEGINTEGERS, opts.paramDim(ir, 'p', 1)) : vsLib.UNKNOWN;
    default: return vsLib.UNKNOWN;
  }
}

// `pow` is deliberately NOT included even though spec §06 lists it alongside
// `sqrt` (nonnegreals): the already-shipped (#260 (a)) `pow` entry is
// exercised by a passing regression test using an odd literal exponent over
// a REALS-support base (`pushfwd(fn(pow(_, 3)), Normal(0, 1))`), which a
// nonnegreals guard would newly refuse. Odd-integer `pow` is genuinely
// invertible over all of ℝ; a parity-aware guard is the correct long-term
// fix but out of scope here — tracked as a known gap.
const PUSHFWD_DOMAIN_GUARDS: Record<string, { isWithin: (vs: any) => boolean, label: string }> = {
  log:    { isWithin: isWithinPositiveDomain, label: 'positive (posreals; spec §07 log domain)' },
  log10:  { isWithin: isWithinPositiveDomain, label: 'positive (posreals; spec §07 log10 domain)' },
  sqrt:   { isWithin: isWithinPositiveDomain, label: 'non-negative (nonnegreals; spec §07 sqrt domain)' },
  log1p:  { isWithin: isWithinGtNegOneDomain, label: 'within (-1, inf) (spec §07 log1p domain)' },
  logit:  { isWithin: isWithinUnitDomain, label: 'within (0, 1) (spec §07 logit domain)' },
  probit: { isWithin: isWithinUnitDomain, label: 'within (0, 1) (spec §07 probit domain)' },
};

// Mirrors invert.rs `is_positive_domain`: TRUE for the continuous sets
// whose boundary at 0 carries no probability mass (posreals, nonnegreals,
// unitinterval, or a known interval with lo >= 0) — i.e. "provably
// nonnegative, continuous". Used for both log/log10 (spec: posreals) and
// sqrt (spec: nonnegreals): the boundary point is measure-zero either way.
function isWithinPositiveDomain(vs: any): boolean {
  if (vs === vsLib.POSREALS || vs === vsLib.NONNEGREALS || vs === vsLib.UNITINTERVAL) return true;
  if (vs && vs.vs === 'interval') return vs.lo >= 0;
  return false;
}
// Mirrors invert.rs `is_gt_neg_one_domain`: `support ⊆ (−1, ∞)`.
function isWithinGtNegOneDomain(vs: any): boolean {
  if (vs === vsLib.POSREALS || vs === vsLib.NONNEGREALS || vs === vsLib.UNITINTERVAL) return true;
  if (vs && vs.vs === 'interval') return vs.lo >= -1;
  return false;
}
// Mirrors invert.rs `is_unit_domain`: `support ⊆ (0, 1)`.
function isWithinUnitDomain(vs: any): boolean {
  if (vs === vsLib.UNITINTERVAL) return true;
  if (vs && vs.vs === 'interval') return vs.lo >= 0 && vs.hi <= 1;
  return false;
}

module.exports = {
  inferTypes, inferExprInScope,
  distributionSupport, supportArgOf, PUSHFWD_DOMAIN_GUARDS,
  isWithinPositiveDomain, isWithinGtNegOneDomain, isWithinUnitDomain,
};
