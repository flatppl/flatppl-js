'use strict';

// =====================================================================
// fixed-eval.ts — demand-driven const-evaluation for type/shape inference.
// =====================================================================
//
// Engine-concepts §17.4 ("resolve, don't rewrite"): type inference may
// need fixed-phase values to fold shape positions. This module is the
// adapter that lets typeinfer ask "what's the value of this IR?"
// without itself importing the value-mode evaluator.
//
// The resolver is **demand-driven** — equivalent to a query system
// (rustc / salsa), Haskell lazy evaluation, or Idris/Agda normal-order
// reduction. Bindings are NOT eagerly evaluated; they're only walked
// when a shape position transitively needs them. This matters when
// the module contains expensive fixed-phase work (`A =
// load_huge_matrix()`, `B = expensive_op(A)`) whose value isn't
// needed for any shape — we never run it.
//
// Shape-observer short-circuit: `length(x)` / `lengthof(x)` /
// `sizeof(x)` are intercepted *before* the operand is evaluated.
// When the operand is a binding whose inferredType carries a literal
// shape, the result is read off the TYPE — `sizeof(B)` returns
// `[m, n]` without ever materialising B. This is the load-bearing
// optimisation that keeps the const-eval cheap on real models.
//
// Why custom evaluator vs delegating everything to
// `sampler.evaluateExpr`: evaluateExpr walks an IR depth-first; we
// need to INTERCEPT the walk at shape-observer calls so the operand
// isn't materialised. Hybrid design: dispatch the recursive walk
// ourselves; rebuild a synthetic IR with already-evaluated operands
// as literals; hand THAT to evaluateExpr for the actual op
// computation. Best of both — evaluateExpr stays the single
// authority for per-op semantics; the short-circuits live where
// they're called.

const samplerLib = require('./sampler.ts');

/**
 * Build a demand-driven const-eval resolver typeinfer can use at
 * shape positions. Returns a callable `(ir, env?) → value | undefined`.
 * Undefined means "couldn't statically resolve" — typeinfer treats
 * that as %dynamic; nothing was rewritten in either case.
 *
 *   - `loweredModule` — bindings to follow self-refs through; the
 *     resolver also consults each binding's `inferredType` for the
 *     shape-observer short-circuit. Optional — without it, only
 *     self-contained IR (no self-refs) is resolvable.
 *   - `baseEnv` — atom-independent env (host's session env); merged
 *     in as the outermost lookup layer. Optional.
 */
function makeResolver(opts?: { loweredModule?: any; baseEnv?: any }) {
  const loweredModule = opts && opts.loweredModule;
  const baseEnv: any = (opts && opts.baseEnv) || {};
  const cache = new Map<string, any>();        // binding name → resolved value (undefined ⇒ couldn't)
  const visiting = new Set<string>();          // cycle protection

  function evalIR(ir: any, env?: any): any | undefined {
    if (!ir) return undefined;
    if (ir.kind === 'lit') return ir.value;
    if (ir.kind === 'ref' && ir.ns === 'self') {
      const name = ir.name;
      // Ref lookup precedence: caller env > baseEnv > binding cache /
      // recursive resolution. Caller env wins so a local override
      // (e.g. axis vars within an aggregate body) is honoured if
      // typeinfer ever passes one.
      if (env && Object.prototype.hasOwnProperty.call(env, name)) return env[name];
      if (Object.prototype.hasOwnProperty.call(baseEnv, name)) return baseEnv[name];
      return resolveBinding(name);
    }
    if (ir.kind === 'call') {
      // Shape-observer short-circuit: read the result off the
      // operand's INFERRED TYPE without recursing into its value.
      // The single most important pattern for keeping const-eval
      // cheap (engine-concepts §17.4).
      const sc = _shapeObserverShortCircuit(ir);
      if (sc !== undefined) return sc;
      // General path: evaluate args (recursively, with short-circuits
      // firing for any sub-expression that hits them), then dispatch
      // the op via sampler.evaluateExpr on a synthesised IR with
      // literal operands.
      return _evalCall(ir, env);
    }
    return undefined;
  }

  function resolveBinding(name: string): any | undefined {
    if (cache.has(name)) return cache.get(name);
    if (visiting.has(name)) return undefined;   // cyclic
    if (!loweredModule || !loweredModule.bindings) return undefined;
    const b = loweredModule.bindings.get(name);
    if (!b || !b.rhs) return undefined;
    visiting.add(name);
    const v = evalIR(b.rhs);
    visiting.delete(name);
    cache.set(name, v);
    return v;
  }

  // Returns a value when the call is a shape-observer whose operand
  // has a statically-known shape; otherwise returns undefined and
  // the caller falls through to general evaluation.
  //
  // Two operand shapes both supported (engine-concepts §17.4 — the
  // short-circuit must apply uniformly):
  //  (a) Self-ref to a binding — read shape from binding's inferredType.
  //  (b) Inline call (e.g. `length(rowstack(...))`) — typeinfer wrote
  //      the operand's type into its `meta.type` slot; read from there.
  // Either path avoids materialising the operand.
  function _shapeObserverShortCircuit(ir: any): any | undefined {
    if (!Array.isArray(ir.args) || ir.args.length !== 1) return undefined;
    if (ir.op !== 'length' && ir.op !== 'lengthof' && ir.op !== 'sizeof') return undefined;
    const arg = ir.args[0];
    if (!arg) return undefined;
    const t = _operandType(arg);
    if (!t || t.kind !== 'array' || !Array.isArray(t.shape)) return undefined;
    const allKnown = t.shape.every((d: any) => typeof d === 'number');
    if (!allKnown) return undefined;
    if (ir.op === 'sizeof') {
      // sizeof returns the dim vector as a rank-1 Value (engine
      // contract §2.1; matches sampler.ARITH_OPS.sizeof).
      const data = new Float64Array(t.shape.length);
      for (let i = 0; i < t.shape.length; i++) data[i] = t.shape[i];
      return { shape: [t.shape.length], data };
    }
    return t.shape[0];   // length / lengthof
  }

  // Resolve the static type of an operand IR. Ref → binding's
  // inferredType; any other expr → its meta.type (written by
  // typeinfer's per-op handlers during the inference pass).
  function _operandType(ir: any): any | undefined {
    if (ir.kind === 'ref' && ir.ns === 'self') {
      const b = loweredModule && loweredModule.bindings && loweredModule.bindings.get(ir.name);
      return b && b.inferredType;
    }
    return ir && ir.meta && ir.meta.type;
  }

  function _evalCall(ir: any, env?: any): any | undefined {
    const args = ir.args || [];
    // Evaluate each positional arg recursively; short-circuit on
    // first failure (`undefined`).
    const evaledArgs: any[] = new Array(args.length);
    for (let i = 0; i < args.length; i++) {
      const v = evalIR(args[i], env);
      if (v === undefined) return undefined;
      evaledArgs[i] = v;
    }
    // Same for kwargs.
    let evaledKwargs: Record<string, any> | undefined;
    if (ir.kwargs) {
      evaledKwargs = {};
      for (const k in ir.kwargs) {
        if (!Object.prototype.hasOwnProperty.call(ir.kwargs, k)) continue;
        const v = evalIR(ir.kwargs[k], env);
        if (v === undefined) return undefined;
        evaledKwargs[k] = v;
      }
    }
    // Synthesise a literal-operand IR and dispatch through
    // sampler.evaluateExpr. evaluateExpr stays the single authority
    // for per-op semantics (ARITH_OPS, get/get_field/get0, record,
    // tuple, polynomial, rand, etc.); we just bypass its operand
    // walk because we've already done it.
    const synthArgs = evaledArgs.map((v) => ({ kind: 'lit', value: v }));
    const synthIR: any = { kind: 'call', op: ir.op, args: synthArgs };
    if (evaledKwargs) {
      const sk: Record<string, any> = {};
      for (const k in evaledKwargs) sk[k] = { kind: 'lit', value: evaledKwargs[k] };
      synthIR.kwargs = sk;
    }
    try { return samplerLib.evaluateExpr(synthIR, baseEnv); }
    catch { return undefined; }
  }

  // Test/inspection surface
  (evalIR as any).knownFixed = cache;
  return evalIR as any;
}

module.exports = { makeResolver };
