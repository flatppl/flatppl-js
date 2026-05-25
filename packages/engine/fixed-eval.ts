'use strict';

// =====================================================================
// fixed-eval.ts — const-evaluation shim for type/shape inference.
// =====================================================================
//
// Engine-concepts §17.4 ("resolve, don't rewrite"): type inference may
// need to compute fixed-phase values to resolve shape positions
// (`iid(M, n)`, `cartpow(set, length(data))`, etc.). The actual
// evaluation work is delegated to the existing deterministic
// evaluator `sampler.evaluateExpr` — same code path, same arithmetic,
// no re-implementation. This shim wraps that call with the
// "try-eval, return undefined on failure" semantics typeinfer needs.
//
// Why the indirection: typeinfer.ts is independent of value-mode
// code (it imports only `types.ts` and `builtins.ts`). Letting it
// `require('./sampler.ts')` would break that layering. Instead, the
// caller (e.g. analyzer.ts) imports both this shim and typeinfer,
// then passes the resolver as an optional opt — `flatppl-eval` ↔
// `flatppl-ir` style dependency-injection, just inside one repo.
//
// Per the §17.4 principle: this shim NEVER mutates the IR. It returns
// values; typeinfer decides what to do with them (typically: embed
// the literal integer into a type-level shape annotation; never
// rewrite the source IR).

const samplerLib = require('./sampler.ts');

/**
 * Make a resolver callback typeinfer can use to evaluate const
 * expressions on demand. Returns a function `(ir, env) → value | undefined`.
 *
 *   - `loweredModule` — the module being inferred, so the resolver
 *     can look up self-refs against bindings it has already
 *     successfully evaluated. Optional — without it, ref lookups
 *     consult only the env arg.
 *   - `baseEnv` — atom-independent env to merge in (session env
 *     from the host, etc.). Optional.
 *
 * The returned resolver carries a `knownFixed` Map on itself so
 * callers (typeinfer) can both consult and grow the cache during
 * a single walk.
 */
function makeResolver(opts?: { loweredModule?: any; baseEnv?: any }) {
  const loweredModule = opts && opts.loweredModule;
  const baseEnv = (opts && opts.baseEnv) || {};
  const knownFixed = new Map<string, any>();

  function tryEval(ir: any, env?: Record<string, any>): any | undefined {
    if (!ir) return undefined;
    // Literal short-circuit — no need to dispatch into the full
    // evaluator for the common case.
    if (ir.kind === 'lit') return ir.value;
    // Self-ref short-circuit: known bindings come straight from cache.
    if (ir.kind === 'ref' && ir.ns === 'self' && knownFixed.has(ir.name)) {
      return knownFixed.get(ir.name);
    }
    // Otherwise delegate to the existing deterministic evaluator.
    // The try-catch turns "evaluator threw" (unbound ref, undefined
    // op, type error) into a clean "couldn't resolve" — typeinfer
    // falls back to %dynamic rather than crashing.
    const callEnv: any = Object.assign({}, baseEnv);
    // Pour knownFixed entries into the env so the evaluator sees
    // already-resolved self-refs without an extra hop.
    for (const [k, v] of knownFixed) callEnv[k] = v;
    if (env) Object.assign(callEnv, env);
    try {
      return samplerLib.evaluateExpr(ir, callEnv);
    } catch {
      return undefined;
    }
  }

  /**
   * Try to evaluate a binding's RHS and stash the result for later
   * lookups. Called by typeinfer after each binding's type has been
   * inferred. No-op (returns undefined) if the RHS can't currently
   * be resolved (refs to unresolved bindings, etc.).
   */
  function tryEvalBinding(name: string): any | undefined {
    if (knownFixed.has(name)) return knownFixed.get(name);
    if (!loweredModule) return undefined;
    const b = loweredModule.bindings && loweredModule.bindings.get(name);
    if (!b || !b.rhs) return undefined;
    const v = tryEval(b.rhs);
    if (v !== undefined) knownFixed.set(name, v);
    return v;
  }

  (tryEval as any).tryEvalBinding = tryEvalBinding;
  (tryEval as any).knownFixed = knownFixed;
  return tryEval as any;
}

module.exports = { makeResolver };
