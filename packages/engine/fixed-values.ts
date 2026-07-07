'use strict';

// =====================================================================
// fixed-values.ts — demand-driven fixed-phase VALUE resolver.
// =====================================================================
//
// Engine-concepts §17.1 ("demand-driven on both sides"). A fixed-phase
// binding is compile-time determinate, so its value CAN be computed
// without sampling. The question is WHEN. Historically `buildDerivations`
// ran an eager `while (progress) { for (binding of bindings) … }` fixpoint
// that evaluated EVERY fixed-phase binding up front into a plain
// `Map<name, value>`. That violates the demand-driven discipline the
// inference side already follows (fixed-eval.ts): it computes values no
// consumer may ever ask for (e.g. `A = load_huge_matrix(); B = f(A)` that
// is never plotted).
//
// `FixedValues` replaces the eager Map with a LAZY, MEMOIZED,
// CYCLE-GUARDED resolver that computes a binding's value only on first
// demand and caches it. The per-binding evaluation logic is the SAME as
// the old loop body (moved here verbatim into `_compute`); only the
// trigger changed from eager-sweep to first-demand, and the value-ref
// gate recurses on demand instead of reading a pre-populated map.
//
// Map-compatible surface. The resolver exposes `.has` / `.get` / `.set`
// plus iteration so the ~100 sites that consume `fixedValues` as a
// `Map` are untouched:
//   - `has(name)`  ≡ "is this resolvable to a concrete fixed value" —
//                    exactly the question every call site asks. It
//                    resolves (and memoises) en route, so a following
//                    `get` is free.
//   - `get(name)`  → the value, or `undefined` when not resolvable
//                    (matching `Map.get` for an absent key).
//   - `set(name,v)`→ writes the cache (kept for parity with the plain
//                    Map some tests inject, and for any warm-cache path).
//
// **iterate-forces-all (load-bearing).** `size` / `forEach` / `keys` /
// `values` / `entries` / `[Symbol.iterator]` FORCE-RESOLVE every
// fixed-phase binding, then delegate to the cache. Iteration means "give
// me ALL fixed values", which cannot be answered lazily. No production
// hot path iterates after the stage-3 re-plumbing (the viewer's bulk
// `setEnv` push and the mat-density full-map copy were the only two, both
// removed); the remaining iterators are test harnesses that mirror the
// old bulk push, and force-resolve-on-iterate keeps them byte-for-byte
// green. Do NOT "optimise" iteration to be cache-only — that silently
// breaks those harnesses and any future bulk consumer.
//
// **No negative cache (load-bearing).** A name that resolves to
// UNRESOLVED is NOT remembered as failed. The old fixpoint let a binding
// that gave up in round N succeed in round N+1 once a sibling resolved;
// a persistent failure set would break that and would permanently
// poison a binding that was only transiently UNRESOLVED because a
// dependency was mid-resolution (a cycle frame). So: cache on success
// only; recompute on each demand for a not-yet-resolved name. Cost is
// re-walking a genuinely-unresolvable binding per demand — bounded, and
// no worse than the old loop's repeated passes.
//
// Reproducibility. Same evaluator (`samplerLib.evaluateExpr`), same
// state-threading resolver (`localResolveValueRef`), same rand seeds,
// raw value cached (never coerced) — byte-identical to the old loop.
//
// Dependency injection. The class takes everything the old loop closed
// over as constructor deps, so this file imports NOTHING from the engine
// (no circular-import risk; trivially unit-testable).

// Module-private sentinel: "this name has no fixed value". A unique
// symbol so it can never collide with a real value (a fixed value may
// legitimately be 0 / null / false / undefined-in-a-record). Never
// exported across the Map surface; never stored in the cache.
const UNRESOLVED: unique symbol = Symbol('flatppl.fixed.unresolved');

interface FixedValuesDeps {
  bindings: any;            // Map<name, BindingInfo>
  derivations: any;         // name → Derivation (the in-progress table)
  resolveMeasureRef: any;   // (name) → measure IR | null
  isMeasureBinding: any;    // (binding) → boolean
  samplerLib: any;          // require('./sampler.ts')
  expandMeasureIR: any;     // (name, derivations) → IR | null
  collectSelfRefs: any;     // (ir) → Set<name>
  lowerExpr: any;           // (ast) → IR
}

class FixedValues {
  _deps: FixedValuesDeps;
  _cache: Map<string, any>;
  _inProgress: Set<string>;

  constructor(deps: FixedValuesDeps) {
    this._deps = deps;
    this._cache = new Map();        // name → resolved value (success only)
    this._inProgress = new Set();   // current resolution stack (cycle guard)
  }

  // ---- Map-compatible public surface -------------------------------

  get(name: string): any {
    const v = this._resolve(name);
    return v === UNRESOLVED ? undefined : v;
  }

  has(name: string): boolean {
    return this._resolve(name) !== UNRESOLVED;
  }

  set(name: string, value: any): this {
    this._cache.set(name, value);
    return this;
  }

  delete(name: string): boolean {
    return this._cache.delete(name);
  }

  // iterate-forces-all (see header) ----------------------------------
  get size(): number { this._forceAll(); return this._cache.size; }
  forEach(cb: any, thisArg?: any): void { this._forceAll(); this._cache.forEach(cb, thisArg); }
  keys(): IterableIterator<string> { this._forceAll(); return this._cache.keys(); }
  values(): IterableIterator<any> { this._forceAll(); return this._cache.values(); }
  entries(): IterableIterator<[string, any]> { this._forceAll(); return this._cache.entries(); }
  [Symbol.iterator](): IterableIterator<[string, any]> {
    this._forceAll();
    return this._cache[Symbol.iterator]();
  }

  // ---- internals ---------------------------------------------------

  // Force-resolve every fixed-phase (or lift-anon, phase==null) binding
  // so iteration can answer "all fixed values". Mirrors the old eager
  // sweep — but only ever reached through an explicit iterate/size call.
  _forceAll(): void {
    const { bindings } = this._deps;
    for (const [name, b] of bindings) {
      if (!b || b.phase == null || b.phase === 'fixed') this._resolve(name);
    }
  }

  // The lazy core. Returns the value or the UNRESOLVED sentinel.
  _resolve(name: string): any {
    if (this._cache.has(name)) return this._cache.get(name);
    const binding = this._deps.bindings.get(name);
    // Phase gate. Only fixed-phase bindings — plus lift-synthesised
    // anonymous bindings (phase == null), which carry lifted *value*
    // expressions — can have a fixed value. A parameterized/stochastic
    // binding never does; reject cheaply so demand can't cascade into
    // the stochastic graph. (Mirrors the old loop's line-308 guard.)
    if (binding && binding.phase != null && binding.phase !== 'fixed') return UNRESOLVED;
    // Cycle gate. Return UNRESOLVED WITHOUT caching — the outer frame
    // owns the verdict (matches resolveConstant's cycle-as-non-constant,
    // and the old fixpoint's "deps not ready this round" retry).
    if (this._inProgress.has(name)) return UNRESOLVED;
    this._inProgress.add(name);
    try {
      const value = this._compute(name, binding);
      if (value === UNRESOLVED) return UNRESOLVED;   // uncached → retried on next demand
      this._cache.set(name, value);
      return value;
    } finally {
      this._inProgress.delete(name);
    }
  }

  // Per-binding evaluation — the old `buildDerivations` loop body
  // (derivations.ts ~lines 310-503) moved here verbatim, with two lazy
  // adaptations marked [LAZY]. Returns the value or UNRESOLVED.
  _compute(name: string, binding: any): any {
    const self = this;
    const {
      bindings, derivations, resolveMeasureRef, isMeasureBinding,
      samplerLib, expandMeasureIR, collectSelfRefs, lowerExpr,
    } = this._deps;
    if (!binding) return UNRESOLVED;

    // Use the post-lift cached IR if present (set by the lift pass);
    // fall back to lowering on demand for bindings the lift never touched.
    const ir = binding.ir
      || (function () {
        try { return lowerExpr(binding.effectiveValue || binding.node.value); }
        catch (_) { return null; }
      })();
    if (!ir) return UNRESOLVED;

    // Collect self-refs into TWO buckets based on where they're reached:
    // "value-context" refs that must resolve before we can evaluate (the
    // classic gate), and "measure-context" refs reached through a measure
    // subtree (e.g. `rand(rstate, ref d)` where d's expansion has
    // `Normal(mu = ref a, sigma = ref b)`). The latter never block — they
    // are resolved lazily at evaluation time via __resolveValueRef, which
    // threads rng state through any stochastic ancestor sampling. This is
    // what makes `data, _ = rand(rstate, lawof(obs))` resolve even when
    // obs's distribution params depend on stochastic ancestors.
    const env: any = { __resolveMeasureRef: resolveMeasureRef };
    const seenMeasure = new Set();
    const valueRefs = new Set<string>();
    const deferredRefs = new Set();

    function collectFor(walkIr: any, inMeasureContext: any, bound?: Set<string>) {
      const refs = collectSelfRefs(walkIr);
      for (const r of refs) {
        // A functionof PARAMETER (a boundary bound per-call by the
        // applying op — `_broadcastApply` binds it into elemEnv) is not a
        // value to resolve globally. Skip it: an implicit-boundary body
        // like `pow(x, 2)` (from `f = functionof(y)`, `y = x^2`) references
        // its boundary `x`, a parameterized `elementof` leaf that would
        // otherwise make the value-ref gate declare the whole binding
        // UNRESOLVED — silently blocking the fixed-phase fold of `f.(X)`.
        if (bound && bound.has(r)) continue;
        if (!bindings.has(r)) continue;
        const dep = bindings.get(r);
        // Function-typed bindings (fn / functionof / kernelof) aren't
        // looked up as values during evaluation — they get resolved via
        // env.__resolveFnBody at the call site. Skip them from the
        // value-ref gate (they never get a value), but recurse into a
        // functionof BODY since it may close over fixed-phase values.
        // The body's own params are bound per-call, so add them to `bound`
        // before recursing (a self-contained implicit-boundary body refers
        // to them directly).
        if (dep && (dep.type === 'fn' || dep.type === 'functionof'
                    || dep.type === 'kernelof')) {
          if (dep.ir && dep.ir.kind === 'call'
              && dep.ir.op === 'functionof' && dep.ir.body) {
            const inner = Array.isArray(dep.ir.params) && dep.ir.params.length
              ? new Set([...(bound || []), ...dep.ir.params]) : bound;
            collectFor(dep.ir.body, inMeasureContext, inner);
          }
          continue;
        }
        // A binding classified as a measure derivation: recurse through
        // the canonical sampleable expansion (what the measure walker
        // sampler.walk walks at runtime) and mark everything beyond it as
        // measure-context.
        if (derivations[r]) {
          if (seenMeasure.has(r)) continue;
          seenMeasure.add(r);
          const expanded = expandMeasureIR(r, derivations);
          if (expanded) { collectFor(expanded, true); continue; }
        }
        if (dep && isMeasureBinding(dep)) {
          // Synthetic anon with measure-construction IR but no derivation
          // entry: recurse through the raw IR hoping params reach value
          // bindings we can resolve.
          if (seenMeasure.has(r)) continue;
          seenMeasure.add(r);
          if (dep.ir) collectFor(dep.ir, true);
          continue;
        }
        if (inMeasureContext) deferredRefs.add(r);
        else                  valueRefs.add(r);
      }
    }
    collectFor(ir, false);

    // [LAZY] value-ref gate. The old loop required each value-context ref
    // to already be in the populated map (`if (!fixedValues.has(r))
    // depsReady=false`), retrying on the next fixpoint round. Here we
    // resolve each on demand: an UNRESOLVED dep means this binding isn't
    // (yet) a fixed value — return UNRESOLVED uncached so a later demand
    // retries (no-negative-cache). The dep-phase fast pre-check is kept
    // to preserve the exact reject ordering.
    for (const r of valueRefs) {
      const dep = bindings.get(r);
      if (dep && dep.phase != null && dep.phase !== 'fixed') return UNRESOLVED;
      const rv = self._resolve(r);
      if (rv === UNRESOLVED) return UNRESOLVED;
      env[r] = rv;
    }

    // State-threading resolver closed over the local env and the binding
    // graph. Called from the measure walker (via evaluateRand →
    // opts.resolveValueRef) whenever a measure-context ref isn't already
    // in env. env is the SAME
    // object the outer evaluateExpr consults, so resolved values cache
    // implicitly — two refs to the same name share one draw.
    function localResolveValueRef(refName: any, state: any): any {
      if (env[refName] !== undefined) return [env[refName], state];
      // [LAZY] cross-binding memo via the resolver itself: a fixed-phase
      // dep is computed once and cached (was: a `fixedValues.has/get`
      // check that only hit if a prior fixpoint round had cached it). A
      // stochastic/cyclic dep returns UNRESOLVED and falls through to the
      // sampling path below, exactly as before.
      const cached = self._resolve(refName);
      if (cached !== UNRESOLVED) {
        env[refName] = cached;
        return [env[refName], state];
      }
      const dep = bindings.get(refName);
      if (!dep) throw new Error(`resolveValueRef: unknown binding '${refName}'`);
      const d = derivations[refName];
      if (d && (d.kind === 'sample' || d.kind === 'alias'
                || d.kind === 'iid' || d.kind === 'record'
                || d.kind === 'weighted')) {
        const measureIR = expandMeasureIR(refName, derivations);
        if (!measureIR) {
          throw new Error(`resolveValueRef: cannot expand measure for '${refName}'`);
        }
        // samplerLib.walk is the in-module measure walker (was
        // traceeval.walk; relocated into sampler.ts in stage 4). It
        // dispatches a bare leaf → walkLeaf → sampleLeafN, iid(<leaf>,n)
        // → walkIid → sampleLeafN, lawof/draw → unwrap, and a genuine
        // composite (joint/record/iid-of-composite/weighted) → its
        // recursive walkers — exactly as before, byte-identical.
        const r = samplerLib.walk(state, measureIR, env, {
          resolveMeasureRef,
          resolveValueRef: localResolveValueRef,
        });
        env[refName] = r.value;
        return [r.value, r.state];
      }
      // Deterministic binding (evaluate-kind, lifted anon, …): walk its
      // own refs through this resolver first, then evaluate.
      const innerIR = (d && d.kind === 'evaluate' && d.ir) || dep.ir;
      if (!innerIR) {
        throw new Error(`resolveValueRef: no IR for '${refName}'`);
      }
      const inner = collectSelfRefs(innerIR);
      for (const n of inner) {
        if (env[n] !== undefined) continue;
        const sub = localResolveValueRef(n, state);
        state = sub[1];
      }
      const v = samplerLib.evaluateExpr(innerIR, env);
      env[refName] = v;
      return [v, state];
    }
    env.__resolveValueRef = localResolveValueRef;

    // filter(pred, data) walks pred's body per element of data. The
    // sampler's evaluateCall uses this hook to resolve a binding name to
    // its (body IR + parameter name) when the binding is a unary
    // fn / functionof / kernelof.
    env.__resolveFnBody = function (bname: any) {
      const fb = bindings.get(bname);
      if (!fb || (fb.type !== 'fn' && fb.type !== 'functionof'
                  && fb.type !== 'kernelof')) {
        return null;
      }
      const fIR = fb.ir;
      if (!fIR || fIR.kind !== 'call' || fIR.op !== 'functionof'
          || !Array.isArray(fIR.params) || !fIR.body) {
        return null;
      }
      return {
        body:        fIR.body,
        params:      fIR.params,
        paramKwargs: fIR.paramKwargs,
        paramName:   fIR.params[0],
      };
    };

    // The synthesised disintegrate effectiveValue can be a measure
    // expression (e.g. `lawof(...)`); evaluating that as a value is a
    // category error. Catch cleanly — UNRESOLVED, not a crash. Same for
    // anonymous bindings whose lifted IR is a measure construction.
    // (Old loop: `catch (_) { continue; }`.)
    try {
      return samplerLib.evaluateExpr(ir, env);
    } catch (_) {
      return UNRESOLVED;
    }
  }
}

module.exports = { FixedValues, UNRESOLVED };
