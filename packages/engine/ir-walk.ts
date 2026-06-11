/**
 * IR walker primitives — single source of truth for how the engine
 * traverses FlatPIR-JSON sub-positions.
 *
 * Background. FlatPIR call nodes carry recursive IR children in
 * several distinct fields, depending on which spec construct they
 * encode (engine-concepts §11, FlatPIR §11):
 *
 *   args         positional IR array              (every call)
 *   callee       IR                               (expression-headed user
 *                                                    call, spec §11 %call —
 *                                                    e.g. an inline
 *                                                    reification applied
 *                                                    directly; ref-headed
 *                                                    calls use `target`, a
 *                                                    {ns,name} string pair,
 *                                                    NOT a child node)
 *   kwargs       object of named IR values        (most calls)
 *   fields       ordered named entries            (record / joint /
 *                  [{name, value: IR}, …]            jointchain / cartprod /
 *                                                    table)
 *   body         IR                               (functionof / kernelof
 *                                                    after lowering)
 *   branches     array of {ref: <name>} | {ir: IR}  (select / superpose /
 *                                                    mixture / ifelse)
 *   selector     IR                               (select)
 *   logweights   array of IR | null               (select)
 *   assigns      array of {name, value: IR}       (load_module)
 *   bijection    pushfwd metadata (READ-side only) — `.fInv.body`,
 *                  `.logVolume.body` (when kind:'fn'), `.paramIRs[*]`
 *                  carry IR. Attached by expandMeasureIR's pushfwd case;
 *                  see the note in forEachIRChild.
 *
 * NOT recursive (carry strings / non-IR metadata):
 *
 *   params, paramKwargs       string arrays
 *   paramSources              [{ kind: 'binding' | 'placeholder', name }]
 *   kind, op, target, ns, name, numType, mode, marginalize,
 *   axisStack, loc, meta       discriminators + diagnostics + dispatch
 *                              metadata
 *
 * Without a centralised primitive, every hand-rolled walker
 * (collectSelfRefs, inlineForProfile, substituteLocals,
 * _substituteAllowingAxes, pir.walkCalls, …) has to re-list these
 * fields and stays in sync by code review. Drift is the dominant
 * bug class — adding `select` to IR was a generator for shape-walker
 * gaps that surfaced months later in completely unrelated viewer code
 * paths. The primitives below collapse that drift surface into one
 * file: future IR-shape extensions update the per-position table
 * here, and every walker picks up the new field automatically.
 *
 * Two flavours of API:
 *
 *   forEachIRChild(node, visit)   — calls visit(child) for each direct
 *                                   IR child; the raw enumeration
 *                                   primitive.
 *   walkIR(node, visit)           — post-order walk of `node` and all
 *                                   descendants; visit(node) fires after
 *                                   its children have been visited. The
 *                                   right tool for accumulators
 *                                   (collectSelfRefs, walkCalls).
 *   mapIR(node, transform)        — functional rebuild: returns a copy
 *                                   of `node` with each IR child replaced
 *                                   by transform(child), then returns
 *                                   transform(rebuilt). Preserves
 *                                   referential identity when nothing
 *                                   changed — important downstream where
 *                                   `===` comparisons gate work
 *                                   (inlineForProfile / substituteLocals).
 *
 * The walkers are pure JS: no engine-internal deps. Every consumer
 * can `require('./ir-walk.ts')` without cycle risk.
 */

/**
 * Enumerate the direct IR children of `node` by calling
 * `visit(child)` on each. Does NOT recurse — that's `walkIR`'s job.
 *
 * Safe to call on any value: non-objects (literals, strings, null,
 * undefined) are no-ops. Skips children that are themselves null /
 * undefined — those occur in the wild (e.g. logweights array entries
 * during partial construction).
 */
function forEachIRChild(node: any, visit: (child: any) => void): void {
  if (!node || typeof node !== 'object') return;

  if (node.callee) visit(node.callee);
  if (node.args) {
    for (let i = 0; i < node.args.length; i++) {
      const a = node.args[i];
      if (a) visit(a);
    }
  }
  if (node.kwargs) {
    for (const k in node.kwargs) {
      const v = node.kwargs[k];
      if (v) visit(v);
    }
  }
  if (node.fields) {
    for (let i = 0; i < node.fields.length; i++) {
      const f = node.fields[i];
      if (f && f.value) visit(f.value);
    }
  }
  if (node.assigns) {
    for (let i = 0; i < node.assigns.length; i++) {
      const a = node.assigns[i];
      if (a && a.value) visit(a.value);
    }
  }
  if (node.body) visit(node.body);
  if (node.branches) {
    for (let i = 0; i < node.branches.length; i++) {
      const b = node.branches[i];
      // `branches` carries TWO shapes in the wild:
      //   - Classifier output: `{ref: <name>}` (named binding,
      //     don't recurse) or `{ir: <IR>}` (inline measure
      //     expression, recurse into `b.ir`).
      //   - Post-`expandMeasureIR` output: bare IR nodes (recognisable
      //     by their `kind` discriminator) — recurse directly.
      if (!b || typeof b !== 'object') continue;
      if (b.ir) visit(b.ir);
      else if (b.kind) visit(b);  // bare-IR branch (post-expand)
    }
  }
  if (node.selector) visit(node.selector);
  if (node.logweights) {
    for (let i = 0; i < node.logweights.length; i++) {
      const w = node.logweights[i];
      if (w) visit(w);
    }
  }
  // `.bijection` (pushfwd metadata attached by expandMeasureIR's pushfwd
  // case, see resolveBijectionMeta) carries IR in three sub-positions:
  // the inverse-function body, the optional log-volume function body, and
  // the registry `paramIRs`. These hold captured `ns:'self'` refs to free
  // boundaries (e.g. an affine shift `b` reading a prior field), so a
  // self-ref collector MUST descend them or the CLM ⊆-invariant
  // undercounts and passes vacuously (measure-lowering-unification-plan
  // critique D / audit H4). Formal-parameter refs inside the bodies are
  // `ns:'%local'`, so a `ns:'self'`-only collector naturally excludes the
  // bound variable; `paramIRs` was already hand-descended in
  // materialiser-shared.prepareDensityRefs, so surfacing it here is
  // idempotent for that consumer.
  //
  // READ-SIDE ONLY. `mapIR` deliberately does NOT mirror this descent:
  // rewriting bijection bodies (substituteLocals / computeClosureIR) is a
  // semantic change deferred to the CLM viewer phase so it lands under its
  // own tests rather than silently flipping every existing rewrite. The
  // resulting asymmetry is safe — walkIR over-reporting refs vs mapIR
  // under-rewriting them is the conservative direction for the ⊆ check.
  if (node.bijection) {
    const bj = node.bijection;
    if (bj.fInv && bj.fInv.body) visit(bj.fInv.body);
    if (bj.logVolume && bj.logVolume.kind === 'fn' && bj.logVolume.body) {
      visit(bj.logVolume.body);
    }
    if (bj.paramIRs) {
      for (const k in bj.paramIRs) {
        const p = bj.paramIRs[k];
        if (p) visit(p);
      }
    }
  }
}

/**
 * Post-order walk: visit every IR descendant of `node`, then visit
 * `node` itself. `visit` is called once per IR object encountered.
 *
 * Cycles are not protected against — FlatPIR is a tree, not a graph
 * (refs are by-name strings, not by-pointer). Callers that traverse
 * via refs need their own visited-set.
 */
function walkIR(node: any, visit: (n: any) => void): void {
  if (!node || typeof node !== 'object') return;
  forEachIRChild(node, child => walkIR(child, visit));
  visit(node);
}

/**
 * Functional rebuild: returns `transform(node')` where `node'` is a
 * shallow copy of `node` with each IR child replaced by `mapIR(child,
 * transform)`.
 *
 * Identity-preserving: when transform returns the input unchanged for
 * every descendant, the original `node` is passed to the outer
 * transform — no shallow copy allocated. This matters for downstream
 * `===` comparisons (e.g. cache lookups keyed by IR reference).
 *
 * Non-IR fields (loc / meta / kind / op / name / numType / …) are
 * carried through the shallow copy untouched. Transformers that need
 * to inspect or rewrite those fields can do so on the result of
 * `mapIR`.
 */
function mapIR<T>(node: any, transform: (n: any) => any): any {
  if (!node || typeof node !== 'object') return transform(node);
  return transform(_mapChildren(node, (c: any) => mapIR(c, transform)));
}

// Shared child-rebuild for mapIR / mapIRScoped: returns a shallow copy
// of `node` with each IR child replaced by `mapChild(child, isBody)`,
// or `node` itself when nothing changed (identity preservation).
// `isBody` flags the `body` sub-position so scope-aware callers can
// extend the shadow set for it (see mapIRScoped).
function _mapChildren(node: any, mapChild: (c: any, isBody: boolean) => any): any {
  let copy: any = null;
  function ensureCopy(): any {
    if (copy) return copy;
    copy = {};
    for (const k in node) copy[k] = node[k];
    return copy;
  }

  if (node.callee) {
    const c2 = mapChild(node.callee, false);
    if (c2 !== node.callee) ensureCopy().callee = c2;
  }
  if (node.args) {
    let changed = false;
    const a2 = new Array(node.args.length);
    for (let i = 0; i < node.args.length; i++) {
      a2[i] = mapChild(node.args[i], false);
      if (a2[i] !== node.args[i]) changed = true;
    }
    if (changed) ensureCopy().args = a2;
  }
  if (node.kwargs) {
    let changed = false;
    const k2: any = {};
    for (const k in node.kwargs) {
      k2[k] = mapChild(node.kwargs[k], false);
      if (k2[k] !== node.kwargs[k]) changed = true;
    }
    if (changed) ensureCopy().kwargs = k2;
  }
  if (node.fields) {
    let changed = false;
    const f2 = new Array(node.fields.length);
    for (let i = 0; i < node.fields.length; i++) {
      const f = node.fields[i];
      if (f && f.value) {
        const nv = mapChild(f.value, false);
        if (nv !== f.value) {
          f2[i] = { ...f, value: nv };
          changed = true;
        } else {
          f2[i] = f;
        }
      } else {
        f2[i] = f;
      }
    }
    if (changed) ensureCopy().fields = f2;
  }
  if (node.assigns) {
    let changed = false;
    const a2 = new Array(node.assigns.length);
    for (let i = 0; i < node.assigns.length; i++) {
      const a = node.assigns[i];
      if (a && a.value) {
        const nv = mapChild(a.value, false);
        if (nv !== a.value) {
          a2[i] = { ...a, value: nv };
          changed = true;
        } else {
          a2[i] = a;
        }
      } else {
        a2[i] = a;
      }
    }
    if (changed) ensureCopy().assigns = a2;
  }
  if (node.body) {
    const b2 = mapChild(node.body, true);
    if (b2 !== node.body) ensureCopy().body = b2;
  }
  if (node.branches) {
    let changed = false;
    const br2 = new Array(node.branches.length);
    for (let i = 0; i < node.branches.length; i++) {
      const b = node.branches[i];
      if (b && b.ir) {
        // Classifier wrapper {ir: <IR>}: rewrite b.ir, preserve wrapper.
        const ni = mapChild(b.ir, false);
        if (ni !== b.ir) {
          br2[i] = { ...b, ir: ni };
          changed = true;
        } else {
          br2[i] = b;
        }
      } else if (b && b.kind) {
        // Post-expand bare IR: recurse directly.
        const ni = mapChild(b, false);
        if (ni !== b) changed = true;
        br2[i] = ni;
      } else {
        br2[i] = b;
      }
    }
    if (changed) ensureCopy().branches = br2;
  }
  if (node.selector) {
    const s2 = mapChild(node.selector, false);
    if (s2 !== node.selector) ensureCopy().selector = s2;
  }
  if (node.logweights) {
    let changed = false;
    const lw2 = new Array(node.logweights.length);
    for (let i = 0; i < node.logweights.length; i++) {
      const w = node.logweights[i];
      lw2[i] = w ? mapChild(w, false) : w;
      if (lw2[i] !== w) changed = true;
    }
    if (changed) ensureCopy().logweights = lw2;
  }

  return copy || node;
}

/**
 * Scope-aware functional rebuild: like `mapIR`, but tracks the active
 * identifier-bound boundary params (see `identifierBoundParams`) and
 * passes the shadow set to `transform(n, shadowed)`. Descending into a
 * reified callable's `body` extends the set with that node's
 * identifier-bound params (nesting-aware). The scoped sibling of
 * `walkIRScoped`, for REWRITING passes that must not touch body refs
 * designating boundary inputs (e.g. alias resolution — rewriting one
 * through an alias chain would desync the body from `params`).
 */
function mapIRScoped(
  node: any, transform: (n: any, shadowed: Set<string>) => any,
  shadowed?: Set<string>,
): any {
  const active = shadowed || new Set<string>();
  if (!node || typeof node !== 'object') return transform(node, active);
  const bound = identifierBoundParams(node);
  let inner = active;
  if (bound && node.body) {
    inner = new Set(active);
    for (const p of bound) inner.add(p);
  }
  const rebuilt = _mapChildren(node, (c: any, isBody: boolean) =>
    mapIRScoped(c, transform, isBody ? inner : active));
  return transform(rebuilt, active);
}

/**
 * The IDENTIFIER-bound boundary-input names of a reified callable node
 * (spec §11 / engine-concepts §8): for `functionof(e, p = a)` the cut
 * node `a`; for the placeholder form (`x = _x_`) nothing — placeholders
 * live in `%local` and never collide with module names. Body refs to
 * these names designate the callable's INPUTS (decoupled from the
 * like-named module nodes per spec §04 boundary substitution), so
 * scope-aware walkers shadow them when descending `body`.
 *
 * Reads `paramSources` (kind 'binding' vs 'placeholder'); a node
 * without it (hand-built synthetic functionofs are placeholder-only)
 * contributes nothing. Returns null when there is nothing to shadow —
 * callers can skip allocating an extended shadow set.
 */
function identifierBoundParams(node: any): string[] | null {
  if (!node || node.kind !== 'call'
      || (node.op !== 'functionof' && node.op !== 'kernelof')
      || !Array.isArray(node.params) || !Array.isArray(node.paramSources)) {
    return null;
  }
  let out: string[] | null = null;
  for (let i = 0; i < node.paramSources.length; i++) {
    const s = node.paramSources[i];
    if (s && s.kind === 'binding' && node.params[i] != null) {
      (out || (out = [])).push(node.params[i]);
    }
  }
  return out;
}

/**
 * Scope-aware post-order walk: like `walkIR`, but tracks the active
 * identifier-bound boundary params (see `identifierBoundParams`) and
 * passes the shadow set to `visit(n, shadowed)`. Descending into a
 * reified callable's `body` extends the set with that node's
 * identifier-bound params (nesting-aware — an inner reification that
 * re-declares an outer name shadows it for its own body).
 *
 * Use this instead of `walkIR` whenever the visitor must distinguish a
 * body ref that designates a BOUNDARY INPUT (fed at application; not an
 * outer-scope dependency) from a genuine outer-scope ref — the
 * spec-shaped IR keeps both in the `self` namespace (spec §11; only
 * placeholders are `%local`), so the distinction is scoping, not
 * namespace.
 */
function walkIRScoped(
  node: any, visit: (n: any, shadowed: Set<string>) => void,
  shadowed?: Set<string>,
): void {
  const active = shadowed || new Set<string>();
  if (!node || typeof node !== 'object') return;
  const bound = identifierBoundParams(node);
  if (bound && node.body) {
    // Children other than `body` (a reified node carries none today,
    // but stay future-proof) walk under the OUTER scope; `body` under
    // the extended scope.
    const inner = new Set(active);
    for (const p of bound) inner.add(p);
    forEachIRChild(node, (child) => {
      if (child === node.body) walkIRScoped(child, visit, inner);
      else walkIRScoped(child, visit, active);
    });
  } else {
    forEachIRChild(node, (child) => walkIRScoped(child, visit, active));
  }
  visit(node, active);
}

module.exports = {
  forEachIRChild,
  walkIR,
  walkIRScoped,
  identifierBoundParams,
  mapIR,
  mapIRScoped,
};
