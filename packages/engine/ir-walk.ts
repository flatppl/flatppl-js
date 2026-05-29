/**
 * IR walker primitives — single source of truth for how the engine
 * traverses FlatPIR-JSON sub-positions.
 *
 * Background. FlatPIR call nodes carry recursive IR children in
 * several distinct fields, depending on which spec construct they
 * encode (engine-concepts §11, FlatPIR §11):
 *
 *   args         positional IR array              (every call)
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
 *
 * NOT recursive (carry strings / non-IR metadata):
 *
 *   params, paramKwargs       string arrays
 *   paramSources              [{ kind: 'binding' | 'placeholder', name }]
 *   kind, op, target, ns, name, numType, mode, marginalize, weightsFrom,
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

  let copy: any = null;
  function ensureCopy(): any {
    if (copy) return copy;
    copy = {};
    for (const k in node) copy[k] = node[k];
    return copy;
  }

  if (node.args) {
    let changed = false;
    const a2 = new Array(node.args.length);
    for (let i = 0; i < node.args.length; i++) {
      a2[i] = mapIR(node.args[i], transform);
      if (a2[i] !== node.args[i]) changed = true;
    }
    if (changed) ensureCopy().args = a2;
  }
  if (node.kwargs) {
    let changed = false;
    const k2: any = {};
    for (const k in node.kwargs) {
      k2[k] = mapIR(node.kwargs[k], transform);
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
        const nv = mapIR(f.value, transform);
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
        const nv = mapIR(a.value, transform);
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
    const b2 = mapIR(node.body, transform);
    if (b2 !== node.body) ensureCopy().body = b2;
  }
  if (node.branches) {
    let changed = false;
    const br2 = new Array(node.branches.length);
    for (let i = 0; i < node.branches.length; i++) {
      const b = node.branches[i];
      if (b && b.ir) {
        // Classifier wrapper {ir: <IR>}: rewrite b.ir, preserve wrapper.
        const ni = mapIR(b.ir, transform);
        if (ni !== b.ir) {
          br2[i] = { ...b, ir: ni };
          changed = true;
        } else {
          br2[i] = b;
        }
      } else if (b && b.kind) {
        // Post-expand bare IR: recurse directly.
        const ni = mapIR(b, transform);
        if (ni !== b) changed = true;
        br2[i] = ni;
      } else {
        br2[i] = b;
      }
    }
    if (changed) ensureCopy().branches = br2;
  }
  if (node.selector) {
    const s2 = mapIR(node.selector, transform);
    if (s2 !== node.selector) ensureCopy().selector = s2;
  }
  if (node.logweights) {
    let changed = false;
    const lw2 = new Array(node.logweights.length);
    for (let i = 0; i < node.logweights.length; i++) {
      const w = node.logweights[i];
      lw2[i] = w ? mapIR(w, transform) : w;
      if (lw2[i] !== w) changed = true;
    }
    if (changed) ensureCopy().logweights = lw2;
  }

  return transform(copy || node);
}

module.exports = {
  forEachIRChild,
  walkIR,
  mapIR,
};
