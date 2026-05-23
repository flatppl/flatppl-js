'use strict';

// Structural disintegration for FlatPPL.
//
// `disintegratePlan(joint, selector, bindings)` walks the joint expression
// and returns a `Plan` describing how to factor it into a kernel and a
// prior. The contract — straight from the FlatPPL spec §06 (Cho-Jacobs
// also; see §13.X / 15-references) — is
//
//     jointchain(prior, kernel)  ≡  joint                              (1)
//
// Plans come in three shapes:
//
//   { kind: 'delegate',    kernel: { binding }, prior: { binding } }
//        — both sides reuse existing bindings. The renderer can re-render
//          those bindings' sub-DAGs with the disintegration result names
//          as targets. Cleanest output when the joint is already factored
//          along binding boundaries.
//
//   { kind: 'synthesized', kernel: ASTNode,    prior: ASTNode }
//        — freshly-constructed AST for both sides. Rendered like any
//          binding's RHS. Used when the joint factors structurally but no
//          existing binding holds either side directly.
//
//   { kind: 'unsupported', reason, blockingNode, detail? }
//        — refused. The caller should fall back to a dependency-trace view
//          and surface the reason to the user.
//
// All AST nodes constructed by this module carry `synthLoc(source)` rather
// than parser-provided locations so consumers can tell rewriter output
// apart from user code.
//
// ===================================================================
// Architecture: decompose → partition → admissibility → synthesize
// ===================================================================
//
// Spec §06 phrases the algorithm as "straightforward graph inspection on
// the joint's factorization structure." We mirror that literally:
//
//   1. `decompose(node)` extracts the joint's **variate decomposition**:
//      a map from variate name to the sub-expression that produces it,
//      plus per-variate dependencies (which other variates' values it
//      reads). One small case per joint shape (lawof-record, joint-kw,
//      joint-positional, jointchain-kw, jointchain-positional, relabel
//      of those), each ~15 lines. Compositions chain: `relabel` recurses
//      into `decompose` on its inner.
//
//   2. `partitionVariates(decomp, selector)` splits the variates into
//      Selected (kernel side) and Unselected (marginal side) — uniform
//      across all decomposition kinds.
//
//   3. `admissibilityCheck(decomp, partition)` enforces the spec
//      invariant: no Unselected variate depends on a Selected variate
//      (otherwise the marginal can't be constructed without integration).
//      Uniform.
//
//   4. `tryDelegate(decomp, partition)` recognises the case where both
//      sides are exactly single binding refs already present in the
//      module, and emits a Delegate plan instead of synthesizing fresh
//      AST. Uniform.
//
//   5. `synthesizePlan(decomp, partition, ctx)` emits the kernel/marginal
//      AST. The emit shape depends on the decomposition kind (the spec
//      identity `lawof(record(n=v)) ≡ joint(n=lawof(v))` means we have a
//      choice; we emit the form that round-trips cleanly to surface
//      source). One small case per kind, ~10-15 lines each.
//
// See `references.md` (Cho & Jacobs 2017/2019; Shan & Ramsey 2017; Pearl
// 1988) for the algebra. The eval-with-rest pattern (§4 of engine-
// concepts) is the density-side dual of this decomposition.

const ast = require('./ast.ts');
const { collectDeps, isMeasureExpr } = require('./analyzer.ts');

// ---------------------------------------------------------------------
// AST construction helpers
// ---------------------------------------------------------------------

function mkIdent(name: any, source: any)        { return ast.Identifier(name, ast.synthLoc(source)); }
function mkKwArg(name: any, value: any, source: any) { return ast.KeywordArg(name, value, ast.synthLoc(source)); }
function mkCall(name: any, args: any, source: any)   {
  return ast.CallExpr(mkIdent(name, source), args, ast.synthLoc(source));
}

// ---------------------------------------------------------------------
// Plan constructors
// ---------------------------------------------------------------------

function delegate(kernelBinding: any, priorBinding: any) {
  return {
    kind: 'delegate',
    kernel: { binding: kernelBinding },
    prior:  { binding: priorBinding },
  };
}

function synthesized(kernelExpr: any, priorExpr: any) {
  return { kind: 'synthesized', kernel: kernelExpr, prior: priorExpr };
}

function unsupported(reason: any, blockingNode: any, detail?: any): { kind: 'unsupported', reason: any, blockingNode: any, detail: any } {
  return { kind: 'unsupported' as const, reason, blockingNode, detail: detail || null };
}

// ---------------------------------------------------------------------
// Selector / field utilities
// ---------------------------------------------------------------------

// Try to read the field map from `record(name = expr, ...)`.
// Returns Map<name, expr> or null if the call isn't a static record.
function tryRecord(node: any) {
  if (!node || node.type !== 'CallExpr') return null;
  if (!node.callee || node.callee.type !== 'Identifier' || node.callee.name !== 'record') return null;
  const fields = new Map();
  for (const arg of node.args) {
    if (arg.type !== 'KeywordArg') return null;
    fields.set(arg.name, arg.value);
  }
  return fields;
}

// Best-effort: introspect `node`'s named output fields. Returns an array of
// field names, or [] if the structure isn't statically resolvable. Used by
// jointchain-positional decomposition and by external callers (renderer).
function namedOutputFields(node: any, bindings: any, seen?: any) {
  if (!node) return [];
  if (!seen) seen = new Set();
  if (node.type === 'Identifier') {
    if (seen.has(node.name)) return [];
    const inner = new Set(seen); inner.add(node.name);
    const b = bindings.get(node.name);
    if (!b || !b.node || !b.node.value) return [];
    return namedOutputFields(b.node.value, bindings, inner);
  }
  if (node.type !== 'CallExpr' || !node.callee || node.callee.type !== 'Identifier') {
    return [];
  }
  const callee = node.callee.name;
  if (callee === 'lawof' || callee === 'kernelof' || callee === 'functionof') {
    const firstArg = node.args.find((a: any) => a.type !== 'KeywordArg');
    if (!firstArg) return [];
    const rec = tryRecord(firstArg);
    if (rec) return [...rec.keys()];
    return namedOutputFields(firstArg, bindings, seen);
  }
  if (callee === 'joint' || callee === 'jointchain') {
    if (node.args.length > 0 && node.args.every((a: any) => a.type === 'KeywordArg')) {
      return node.args.map((a: any) => a.name);
    }
    return [];
  }
  if (callee === 'relabel') {
    if (node.args.length >= 2 && node.args[1].type === 'ArrayLiteral') {
      const out: any[] = [];
      for (const el of node.args[1].elements) {
        if (el.type === 'StringLiteral') out.push(el.value);
      }
      return out;
    }
    return [];
  }
  return [];
}

// Walk `node`'s value-graph ancestors (via b.deps), returning the set of
// binding names reachable. Stops at unbound names. Used by the lawof-record
// decomposition's per-variate dependency computation.
function collectAncestorNames(node: any, bindings: any) {
  const result = new Set<string>();
  const definedNames = new Set(bindings.keys());
  function visit(name: any) {
    if (result.has(name)) return;
    result.add(name);
    const b = bindings.get(name);
    if (!b || !b.deps) return;
    for (const d of b.deps) visit(d);
  }
  const { deps } = collectDeps(node, definedNames);
  for (const d of deps) visit(d);
  return result;
}

// ---------------------------------------------------------------------
// Decomposition: { kind, variates: ordered map, allFields }
// ---------------------------------------------------------------------
//
// Each variate carries:
//   - `name`: the field name in the joint's output.
//   - `expr`: the AST node that produces this variate. For lawof-record
//     it's `lawof(<original variate-ref>)`; for joint-keyword it's the
//     component measure M_i; for jointchain it's the per-step kernel/
//     measure. Used by `synthesizePlan` to build the marginal and
//     kernel sides.
//   - `bindingRef`: when the variate's value is a bare Identifier
//     referencing an outer-scope binding, the binding's name. Drives
//     the Delegate optimization.
//   - `deps`: the set of OTHER variate names whose value this variate
//     reads. Independent variates (joint shape) have empty deps;
//     chain variates have a prefix of the chain; lawof-record
//     variates have whichever fields are in their value-graph
//     ancestor closure.
//
// `kind` selects the synthesize-shape ('lawof-record' emits
// `lawof(record(...))`; 'joint' emits `joint(...)`; 'jointchain' emits
// `jointchain(...)`). Per spec identities they would all be
// interconvertible, but tests and the renderer expect a stable shape
// per input form, so we preserve it.

type VariateInfo = {
  name: string,
  expr: any,
  bindingRef?: string,
  deps: Set<string>,
  // Source-component index. Variates that came from the same source
  // expression (e.g. all fields from `prior = lawof(record(...))` when
  // prior is one positional component of a jointchain) share the same
  // index and must be partitioned together — the selector can't take a
  // subset of one component without structurally violating the
  // decomposition. For shapes where one variate ≡ one component
  // (lawof-record, joint-kw, jointchain-kw), each variate has its own
  // unique componentIdx and the constraint is vacuous.
  componentIdx: number,
};

type ComponentInfo = {
  expr: any,         // The component's AST node (the source expression).
  fields: string[],  // The variate field names this component contributes.
  bindingRef?: string,
};

type Decomposition = {
  kind: 'lawof-record' | 'joint' | 'jointchain',
  variates: VariateInfo[],   // preserves source order
  allFields: Set<string>,
  // Component-level view of the same decomposition. For jointchain with
  // multi-field positional components this differs from the variate
  // view (one component → multiple variates). For other shapes
  // components ≡ variates (one per slot). Synthesize uses this view
  // when rebuilding kernel/prior so positional-jointchain components
  // stay together.
  components: ComponentInfo[],
  // For lawof-record, the original variate-binding-name keyed by field
  // name. Used by synthesize to rebuild lawof(record(field=variate)).
  recordVar?: Map<string, string>,
  // For jointchain, was the source positional (vs keyword)? Synthesize
  // uses this to preserve the surface shape when emitting.
  positional?: boolean,
};

type DecomposeResult = Decomposition | { kind: 'unsupported', reason: string, blockingNode: any, detail?: string };

function isUnsupported(d: any): d is { kind: 'unsupported', reason: string, blockingNode: any, detail?: string } {
  return d && d.kind === 'unsupported';
}

// ---------------------------------------------------------------------
// decompose: one entry per joint shape. Each case is small and isolated.
// ---------------------------------------------------------------------

function decompose(node: any, bindings: any, ctx: any): DecomposeResult {
  if (!node || node.type !== 'CallExpr' || !node.callee
      || node.callee.type !== 'Identifier') {
    return unsupported('joint expression is not a measure-algebra call', node);
  }
  const callee = node.callee.name;
  switch (callee) {
    case 'lawof':      return decomposeLawof(node, bindings, ctx);
    case 'joint':      return decomposeJoint(node, bindings, ctx);
    case 'jointchain': return decomposeJointchain(node, bindings, ctx);
    case 'relabel':    return decomposeRelabel(node, bindings, ctx);

    // Explicit refusals — spec §06 says these may be intractable and the
    // engine isn't required to support them. Each carries a clear reason
    // so downstream code can surface it.
    case 'kchain':
      return unsupported(
        "'kchain' has no structural disintegration rule", node,
        'kchain(M, K1, ..., Kn) marginalizes out intermediate variates; recovering them from the result requires integration');
    case 'pushfwd':
      return unsupported(
        "'pushfwd' has no v1+v2 structural disintegration rule", node,
        'projection (subset selection) and bijection (block-separable transform) cases require post-transform Plan annotations');
    case 'weighted': case 'logweighted': case 'bayesupdate':
      return unsupported(`'${callee}' has no v1+v2 disintegration rule`, node,
        'free-variable scope rule deferred to v3');
    case 'truncate':
      return unsupported("'truncate' has no v1+v2 disintegration rule", node,
        'free-variable scope rule deferred to v3');
    case 'iid':
      return unsupported("'iid' has no v1+v2 disintegration rule", node,
        'index-based selectors deferred to v3');
    case 'superpose': case 'normalize':
      return unsupported(`'${callee}' has no structural disintegration rule`, node);

    // Reification operators are kernels/functions, not joint measures.
    case 'kernelof':
    case 'functionof':
      return unsupported(
        `${callee}() is a kernel/function; disintegrate operates on joint measures`, node);

    default:
      return unsupported(`'${callee}' has no structural disintegration rule`, node);
  }
}

// `lawof(record(n1 = v1, n2 = v2, ...))` — the slice-and-rebuild shape.
// Each entry's value is a stochastic value node (Identifier per the
// FlatPPL joint-measure pattern). Per-variate dependency comes from the
// value-graph ancestor walk: n_i depends on n_j iff v_j is in v_i's
// transitive value-graph ancestors.
function decomposeLawof(node: any, bindings: any, ctx: any): DecomposeResult {
  if (node.args.length !== 1) {
    return unsupported('lawof must be unary', node);
  }
  const rec = tryRecord(node.args[0]);
  if (!rec) {
    return unsupported('lawof of non-record cannot be structurally disintegrated', node.args[0]);
  }
  for (const [fname, val] of rec) {
    if (val.type !== 'Identifier') {
      return unsupported(
        `record field '${fname}' is not a bare identifier — structural disintegration requires named stochastic nodes`,
        val);
    }
  }
  const recordVar = new Map<string, string>();
  for (const [fname, val] of rec) recordVar.set(fname, val.name);

  // Value-graph ancestors for each variate-binding, used to compute
  // per-field dependencies. One variate per record field; each variate
  // is its own component (record-field independence at the slot level —
  // the inter-field dependency is encoded via `deps`, not via component
  // grouping).
  const variates: VariateInfo[] = [];
  const components: ComponentInfo[] = [];
  let idx = 0;
  for (const [fname, val] of rec) {
    const ancestors = collectAncestorNames(val, bindings);
    ancestors.delete(val.name);
    const deps = new Set<string>();
    for (const otherField of recordVar.keys()) {
      if (otherField === fname) continue;
      const otherVar = recordVar.get(otherField)!;
      if (ancestors.has(otherVar)) deps.add(otherField);
    }
    variates.push({
      name: fname,
      expr: mkCall('lawof', [mkIdent(val.name, ctx.source)], ctx.source),
      bindingRef: val.name,
      deps,
      componentIdx: idx,
    });
    components.push({
      expr: val,
      fields: [fname],
      bindingRef: val.name,
    });
    idx++;
  }
  return {
    kind: 'lawof-record',
    variates,
    components,
    allFields: new Set(recordVar.keys()),
    recordVar,
  };
}

// `joint(n1 = M1, n2 = M2, ...)` keyword form — independent product.
// All variates have empty deps. (Positional form falls through to
// Unsupported per the existing rule; the renderer's `relabel(positional
// joint, names)` lift handles the renamed-positional case via
// `decomposeRelabel`.)
function decomposeJoint(node: any, bindings: any, ctx: any): DecomposeResult {
  if (node.args.length === 0) {
    return unsupported('empty joint()', node);
  }
  if (!node.args.every((a: any) => a.type === 'KeywordArg')) {
    return unsupported(
      'positional joint(M1, M2, ...) has no structural disintegration rule (use joint(name=M, ...) for selector matching)',
      node);
  }
  const variates: VariateInfo[] = node.args.map((a: any, i: number) => ({
    name: a.name,
    expr: a.value,
    bindingRef: a.value && a.value.type === 'Identifier' ? a.value.name : undefined,
    deps: new Set<string>(),
    componentIdx: i,
  }));
  const components: ComponentInfo[] = node.args.map((a: any) => ({
    expr: a.value,
    fields: [a.name],
    bindingRef: a.value && a.value.type === 'Identifier' ? a.value.name : undefined,
  }));
  void bindings; void ctx;
  return {
    kind: 'joint',
    variates,
    components,
    allFields: new Set(variates.map(v => v.name)),
  };
}

// `jointchain(...)` — chain dependency: variate i depends on variates
// 1..i-1. Keyword and positional both supported. Positional uses
// `namedOutputFields` to extract a per-component field name; if any
// component lacks a statically-resolvable field, the decomposition
// fails (Unsupported).
function decomposeJointchain(node: any, bindings: any, ctx: any): DecomposeResult {
  if (node.args.length === 0) {
    return unsupported('empty jointchain()', node);
  }
  const allKeyword = node.args.every((a: any) => a.type === 'KeywordArg');
  const allPositional = node.args.every((a: any) => a.type !== 'KeywordArg');
  if (!allKeyword && !allPositional) {
    return unsupported('jointchain mixing keyword and positional args', node);
  }
  const variates: VariateInfo[] = [];
  const components: ComponentInfo[] = [];
  const priorNames: string[] = [];
  if (allKeyword) {
    // One variate per kwarg slot, one component per kwarg slot.
    for (let i = 0; i < node.args.length; i++) {
      const a = node.args[i];
      const deps = new Set(priorNames);
      variates.push({
        name: a.name,
        expr: a.value,
        bindingRef: a.value && a.value.type === 'Identifier' ? a.value.name : undefined,
        deps,
        componentIdx: i,
      });
      components.push({
        expr: a.value,
        fields: [a.name],
        bindingRef: a.value && a.value.type === 'Identifier' ? a.value.name : undefined,
      });
      priorNames.push(a.name);
    }
  } else {
    // Positional. Each component contributes one OR MORE variates via
    // its named output fields. Variates within the same component share
    // the same dependency set (all previous components' variates) and
    // the same componentIdx — they must be partitioned together
    // (selector can't take a subset of one component).
    for (let i = 0; i < node.args.length; i++) {
      const a = node.args[i];
      const fields = namedOutputFields(a, bindings);
      if (fields.length === 0) {
        return unsupported(
          'positional jointchain component has no statically-resolvable named output fields',
          a);
      }
      const depsBefore = new Set(priorNames);
      for (const name of fields) {
        variates.push({
          name,
          expr: a,
          bindingRef: a.type === 'Identifier' ? a.name : undefined,
          deps: new Set(depsBefore),
          componentIdx: i,
        });
        priorNames.push(name);
      }
      components.push({
        expr: a,
        fields,
        bindingRef: a.type === 'Identifier' ? a.name : undefined,
      });
    }
  }
  void ctx;
  return {
    kind: 'jointchain',
    variates,
    components,
    allFields: new Set(variates.map(v => v.name)),
    positional: !allKeyword,
  };
}

// `relabel(M, [names])` — pure rename on the variate space. Lift to the
// inner decomposition and apply the rename. Two patterns:
//   - relabel(positional joint/jointchain, [names]): names map onto the
//     positional components in order, lifting to keyword form before
//     decomposing.
//   - relabel(inner-with-named-fields, [names]): names replace the
//     inner's existing field names, positionally.
function decomposeRelabel(node: any, bindings: any, ctx: any): DecomposeResult {
  if (node.args.length !== 2) {
    return unsupported('relabel must take (measure, [names])', node);
  }
  const inner = node.args[0];
  const namesArg = node.args[1];
  if (namesArg.type !== 'ArrayLiteral') {
    return unsupported('relabel names argument must be an array literal', namesArg);
  }
  const names: string[] = [];
  for (const el of namesArg.elements) {
    if (el.type !== 'StringLiteral') {
      return unsupported('relabel names must be string literals', el);
    }
    names.push(el.value);
  }
  // Lift positional joint/jointchain to keyword form.
  if (inner && inner.type === 'CallExpr'
      && inner.callee && inner.callee.type === 'Identifier'
      && (inner.callee.name === 'joint' || inner.callee.name === 'jointchain')
      && inner.args.length === names.length
      && inner.args.every((a: any) => a.type !== 'KeywordArg')) {
    const lifted = ast.CallExpr(
      inner.callee,
      inner.args.map((a: any, i: any) => mkKwArg(names[i], a, ctx.source)),
      ast.synthLoc(ctx.source),
    );
    return decompose(lifted, bindings, ctx);
  }
  // Other inner shapes: try to decompose, then positionally rename the
  // resulting variates.
  const innerDecomp = decompose(inner, bindings, ctx);
  if (isUnsupported(innerDecomp)) return innerDecomp;
  if (innerDecomp.variates.length !== names.length) {
    return unsupported(
      'relabel name count does not match inner decomposition field count',
      namesArg);
  }
  const renamed: VariateInfo[] = innerDecomp.variates.map((v, i) => ({
    name: names[i],
    expr: v.expr,
    bindingRef: v.bindingRef,
    deps: new Set([...v.deps].map(d => {
      const idx = innerDecomp.variates.findIndex(w => w.name === d);
      return names[idx];
    })),
    componentIdx: v.componentIdx,
  }));
  // Rename components' field lists positionally to match.
  const renamedComponents: ComponentInfo[] = innerDecomp.components.map(c => ({
    expr: c.expr,
    fields: c.fields.map(fname => {
      const idx = innerDecomp.variates.findIndex(w => w.name === fname);
      return names[idx];
    }),
    bindingRef: c.bindingRef,
  }));
  // Keep the inner kind so synthesize emits the same shape.
  return {
    kind: innerDecomp.kind,
    variates: renamed,
    components: renamedComponents,
    allFields: new Set(names),
    recordVar: innerDecomp.recordVar
      ? new Map([...innerDecomp.recordVar].map(([oldName, varName]) => {
          const idx = innerDecomp.variates.findIndex(w => w.name === oldName);
          return [names[idx], varName] as [string, string];
        }))
      : undefined,
    positional: innerDecomp.positional,
  };
}

// ---------------------------------------------------------------------
// Partition + admissibility (uniform across decomposition kinds)
// ---------------------------------------------------------------------

type Partition = {
  selected: VariateInfo[],   // preserves source order
  unselected: VariateInfo[], // preserves source order
};

type PartitionResult = Partition | { kind: 'unsupported', reason: string, blockingNode: any, detail?: string };

function partitionVariates(decomp: Decomposition, selector: string[], node: any): PartitionResult {
  for (const s of selector) {
    if (!decomp.allFields.has(s)) {
      return unsupported(`selector '${s}' is not a field of the joint`, node);
    }
  }
  const selected   = decomp.variates.filter(v => selector.includes(v.name));
  const unselected = decomp.variates.filter(v => !selector.includes(v.name));
  if (selected.length === 0) {
    return unsupported('selector matches no joint fields', node);
  }
  if (unselected.length === 0) {
    return unsupported('selector covers all fields; nothing left for the prior', node);
  }
  // Component-boundary alignment: if a component contributes multiple
  // variates (only happens for jointchain-positional with multi-field
  // components), the selector must take EITHER all variates of the
  // component OR none. A partial pick would mean splitting the
  // component, which structurally violates the decomposition.
  for (const c of decomp.components) {
    if (c.fields.length <= 1) continue;
    const inSelector = c.fields.filter(f => selector.includes(f)).length;
    if (inSelector > 0 && inSelector < c.fields.length) {
      return unsupported(
        `selector splits a single jointchain component (fields: ${c.fields.join(', ')})`,
        c.expr);
    }
  }
  return { selected, unselected };
}

// The disintegration admissibility invariant: no unselected variate may
// depend on a selected variate. (Otherwise the marginal can't be
// constructed without integration — the selected value would need to be
// available when sampling the unselected.) The chain-suffix constraint
// for jointchain follows from this as a special case.
function admissibilityCheck(decomp: Decomposition, partition: Partition, node: any): { ok: true } | { kind: 'unsupported', reason: string, blockingNode: any, detail?: string } {
  const selectedNames = new Set(partition.selected.map(v => v.name));
  for (const u of partition.unselected) {
    for (const dep of u.deps) {
      if (selectedNames.has(dep)) {
        // Pinpoint the blocking node — for lawof-record we have the
        // original record entry, for jointchain the chain-component
        // expression. Use u.expr as the closest available locus.
        return unsupported(
          decomp.kind === 'jointchain'
            ? 'selector does not pick a contiguous suffix of the chain — non-suffix disintegration of a chain is not structural'
            : `unselected field '${u.name}' depends on selected field '${dep}' — would require integration`,
          u.expr || node,
          decomp.kind === 'lawof-record'
            ? 'selected variable feeds into the unselected (marginal) side'
            : undefined);
      }
    }
  }
  return { ok: true };
}

// ---------------------------------------------------------------------
// Delegate detection: both sides exactly one bound binding ref
// ---------------------------------------------------------------------

function tryDelegate(decomp: Decomposition, partition: Partition) {
  // Delegate when:
  //   - each side is exactly one source COMPONENT (not just one variate
  //     — a single component might contribute multiple fields).
  //   - both components' expr is a bare binding ref.
  //   - the decomposition kind is jointchain (the only case where the
  //     delegate-style "the existing binding pair already factors the
  //     joint" interpretation matches the user's source structure).
  if (decomp.kind !== 'jointchain') return null;
  // Map variates back to their components and group.
  const sIdxs = new Set(partition.selected.map(v => v.componentIdx));
  const uIdxs = new Set(partition.unselected.map(v => v.componentIdx));
  if (sIdxs.size !== 1 || uIdxs.size !== 1) return null;
  const sComp = decomp.components[[...sIdxs][0]];
  const uComp = decomp.components[[...uIdxs][0]];
  if (!sComp.bindingRef || !uComp.bindingRef) return null;
  return delegate(sComp.bindingRef, uComp.bindingRef);
}

// ---------------------------------------------------------------------
// Synthesize: emit kernel + marginal AST.
// ---------------------------------------------------------------------
//
// Shape choice per decomposition kind:
//
//   lawof-record:  prior  = lawof(record(unselected fields with original variate refs))
//                  kernel = kernelof(record(selected fields with original variate refs),
//                                    <unselected name>=<unselected variate>, ...)
//                  (the slice-and-rebuild form, matching spec §06's
//                  Bayesian-model use case.)
//
//   joint:         prior  = joint(unselected fields → original measure exprs)
//                  kernel = joint(selected fields → original measure exprs)
//                  (constant kernel — components are independent.)
//
//   jointchain:    prior  = jointchain(unselected fields ...) [or the single
//                           component if there's only one]
//                  kernel = functionof/kernelof(jointchain or single component,
//                                               <unselected name>=<unselected name>, ...)
//                  (the chain semantics — earlier variates are inputs to
//                  later ones — is encoded as explicit boundary inputs.)
//
// Normalisation invariants:
//   - A single-field `joint(name=M)` over the same M is the identity
//     under relabel; we collapse this to plain M (matching how spec
//     identities §06 reduce singleton joints).
//   - jointchain with one element is just that element.

function synthesizePlan(decomp: Decomposition, partition: Partition, bindings: any, ctx: any) {
  switch (decomp.kind) {
    case 'lawof-record':  return synthesizeLawofRecord(decomp, partition, ctx);
    case 'joint':         return synthesizeJoint(decomp, partition, ctx);
    case 'jointchain':    return synthesizeJointchain(decomp, partition, bindings, ctx);
  }
}

function synthesizeLawofRecord(decomp: Decomposition, partition: Partition, ctx: any) {
  const recordVar = decomp.recordVar!;
  // prior = lawof(record(unselected fields, original variate refs))
  const priorRecordArgs = partition.unselected.map(u =>
    mkKwArg(u.name, mkIdent(recordVar.get(u.name)!, ctx.source), ctx.source));
  const priorExpr = mkCall('lawof',
    [mkCall('record', priorRecordArgs, ctx.source)], ctx.source);

  // kernel = kernelof(record(selected fields, original variate refs),
  //                   <unselected name>=<variate>, ...)
  const kernelRecordArgs = partition.selected.map(s =>
    mkKwArg(s.name, mkIdent(recordVar.get(s.name)!, ctx.source), ctx.source));
  const kernelArgs: any[] = [mkCall('record', kernelRecordArgs, ctx.source)];
  for (const u of partition.unselected) {
    kernelArgs.push(mkKwArg(u.name, mkIdent(recordVar.get(u.name)!, ctx.source), ctx.source));
  }
  return synthesized(mkCall('kernelof', kernelArgs, ctx.source), priorExpr);
}

function synthesizeJoint(decomp: Decomposition, partition: Partition, ctx: any) {
  // Both sides stay as joint(...) — including single-field joints.
  // A single-field `joint(name = M)` is NOT semantically `M`; it's
  // `relabel(M, [name])`, a measure over single-field records. The
  // shape difference matters for downstream type/density use.
  const kernelExpr = mkCall('joint',
    partition.selected.map(s => mkKwArg(s.name, s.expr, ctx.source)),
    ctx.source);
  const priorExpr = mkCall('joint',
    partition.unselected.map(u => mkKwArg(u.name, u.expr, ctx.source)),
    ctx.source);
  void decomp;
  return synthesized(kernelExpr, priorExpr);
}

function synthesizeJointchain(decomp: Decomposition, partition: Partition, bindings: any, ctx: any) {
  // Synthesize at the COMPONENT level (not the variate level) so
  // positional multi-field components stay together. Group the
  // partition's variates by their componentIdx.
  const sCompIdxs = [...new Set(partition.selected.map(v => v.componentIdx))].sort((a, b) => a - b);
  const uCompIdxs = [...new Set(partition.unselected.map(v => v.componentIdx))].sort((a, b) => a - b);
  const sComps = sCompIdxs.map(i => decomp.components[i]);
  const uComps = uCompIdxs.map(i => decomp.components[i]);

  // Prior expression: jointchain (preserving the source shape — keyword
  // or positional) over unselected components. Single-component prior
  // collapses to the component expression itself.
  const priorExpr = uComps.length === 1
    ? uComps[0].expr
    : (decomp.positional
        ? mkCall('jointchain', uComps.map(c => c.expr), ctx.source)
        : mkCall('jointchain',
            uComps.map(c => mkKwArg(c.fields[0], c.expr, ctx.source)),
            ctx.source));

  // Kernel body: same shape, over selected components.
  const kernelBody = sComps.length === 1
    ? sComps[0].expr
    : (decomp.positional
        ? mkCall('jointchain', sComps.map(c => c.expr), ctx.source)
        : mkCall('jointchain',
            sComps.map(c => mkKwArg(c.fields[0], c.expr, ctx.source)),
            ctx.source));

  // Boundary inputs are the unselected variates' field names (one
  // boundary per variate, including all fields of multi-field
  // positional components).
  const unselectedNames = partition.unselected.map(u => u.name);
  const kernelExpr = wrapAsKernelOrFunctionOf(kernelBody, unselectedNames, bindings, ctx);
  return synthesized(kernelExpr, priorExpr);
}

// Build `kernelof(body, n=n, ...)` (when body is a value) or
// `functionof(body, n=n, ...)` (when body is a measure) — or just
// `body` if no boundary names are supplied (constant kernel = the
// measure / value itself).
//
// Spec §sec:kernelof: `kernelof`'s first arg must not be a measure;
// reifying a measure-typed body uses `functionof` instead.
function wrapAsKernelOrFunctionOf(body: any, boundaryNames: string[], bindings: any, ctx: any) {
  if (boundaryNames.length === 0) return body;
  const args: any[] = [body];
  for (const n of boundaryNames) {
    args.push(mkKwArg(n, mkIdent(n, ctx.source), ctx.source));
  }
  const op = isMeasureExpr(body, bindings) ? 'functionof' : 'kernelof';
  return mkCall(op, args, ctx.source);
}

// ---------------------------------------------------------------------
// Top-level dispatcher
// ---------------------------------------------------------------------

function disintegratePlan(joint: any, selector: any, bindings: any, ctx: any): any {
  ctx = ctx || { seen: new Set(), source: null };

  // Unwrap Identifiers (cycle-guarded).
  let node = joint;
  while (node && node.type === 'Identifier') {
    if (ctx.seen.has(node.name)) {
      return unsupported('cycle in identifier resolution', node);
    }
    const seen = new Set(ctx.seen); seen.add(node.name);
    const b = bindings.get(node.name);
    if (!b || !b.node || !b.node.value) {
      return unsupported(`unknown or value-less identifier '${node.name}'`, node);
    }
    if (!ctx.source) ctx.source = node.name;
    node = b.node.value;
    ctx = Object.assign({}, ctx, { seen });
  }

  // 1. Decompose.
  const decomp = decompose(node, bindings, ctx);
  if (isUnsupported(decomp)) {
    return unsupported(decomp.reason, decomp.blockingNode, decomp.detail);
  }

  // 2. Partition.
  const partition = partitionVariates(decomp, selector, node);
  if (isUnsupported(partition)) {
    return unsupported(partition.reason, partition.blockingNode, partition.detail);
  }

  // 3. Admissibility.
  const adm = admissibilityCheck(decomp, partition, node);
  if (isUnsupported(adm)) {
    return unsupported(adm.reason, adm.blockingNode, adm.detail);
  }

  // 4. Delegate if the user already wrote the factorization out as
  // distinct binding refs.
  const del = tryDelegate(decomp, partition);
  if (del) return del;

  // 5. Synthesize.
  return synthesizePlan(decomp, partition, bindings, ctx);
}

module.exports = {
  disintegratePlan,
  // Plan constructors (exported for the renderer / tests).
  delegate, synthesized, unsupported,
  // AST helpers (handy for tests).
  mkIdent, mkKwArg, mkCall,
  // Introspection helpers (re-exported in case other engine pieces grow
  // to need them — e.g., visualization for kernel/function distinction).
  namedOutputFields, tryRecord,
};
