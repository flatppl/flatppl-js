'use strict';

// Structural disintegration for FlatPPL.
//
// `disintegratePlan(joint, selector, bindings)` walks the joint expression
// and returns a `Plan` describing how to factor it into a kernel and a
// prior. The contract — straight from the FlatPPL spec and Cho-Jacobs §3 —
// is `jointchain(prior, kernel) ≡ joint`.
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
// See `references.md` (Cho & Jacobs 2017/2019; Shan & Ramsey 2017; Pearl
// 1988) for the underlying algebra and the slicing literature this draws
// on.

const ast = require('./ast');
const { collectDeps } = require('./analyzer');

// Minimal AST construction helpers, all stamped with synthLoc(source).
function mkIdent(name, source)        { return ast.Identifier(name, ast.synthLoc(source)); }
function mkString(value, source)      { return ast.StringLiteral(value, JSON.stringify(value), ast.synthLoc(source)); }
function mkArray(elements, source)    { return ast.ArrayLiteral(elements, ast.synthLoc(source)); }
function mkKwArg(name, value, source) { return ast.KeywordArg(name, value, ast.synthLoc(source)); }
function mkCall(name, args, source)   {
  return ast.CallExpr(mkIdent(name, source), args, ast.synthLoc(source));
}

// Plan constructors.
function delegate(kernelBinding, priorBinding) {
  return {
    kind: 'delegate',
    kernel: { binding: kernelBinding },
    prior:  { binding: priorBinding },
  };
}

function synthesized(kernelExpr, priorExpr) {
  return { kind: 'synthesized', kernel: kernelExpr, prior: priorExpr };
}

function unsupported(reason, blockingNode, detail) {
  return { kind: 'unsupported', reason, blockingNode, detail: detail || null };
}

// === Selector / field utilities ====================================

// Normalise a selector AST argument to an array of field names.
// disintegrate("name", joint)   -> ['name']
// disintegrate(["a", "b"], j)   -> ['a', 'b']
// Returns null if the selector isn't a static string-or-string-array.
function normaliseSelector(selectorAst) {
  if (!selectorAst) return null;
  if (selectorAst.type === 'StringLiteral') return [selectorAst.value];
  if (selectorAst.type === 'ArrayLiteral') {
    const out = [];
    for (const el of selectorAst.elements) {
      if (el.type !== 'StringLiteral') return null;
      out.push(el.value);
    }
    return out;
  }
  return null;
}

// Try to read the field map from `record(name = expr, ...)`.
// Returns Map<name, expr> or null if the call isn't a static record.
function tryRecord(node) {
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
// field names, or [] if the structure isn't statically resolvable. Used to
// match selectors against jointchain components and to recognise positional
// joints whose components carry implicit names.
function namedOutputFields(node, bindings, seen) {
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
    const firstArg = node.args.find(a => a.type !== 'KeywordArg');
    if (!firstArg) return [];
    const rec = tryRecord(firstArg);
    if (rec) return [...rec.keys()];
    return namedOutputFields(firstArg, bindings, seen);
  }
  if (callee === 'joint' || callee === 'jointchain') {
    if (node.args.length > 0 && node.args.every(a => a.type === 'KeywordArg')) {
      return node.args.map(a => a.name);
    }
    return [];
  }
  if (callee === 'relabel') {
    if (node.args.length >= 2 && node.args[1].type === 'ArrayLiteral') {
      const out = [];
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
// binding names reachable. Stops at unbound names. Used by the
// `lawof(record(...))` slice-and-rebuild for the admissibility check and
// for boundary-set computation.
function collectAncestorNames(node, bindings) {
  const result = new Set();
  const definedNames = new Set(bindings.keys());
  function visit(name) {
    if (result.has(name)) return;
    result.add(name);
    const b = bindings.get(name);
    if (!b || !b.deps) return;
    for (const d of b.deps) visit(d);
  }
  // The entry point is an expression — collect its direct deps first.
  const { deps } = collectDeps(node, definedNames);
  for (const d of deps) visit(d);
  return result;
}

// === Top-level dispatcher ==========================================

function disintegratePlan(joint, selector, bindings, ctx) {
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

  if (!node || node.type !== 'CallExpr' || !node.callee || node.callee.type !== 'Identifier') {
    return unsupported('joint expression is not a measure-algebra call', node);
  }

  const callee = node.callee.name;
  switch (callee) {
    case 'lawof':      return ruleLawof(node, selector, bindings, ctx);
    case 'joint':      return ruleJoint(node, selector, bindings, ctx);
    case 'jointchain': return ruleJointchain(node, selector, bindings, ctx);

    // v2 — placeholders, filled in below.
    case 'chain':      return ruleChain(node, selector, bindings, ctx);
    case 'relabel':    return ruleRelabel(node, selector, bindings, ctx);
    case 'pushfwd':    return rulePushfwd(node, selector, bindings, ctx);

    // v3 territory.
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

// === Rule: lawof(record(name = node, ...)) =========================
//
// This is the slice-and-rebuild case. The joint is the law of a record
// over named stochastic value nodes. We classify the record's entries by
// the selector, check admissibility (unselected entries must not depend
// on any selected entry), compute the kernel's boundary inputs, and
// rebuild two FlatPPL expressions:
//   prior  = lawof(record(<unselected>))
//   kernel = kernelof(record(<selected>), <boundary> = <boundary>, ...)
//
// Per Cho-Jacobs Definition 3.5: this is the canonical disintegration of
// a joint state into marginal-on-X plus channel-X→Y. Our admissibility
// check enforces the structural condition that makes the channel realisable
// without integration (Pearl-style factorization).

function ruleLawof(node, selector, bindings, ctx) {
  if (node.args.length !== 1) {
    return unsupported('lawof must be unary', node);
  }
  const rec = tryRecord(node.args[0]);
  if (!rec) {
    return unsupported('lawof of non-record cannot be structurally disintegrated', node.args[0]);
  }

  // Each record entry's value should be an Identifier referring to a node
  // in the value graph; that's the FlatPPL pattern for joint measures.
  for (const [fname, val] of rec) {
    if (val.type !== 'Identifier') {
      return unsupported(
        `record field '${fname}' is not a bare identifier — structural disintegration requires named stochastic nodes`,
        val);
    }
  }

  const allFields = [...rec.keys()];
  const selected   = allFields.filter(f => selector.includes(f));
  const unselected = allFields.filter(f => !selector.includes(f));

  // Selector must be a subset of the record's fields.
  for (const s of selector) {
    if (!rec.has(s)) {
      return unsupported(`selector '${s}' is not a field of the joint's record`, node);
    }
  }
  if (selected.length === 0) {
    return unsupported('selector matches no record fields', node);
  }
  if (unselected.length === 0) {
    return unsupported('selector covers all fields; nothing left for the prior', node);
  }

  // Admissibility: unselected entries' value-graph ancestors must not
  // include any selected entry's bound node. (Equivalent to "selected
  // does not feed into unselected" — a cycle test in the marginal direction.)
  const selectedNames = new Set(selected.map(f => rec.get(f).name));
  for (const u of unselected) {
    const uVar = rec.get(u).name;
    const ancestors = collectAncestorNames(rec.get(u), bindings);
    ancestors.delete(uVar); // exclude self
    for (const sel of selectedNames) {
      if (ancestors.has(sel)) {
        return unsupported(
          `unselected field '${u}' depends on selected field '${nameOf(rec, sel)}' — would require integration`,
          rec.get(u),
          'selected variable feeds into the unselected (marginal) side');
      }
    }
  }

  // Boundary set: every unselected field becomes a boundary input of the
  // kernel, named by the field name and bound to the field's underlying
  // variable. This matches the categorical signature
  //   kernel : prior_space -> joint_space
  // even when a particular selected entry doesn't actually reference some
  // boundary; the kernel is then constant in that input. We don't try to
  // minimise the input set — that's a downstream optimisation.

  // Build prior  = lawof(record(<unselected fields>))
  const priorRecordArgs = unselected.map(u =>
    mkKwArg(u, mkIdent(rec.get(u).name, ctx.source), ctx.source));
  const priorExpr = mkCall('lawof', [mkCall('record', priorRecordArgs, ctx.source)], ctx.source);

  // Build kernel = kernelof(record(<selected fields>), <field>=<var>, ...)
  const kernelRecordArgs = selected.map(s =>
    mkKwArg(s, mkIdent(rec.get(s).name, ctx.source), ctx.source));
  const kernelArgs = [mkCall('record', kernelRecordArgs, ctx.source)];
  for (const u of unselected) {
    kernelArgs.push(mkKwArg(u, mkIdent(rec.get(u).name, ctx.source), ctx.source));
  }
  const kernelExpr = mkCall('kernelof', kernelArgs, ctx.source);

  return synthesized(kernelExpr, priorExpr);
}

function nameOf(rec, varName) {
  for (const [f, v] of rec) if (v.type === 'Identifier' && v.name === varName) return f;
  return varName;
}

// === Rule: joint(name = M, ...) (independent product) ==============
//
// Components are independent, so the kernel side and prior side are each
// just another `joint(...)` over their share. Constant kernel (independent
// of prior values) is correct because of the independence — Cho-Jacobs
// section on parallel composition.

function ruleJoint(node, selector, bindings, ctx) {
  // Only the keyword form is structurally tractable as written. Positional
  // joint(M1, M2, ...) merges variates via `cat`; without explicit field
  // names there's nothing for the selector to match.
  if (node.args.length === 0) {
    return unsupported('empty joint()', node);
  }
  if (!node.args.every(a => a.type === 'KeywordArg')) {
    return unsupported(
      'positional joint(M1, M2, ...) has no structural disintegration rule (use joint(name=M, ...) for selector matching)',
      node);
  }

  const fields = new Map();
  for (const a of node.args) fields.set(a.name, a.value);

  for (const s of selector) {
    if (!fields.has(s)) {
      return unsupported(`selector '${s}' is not a field of joint(...)`, node);
    }
  }
  const selectedNames = node.args.filter(a => selector.includes(a.name));
  const unselectedNames = node.args.filter(a => !selector.includes(a.name));
  if (selectedNames.length === 0) {
    return unsupported('selector matches no joint fields', node);
  }
  if (unselectedNames.length === 0) {
    return unsupported('selector covers all fields; nothing left for the prior', node);
  }

  // Each side is a fresh joint(...) over its share. Reuse the original
  // value sub-expressions verbatim — they still type-check as measures.
  const kernelExpr = mkCall('joint',
    selectedNames.map(a => mkKwArg(a.name, a.value, ctx.source)),
    ctx.source);
  const priorExpr = mkCall('joint',
    unselectedNames.map(a => mkKwArg(a.name, a.value, ctx.source)),
    ctx.source);

  return synthesized(kernelExpr, priorExpr);
}

// === Rule: jointchain(...) =========================================
//
// Two forms. Both factor cleanly only when the selector picks a contiguous
// suffix of the chain — otherwise we'd be inverting the dependency order
// and would need analytic reasoning, not structural rewriting.

function ruleJointchain(node, selector, bindings, ctx) {
  if (node.args.length === 0) {
    return unsupported('empty jointchain()', node);
  }
  const allKeyword = node.args.every(a => a.type === 'KeywordArg');
  const allPositional = node.args.every(a => a.type !== 'KeywordArg');

  if (allKeyword) {
    return jointchainKeyword(node, selector, ctx);
  }
  if (allPositional) {
    return jointchainPositional(node, selector, bindings, ctx);
  }
  return unsupported('jointchain mixing keyword and positional args', node);
}

function jointchainKeyword(node, selector, ctx) {
  const fieldNames = node.args.map(a => a.name);

  for (const s of selector) {
    if (!fieldNames.includes(s)) {
      return unsupported(`selector '${s}' is not a field of jointchain(...)`, node);
    }
  }

  // Suffix check: every selected field must appear at the tail.
  const firstSelectedIdx = fieldNames.findIndex(f => selector.includes(f));
  if (firstSelectedIdx < 0) {
    return unsupported('selector matches no jointchain fields', node);
  }
  for (let i = firstSelectedIdx; i < fieldNames.length; i++) {
    if (!selector.includes(fieldNames[i])) {
      return unsupported(
        'selector does not pick a contiguous suffix of the chain — non-suffix disintegration of a chain is not structural',
        node.args[i]);
    }
  }
  if (firstSelectedIdx === 0) {
    return unsupported(
      'selector covers all chain fields; nothing left for the prior', node);
  }

  const priorArgs  = node.args.slice(0, firstSelectedIdx);
  const kernelArgs = node.args.slice(firstSelectedIdx);

  // A jointchain of one element is just that element.
  const kernelBody = (kernelArgs.length === 1)
    ? kernelArgs[0].value
    : mkCall('jointchain', kernelArgs.map(a => mkKwArg(a.name, a.value, ctx.source)), ctx.source);
  const priorExpr  = (priorArgs.length === 1)
    ? priorArgs[0].value
    : mkCall('jointchain', priorArgs.map(a => mkKwArg(a.name, a.value, ctx.source)), ctx.source);

  // Wrap kernel side in `kernelof(body, n1=n1, ...)` so the chain's
  // implicit "earlier-field-as-input" semantics is encoded as explicit
  // boundary inputs. The boundary names refer to the joint's variates,
  // which typically aren't bindings in the host scope — so when the
  // renderer walks the resulting kernelof, `extractBoundaries` produces
  // synthetic boundary nodes for them, exactly as if the user had hand-
  // written the kernel.
  const kernelExpr = wrapAsKernelOf(kernelBody, priorArgs.map(a => a.name), ctx);
  return synthesized(kernelExpr, priorExpr);
}

// Build `kernelof(body, n=n, ...)` — or just `body` if no boundary names
// are supplied (constant kernel = measure).
function wrapAsKernelOf(body, boundaryNames, ctx) {
  if (boundaryNames.length === 0) return body;
  const args = [body];
  for (const n of boundaryNames) {
    args.push(mkKwArg(n, mkIdent(n, ctx.source), ctx.source));
  }
  return mkCall('kernelof', args, ctx.source);
}

// Positional jointchain — bayesian_inference_3's case. Each component is
// an inline measure or a binding identifier whose named output fields we
// can introspect (via `namedOutputFields`). We try to find the contiguous
// suffix split in component-space such that the union of those components'
// fields exactly equals the selector. If the components are bare
// Identifiers, emit a Delegate plan referencing them by name; otherwise
// synthesize.
function jointchainPositional(node, selector, bindings, ctx) {
  const components = node.args.map(arg => ({
    arg,
    fields: namedOutputFields(arg, bindings),
  }));

  // For each component, the set of its named fields covered by the selector.
  // We want: there's some split index `k` such that components [k..] cover
  // exactly the selector and components [0..k-1] don't intersect it.
  const allComponentFields = new Set();
  for (const c of components) for (const f of c.fields) allComponentFields.add(f);

  for (const s of selector) {
    if (!allComponentFields.has(s)) {
      return unsupported(
        `selector '${s}' is not in any jointchain component's named output fields`,
        node);
    }
  }

  // Find the first component that has any selector-matching field.
  const firstHitIdx = components.findIndex(c => c.fields.some(f => selector.includes(f)));
  if (firstHitIdx < 0) {
    return unsupported('selector matches no jointchain component', node);
  }
  if (firstHitIdx === 0) {
    return unsupported('selector covers all jointchain components; nothing left for the prior', node);
  }

  // Past the split, every component must be entirely on the kernel side
  // (all its fields in selector). Before the split, no component may
  // contribute any field to the selector.
  for (let i = 0; i < firstHitIdx; i++) {
    if (components[i].fields.some(f => selector.includes(f))) {
      return unsupported('non-suffix split in positional jointchain', node.args[i]);
    }
  }
  for (let i = firstHitIdx; i < components.length; i++) {
    for (const f of components[i].fields) {
      if (!selector.includes(f)) {
        return unsupported(
          `kernel-side component contributes field '${f}' that is not in the selector`,
          node.args[i]);
      }
    }
  }
  // Selector must be exactly the union of kernel-side fields.
  const kernelFields = new Set();
  for (let i = firstHitIdx; i < components.length; i++) {
    for (const f of components[i].fields) kernelFields.add(f);
  }
  for (const s of selector) {
    if (!kernelFields.has(s)) {
      return unsupported(
        `selector '${s}' is not provided by any kernel-side component`, node);
    }
  }

  const priorComps = components.slice(0, firstHitIdx);
  const kernelComps = components.slice(firstHitIdx);

  // Delegate when each side is exactly one component and that component
  // is a binding identifier — the user-visible structure is preserved.
  const canDelegate = priorComps.length === 1 && kernelComps.length === 1
    && priorComps[0].arg.type === 'Identifier'
    && kernelComps[0].arg.type === 'Identifier';
  if (canDelegate) {
    return delegate(kernelComps[0].arg.name, priorComps[0].arg.name);
  }

  // Otherwise synthesize. A single-component side collapses to that
  // component verbatim; multi-component sides become a positional
  // jointchain. The kernel side is wrapped in kernelof(...) with the
  // prior-side fields as boundary inputs, matching the keyword-form
  // treatment.
  function build(side) {
    if (side.length === 1) return side[0].arg;
    return mkCall('jointchain', side.map(c => c.arg), ctx.source);
  }
  const priorBoundaryNames = [];
  let priorFieldsAllKnown = true;
  for (const c of priorComps) {
    if (c.fields.length === 0) { priorFieldsAllKnown = false; break; }
    for (const f of c.fields) priorBoundaryNames.push(f);
  }
  const kernelBody = build(kernelComps);
  const kernelExpr = priorFieldsAllKnown
    ? wrapAsKernelOf(kernelBody, priorBoundaryNames, ctx)
    : kernelBody;
  return synthesized(kernelExpr, build(priorComps));
}

// === Rule: chain(M, K1, ..., Kn) ====================================
//
// `chain` is Kleisli composition that drops intermediate variates: only
// the last kernel's variate survives. The output measure equals the
// marginalization integral $\int K_n(\ldots, B)\, d(\mathrm{chain\ldots})$.
// Disintegrating the result generally requires inverting that integral,
// which is exactly the analytic Bayes-update problem we're refusing to
// engage with in this rewriter (cf. Hakaru's symbolic disintegration).
//
// We bail out with a clear reason. The single-component degenerate form
// `chain(M)` is just `M`; we treat that as Unsupported here too because
// users won't write it, and the dispatcher would already have unwrapped
// any Identifier alias before reaching this rule.

function ruleChain(node, selector, bindings, ctx) {
  void selector; void bindings; void ctx;
  return unsupported(
    "'chain' has no structural disintegration rule", node,
    'chain(M, K1, ..., Kn) marginalizes out intermediate variates; recovering them from the result requires integration');
}

// === Rule: relabel(M, [n1, n2, ...]) ================================
//
// `relabel` only renames an existing positional structure. The
// structurally tractable cases are when the inner expression is a
// positional `joint(...)` or positional `jointchain(...)`; we lift the
// names onto the components and recurse. (relabel(M, [n]) where M is a
// scalar measure can't be split — there's only one variate — so we let
// the recursive call return Unsupported with the right reason.)

function ruleRelabel(node, selector, bindings, ctx) {
  if (node.args.length !== 2) {
    return unsupported('relabel must take (measure, [names])', node);
  }
  const inner = node.args[0];
  const namesArg = node.args[1];
  if (namesArg.type !== 'ArrayLiteral') {
    return unsupported('relabel names argument must be an array literal', namesArg);
  }
  const names = [];
  for (const el of namesArg.elements) {
    if (el.type !== 'StringLiteral') {
      return unsupported('relabel names must be string literals', el);
    }
    names.push(el.value);
  }

  // The interesting case: relabel(joint(M1, M2, ...), [n1, n2, ...]) or
  // the corresponding jointchain form. Lift the names onto the components
  // and recurse.
  if (inner && inner.type === 'CallExpr'
      && inner.callee && inner.callee.type === 'Identifier'
      && (inner.callee.name === 'joint' || inner.callee.name === 'jointchain')
      && inner.args.length === names.length
      && inner.args.every(a => a.type !== 'KeywordArg')) {
    const lifted = ast.CallExpr(
      inner.callee,
      inner.args.map((a, i) => mkKwArg(names[i], a, ctx.source)),
      ast.synthLoc(ctx.source),
    );
    return disintegratePlan(lifted, selector, bindings, ctx);
  }

  // Other inner shapes (a bare distribution, an Identifier, another
  // measure-algebra op): rename doesn't change the disintegration story
  // because there's no field structure to split along. Defer to inner
  // when names map 1-to-1 onto known fields, otherwise refuse.
  const innerFields = namedOutputFields(inner, bindings);
  if (innerFields.length === names.length) {
    // Pure rename: build a selector translated back into inner's field
    // names (positionally) and recurse.
    const reverseMap = new Map();
    for (let i = 0; i < names.length; i++) reverseMap.set(names[i], innerFields[i]);
    const translated = [];
    for (const s of selector) {
      const t = reverseMap.get(s);
      if (t == null) {
        return unsupported(
          `selector '${s}' is not a field of the relabeled measure`, namesArg);
      }
      translated.push(t);
    }
    return disintegratePlan(inner, translated, bindings, ctx);
  }

  return unsupported(
    'relabel inner expression has no statically-resolvable field structure',
    inner,
    'only relabel of a positional joint/jointchain or a measure with named output fields is supported');
}

// === Rule: pushfwd(f, M) ============================================
//
// Two distinct sub-cases the spec calls out:
//
//  - **Projection**: `pushfwd(fn(get(_, [k1, k2, ...])), M)` selects a
//    subset of M's named fields. If the disintegration selector is
//    contained in the projected key set, we could recurse on M with the
//    same selector and re-wrap each side. But the prior side then lives
//    in M's full space — projection past the disintegration changes the
//    space, so the algebraic identity `jointchain(prior, kernel) = joint`
//    only holds modulo the projection. Surfacing this carefully would
//    require extending Plan to carry a "post-projection" annotation; we
//    defer.
//
//  - **Bijection**: `pushfwd(bijection(f, f_inv, vol), M)`. Generally
//    not block-separable along the disintegration boundary unless the
//    bijection is declared as a product of per-coordinate maps; we don't
//    have that information today.
//
// Both fall to Unsupported with detailed reasons.

function rulePushfwd(node, selector, bindings, ctx) {
  void selector; void bindings; void ctx;
  if (node.args.length !== 2) {
    return unsupported('pushfwd must take (f, M)', node);
  }
  return unsupported(
    "'pushfwd' has no v1+v2 structural disintegration rule", node,
    'projection (subset selection) and bijection (block-separable transform) cases require post-transform Plan annotations');
}

module.exports = {
  disintegratePlan,
  // Plan constructors (exported for the renderer / tests).
  delegate, synthesized, unsupported,
  // AST helpers (handy for tests).
  mkIdent, mkString, mkArray, mkKwArg, mkCall,
  // Introspection helpers (re-exported in case other engine pieces grow
  // to need them — e.g., visualization for kernel/function distinction).
  namedOutputFields, tryRecord, normaliseSelector,
};
