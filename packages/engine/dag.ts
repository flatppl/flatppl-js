'use strict';

const { extractBoundaries, collectDeps, isMeasureExpr, computePhasesForScope } = require('./analyzer.ts');
const { resolveCallableAlias } = require('./ir-shared.ts');

// Effective doc-comment for a binding (spec §04 doc-comments): the binding's
// OWN doc wins; otherwise, if it is a bare alias — including a cross-module one
// like `theta1_dist = priors.theta1_dist`, which the linker rewrites to a
// namespaced ref — inherit the canonical target's doc, recursively across
// module hops (resolveCallableAlias follows the whole chain in the linked
// graph). So the DAG hover shows the doc that lives on the originating binding.
// Returns null when none is found.
function effectiveDoc(name: string, bindings: any): any {
  if (!bindings) return null;
  const b = bindings.get(name);
  if (b && b.node && b.node.doc) return b.node.doc;
  const canon = resolveCallableAlias(name, bindings);
  if (canon !== name) {
    const cb = bindings.get(canon);
    if (cb && cb.node && cb.node.doc) return cb.node.doc;
  }
  return null;
}

// Resolve a binding's "effective" RHS view. For most bindings this is
// just the literal RHS; for disintegration results that have a
// synthesized Plan, it returns the Plan's expression so the renderer
// can treat them as bona-fide kernelof/lawof bindings.
function eff(b: any) {
  if (!b) return { value: null, deps: [], callDeps: [] };
  return {
    value:    b.effectiveValue    != null ? b.effectiveValue    : (b.node && b.node.value),
    deps:     b.effectiveDeps     != null ? b.effectiveDeps     : (b.deps     || []),
    callDeps: b.effectiveCallDeps != null ? b.effectiveCallDeps : (b.callDeps || []),
  };
}

// Reification kinds for visualization purposes.
//   'measure' — lawof(x): always produces a measure (closed or parametric)
//   'kernel'  — kernelof(x, ...) or functionof(measure, ...): a Markov kernel
//   'function'— functionof(value, ...): a deterministic function
//   'lambda'  — fn(...): a lambda
function reificationKind(binding: any, bindings: any) {
  if (!binding) return null;
  switch (binding.type) {
    case 'lawof':     return 'measure';
    case 'kernelof':  return 'kernel';
    case 'fn':        return 'lambda';
    // bayesupdate(L, prior) produces a measure (the unnormalized
    // posterior). Surface it as such so the DAG renders it with the
    // same color/shape as any other measure-producing operation
    // (e.g. joint_model). bayesupdate isn't a reification — no
    // bubble is drawn — so the `isReifAnchor` rule keeps it on the
    // default solid fill, matching joint_model's look.
    case 'bayesupdate': return 'measure';
    case 'functionof': {
      const firstArg = firstPositionalArg(eff(binding).value);
      return isMeasureExpr(firstArg, bindings) ? 'kernel' : 'function';
    }
    case 'call': {
      // Plain call bindings that *produce* a measure (e.g.
      // `theta1_dist = Normal(...)`) read as measures in the graph,
      // even though they're not explicitly reified via `lawof`.
      return isMeasureExpr(eff(binding).value, bindings) ? 'measure' : null;
    }
    default: return null;
  }
}

function firstPositionalArg(callExpr: any) {
  if (!callExpr || !callExpr.args) return null;
  for (const arg of callExpr.args) {
    if (arg.type !== 'KeywordArg') return arg;
  }
  return null;
}

// Collect every Placeholder reference (in `_name_` form) under an AST subtree.
function collectPlaceholders(node: any, out: Set<any>) {
  if (!node) return;
  if (node.type === 'Placeholder') {
    out.add('_' + (node.name || '') + '_');
    return;
  }
  switch (node.type) {
    case 'BinaryExpr':   collectPlaceholders(node.left, out); collectPlaceholders(node.right, out); break;
    case 'UnaryExpr':    collectPlaceholders(node.operand, out); break;
    case 'CallExpr':
      collectPlaceholders(node.callee, out);
      for (const a of node.args) collectPlaceholders(a, out);
      break;
    case 'IndexExpr':
      collectPlaceholders(node.object, out);
      for (const i of node.indices) collectPlaceholders(i, out);
      break;
    case 'FieldAccess':  collectPlaceholders(node.object, out); break;
    case 'KeywordArg':   collectPlaceholders(node.value, out); break;
    case 'ArrayLiteral':
    case 'TupleLiteral':
      for (const el of node.elements) collectPlaceholders(el, out);
      break;
  }
}

// Render an AST node to a short source-like string. Used both for node
// labels (with default short maxLen) and hover tooltips (with a longer
// maxLen). Recurses into compound expressions; truncates with "…".
function renderExprShort(node: any, maxLen: number) {
  if (maxLen == null) maxLen = 24;
  function r(n: any): any {
    if (!n) return '';
    switch (n.type) {
      case 'Identifier':     return n.name;
      case 'NumberLiteral':  return n.raw != null ? n.raw : String(n.value);
      case 'StringLiteral':  return '"' + n.value + '"';
      case 'BoolLiteral':    return String(n.value);
      case 'ConstantRef':    return n.name;
      case 'SetRef':         return n.name;
      case 'Placeholder':    return '_' + (n.name || '') + '_';
      case 'Hole':           return '_';
      case 'BinaryExpr':     return r(n.left) + ' ' + n.op + ' ' + r(n.right);
      case 'UnaryExpr':      return n.op + r(n.operand);
      case 'CallExpr':       return r(n.callee) + '(' + (n.args || []).map(r).join(', ') + ')';
      case 'KeywordArg':     return n.name + ' = ' + r(n.value);
      case 'IndexExpr':      return r(n.object) + '[' + (n.indices || []).map(r).join(', ') + ']';
      case 'FieldAccess':    return r(n.object) + '.' + n.field;
      case 'ArrayLiteral':   return '[' + (n.elements || []).map(r).join(', ') + ']';
      case 'TupleLiteral':   return '(' + (n.elements || []).map(r).join(', ') + ')';
      default:               return '(…)';
    }
  }
  const s = r(node);
  return s.length > maxLen ? s.slice(0, maxLen - 1) + '…' : s;
}


/**
 * Compute the ancestor sub-DAG of a node.
 * For functionof/kernelof, boundary inputs stop the backwards trace.
 * (lawof is unary and never specifies boundaries directly.)
 *
 * Disintegration results are rendered uniformly with the rest: the
 * analyzer attaches an `effectiveValue/Deps/CallDeps` view derived from
 * the synthesized Plan, and `eff()` reads through to those when present.
 * The renderer just walks the effective expression like any user-written
 * kernelof/lawof binding — boundary inputs declared in the synthesized
 * `kernelof(...)` produce boundaries (and synthetic boundary nodes when
 * the input names don't resolve to bindings in scope).
 *
 * @param {Map} bindings - from analyzer: Map<name, BindingInfo>
 * @param {string} nodeName - the target node name
 * @returns {{ nodes: object[], edges: object[] }}
 */
/**
 * Walk a binding's RHS AST to classify each Identifier by whether it
 * appears wrapped in `lawof(...)` or `draw(...)` (the two domain-
 * lifting operators we surface as synthetic nodes), or used directly.
 *
 * Returns:
 *   {
 *     wrapped: Map<refName, 'lawof' | 'draw'>   — innermost wrapper per ref;
 *                                                 first occurrence wins
 *     direct:  Set<refName>                     — refs used outside any wrapper
 *   }
 *
 * A ref can appear in both maps when the same name is used both
 * directly and inside a wrapper (rare but possible). Refs only
 * inside a wrapper get a synthetic node + edges through it; refs
 * with any direct use additionally get a direct edge. Refs with
 * neither — i.e. AST shapes the walker doesn't traverse — fall
 * through to a direct edge in the caller (defensive default).
 *
 * Skip behaviour:
 *   * If the binding's TOP-LEVEL RHS is itself `draw(...)` (with
 *     binding.type === 'draw') or `lawof(...)` (with binding.type
 *     === 'lawof'), the binding ALREADY represents that operator
 *     and shouldn't redundantly synthesise it. We descend into the
 *     args of the top-level wrapper as if we were starting fresh.
 *   * Reification scopes (functionof / kernelof / fn bodies) belong
 *     to a different namespace; we don't recurse into them.
 */
function collectWrappedRefs(binding: any) {
  const wrapped = new Map<string, any>();
  const direct = new Set<string>();
  if (!binding || !binding.node || !binding.node.value) {
    return { wrapped, direct };
  }
  const root = binding.node.value;
  // If the binding IS a draw / lawof, descend past the top-level
  // operator before classifying.
  let entryNodes = [root];
  if (root.type === 'CallExpr' && root.callee && root.callee.type === 'Identifier'
      && Array.isArray(root.args)) {
    const op = root.callee.name;
    if ((op === 'draw' && binding.type === 'draw')
        || (op === 'lawof' && binding.type === 'lawof')) {
      entryNodes = root.args;
    }
  }
  for (const n of entryNodes) classifyRefUsages(n, wrapped, direct, null);
  return { wrapped, direct };
}

function classifyRefUsages(node: any, wrapped: Map<string, any>, direct: Set<string>, currentWrapper: any) {
  if (!node || typeof node !== 'object') return;

  if (node.type === 'Identifier') {
    if (currentWrapper) {
      if (!wrapped.has(node.name)) wrapped.set(node.name, currentWrapper);
    } else {
      direct.add(node.name);
    }
    return;
  }

  if (node.type === 'CallExpr') {
    const calleeName = node.callee && node.callee.type === 'Identifier'
      ? node.callee.name : null;
    // Switch wrapper when entering draw / lawof. Innermost wins; we
    // don't track nested wrappers (rare and visually noisy).
    let next = currentWrapper;
    if (calleeName === 'draw' || calleeName === 'lawof') next = calleeName;
    // Don't descend into reification scopes — different namespace.
    if (calleeName === 'functionof' || calleeName === 'kernelof' || calleeName === 'fn') {
      return;
    }
    if (Array.isArray(node.args)) {
      for (const a of node.args) classifyRefUsages(a, wrapped, direct, next);
    }
    return;
  }

  if (node.type === 'KeywordArg') {
    classifyRefUsages(node.value, wrapped, direct, currentWrapper);
    return;
  }
  if (node.type === 'BinaryExpr') {
    classifyRefUsages(node.left,    wrapped, direct, currentWrapper);
    classifyRefUsages(node.right,   wrapped, direct, currentWrapper);
    return;
  }
  if (node.type === 'UnaryExpr') {
    classifyRefUsages(node.operand, wrapped, direct, currentWrapper);
    return;
  }
  if (node.type === 'IndexExpr') {
    classifyRefUsages(node.object,  wrapped, direct, currentWrapper);
    if (Array.isArray(node.indices)) {
      for (const i of node.indices) classifyRefUsages(i, wrapped, direct, currentWrapper);
    }
    return;
  }
  if (node.type === 'FieldAccess') {
    classifyRefUsages(node.object, wrapped, direct, currentWrapper);
    return;
  }
  if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') {
    if (Array.isArray(node.elements)) {
      for (const e of node.elements) classifyRefUsages(e, wrapped, direct, currentWrapper);
    }
    return;
  }
  // Other AST node types (literals, placeholders, holes, set/const
  // refs, slice-all) carry no Identifier children we care about.
}

// Collect cross-module member references `mod.field` (spec §04 Module
// composition) used INSIDE a binding's RHS, where `mod` is a `load_module`
// binding (type 'module'). A TOP-LEVEL FieldAccess is skipped: that's an
// ALIAS (`f_b = common.f_b`, `theta1 = common.theta1_dist`) whose own
// binding node already stands for the member under its local name. Only
// genuinely INLINE uses (`common.f_a(theta2)`) become member nodes.
// Returns Map<moduleName, Set<field>>.
function collectModuleMemberRefs(value: any, bindings: any) {
  const out = new Map();
  (function walk(node: any, isRoot: boolean) {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'FieldAccess') {
      const obj = node.object;
      if (!isRoot && obj && obj.type === 'Identifier' && typeof node.field === 'string') {
        const mb = bindings.get(obj.name);
        if (mb && mb.type === 'module') {
          if (!out.has(obj.name)) out.set(obj.name, new Set());
          out.get(obj.name).add(node.field);
        }
      }
      walk(node.object, false);
      return;
    }
    for (const k in node) {
      const v = node[k];
      if (Array.isArray(v)) { for (const c of v) walk(c, false); }
      else if (v && typeof v === 'object' && v.type) walk(v, false);
    }
  })(value, true);
  return out;
}

// If a binding is a BARE alias of a module member (`f_b = common.f_b`,
// the whole RHS is `mod.field` on a `load_module` binding), return its
// `{ module, field }` origin so the binding's own node can carry it — a
// double-click then drills into the module at that member (spec §04),
// exactly like an inline member node. Returns undefined otherwise.
function aliasModuleMember(value: any, bindings: any) {
  if (value && value.type === 'FieldAccess' && value.object
      && value.object.type === 'Identifier' && typeof value.field === 'string') {
    const mb = bindings.get(value.object.name);
    if (mb && mb.type === 'module') {
      return { module: value.object.name, field: value.field };
    }
  }
  return undefined;
}

function computeSubDAG(bindings: any, nodeName: string, opts?: any) {
  const binding = bindings.get(nodeName);
  if (!binding) return { nodes: [], edges: [] };
  // Optional linked (cross-module) bindings: when a member node is
  // synthesised for `mod.field`, its type/kind/phase come from the
  // linked binding `mod$field` (the spliced module member).
  const linkedBindings = (opts && opts.linkedBindings) || null;

  // Disintegration with an Unsupported plan → plain dep trace using the
  // analyzer-recorded deps (the user's literal disintegrate(...) call).
  // Synthesized plans expose their RHS via eff().value/deps below, so the
  // root binding renders like any user-written kernelof/lawof.

  const rootValue = eff(binding).value;
  let boundaryVars = new Set();
  let boundaryLabels = new Map(); // varName -> argName
  // Implicit reification: `kernelof(record(obs = obs))` / `functionof(body)`
  // with NO explicit boundary kwargs reifies the `elementof` leaves it reaches
  // as its boundaries (spec §04). extractBoundaries only sees the explicit
  // kwargs, so flag the discovered elementof-leaf inputs (`b.type === 'input'`)
  // as boundaries below — matching the explicit-kwarg form's backward-trace
  // stop. (The example bayesian_inference_1 uses exactly this implicit form.)
  let implicitKernelRoot = false;

  if (binding.type === 'functionof' || binding.type === 'kernelof') {
    const boundaries = extractBoundaries(rootValue);
    if (boundaries && boundaries.size > 0) {
      boundaryVars = new Set(boundaries.values());
      for (const [argName, varName] of boundaries) {
        boundaryLabels.set(varName, argName);
      }
    } else if (!binding.disintegratePlan) {
      // No explicit boundary kwargs AND not a disintegrate-synthesised kernel
      // (whose ancestors are free inputs of the joint, not reified boundaries)
      // ⇒ a genuine implicit `kernelof(record(…))` / `functionof(body)` whose
      // elementof leaves are its boundaries.
      implicitKernelRoot = true;
    }
  }

  const visited = new Map();
  const edges: any[] = [];

  function visit(name: string, useEffective?: boolean) {
    if (visited.has(name)) return;
    const b = bindings.get(name);
    // Effective-RHS overlays apply only at the inspection root. Transitive
    // visits use the literal RHS so a top-level view still reflects the
    // user's source structure — e.g., descending through `prior2` reaches
    // `joint_model`, not the rewriter's synthesized view of it.
    const e = useEffective ? eff(b) : { value: b && b.node && b.node.value, deps: (b && b.deps) || [], callDeps: (b && b.callDeps) || [] };
    // Inline cross-module member uses in this RHS → explicit member nodes.
    const memberRefs = collectModuleMemberRefs(e.value, bindings);
    const isBoundary = boundaryVars.has(name)
      || (implicitKernelRoot && name !== nodeName && b && b.type === 'input');

    // If this binding is a disintegration result whose Plan came back
    // Unsupported, surface that to the renderer so it can mark the node
    // visually — the trace through it is the literal source, not a
    // structural decomposition.
    const planUnsupported = b && b.disintegratePlan && b.disintegratePlan.kind === 'unsupported'
      ? b.disintegratePlan : null;

    // Type-error surfacing: any analyzer-level error diagnostic that
    // landed on this binding (typeinfer mismatches, undefined refs,
    // …) gets passed through so the renderer can mark the node and
    // the plot pane can show a "semantically invalid" message
    // instead of a generic "not plottable".
    const errorDiags = b && b.diagnostics
      ? b.diagnostics.filter((d: any) => d.severity === 'error')
      : null;

    visited.set(name, {
      id: name,
      label: boundaryLabels.get(name),
      type: b ? b.type : 'unknown',
      kind: reificationKind(b, bindings),
      phase: b ? b.phase : undefined,
      expr: b ? b.rhs : '',
      // Doc-comment attached by the parser (spec §04 §sec:documentation;
      // shape: `{ markup, lines }` or null). Surfaced by the DAG-view
      // hover-info renderer below the existing label/expr line.
      doc: effectiveDoc(name, linkedBindings || bindings),
      line: b ? b.line : -1,
      isBoundary,
      isTarget: name === nodeName,
      unsupported: planUnsupported ? true : undefined,
      unsupportedReason: planUnsupported ? planUnsupported.reason : undefined,
      unsupportedDetail: planUnsupported && planUnsupported.detail ? planUnsupported.detail : undefined,
      errors: (errorDiags && errorDiags.length > 0) ? errorDiags : undefined,
      // FlatPIR-style inferred type/shape for the info bar. Pre-
      // rendered with types.show() so the webview doesn't need to
      // duplicate the rendering logic.
      inferredType: (b && b.inferredType)
        ? require('./types.ts').show(b.inferredType)
        : undefined,
      // Bare alias of a module member (`f_b = common.f_b`) → its origin,
      // so a double-click drills into the module at that member (spec §04).
      moduleMember: aliasModuleMember(e.value, bindings),
    });

    if (isBoundary || !b) return;

    // For lawof/functionof/kernelof, optionally synthesize:
    //   - an anonymous "expression target" node, when the first positional
    //     arg is a compound expression, so the bubble has a clear value-
    //     being-reified that external nodes can tether to;
    //   - boundary input nodes for placeholder kwargs (varName not bound).
    // Skipped for fn-like bindings (their reification has no scope), and
    // for transitive visits to disintegration results (they appear as
    // plain nodes in someone else's trace).
    let inlineExprDeps: any = null;
    let inlineExprId: string | null = null;
    if ((b.type === 'lawof' || b.type === 'functionof' || b.type === 'kernelof')
        && !isFnLike(bindings, name)
        && (useEffective || !b.disintegrateRole)
        && e.value) {
      const firstArg = firstPositionalArg(e.value);
      const placeholdersInBody = new Set();
      if (firstArg) collectPlaceholders(firstArg, placeholdersInBody);

      // Synthesize anonymous expression target when first positional arg
      // is not a bare identifier.
      if (firstArg && firstArg.type !== 'Identifier') {
        const definedNames = new Set(bindings.keys());
        const argResult = collectDeps(firstArg, definedNames);
        inlineExprDeps = argResult.deps;
        inlineExprId = name + ':target';
        // The inline expression's phase is the MAX of its own deps,
        // not b.phase. The wrapping binding (lawof / functionof /
        // kernelof) has its phase computed under absorption rules —
        // lawof reports 'fixed' precisely because it absorbs the
        // stochasticity of its body. The inline body itself is still
        // stochastic per the structural rule (its value depends on
        // stochastic ancestors before lawof absorbs them). Painting
        // the synthetic node with b.phase would mis-classify it as
        // fixed and render it in the fixed-grey style, hiding the
        // stochastic-purple flow into the lawof bubble. Compute
        // from deps instead.
        let inlinePhase = 'fixed';
        for (const dep of inlineExprDeps) {
          const depB = bindings.get(dep);
          const dp = depB && depB.phase;
          if (dp === 'stochastic') { inlinePhase = 'stochastic'; break; }
          if (dp === 'parameterized') inlinePhase = 'parameterized';
        }
        visited.set(inlineExprId, {
          id: inlineExprId,
          // Empty label — anonymous bridge node, identifies itself via
          // hover (rendered first-arg expression).
          label: '',
          type: 'call',
          phase: inlinePhase,
          expr: renderExprShort(firstArg, 200),
          line: b.line,
          isBoundary: false,
          isTarget: false,
        });
        // The inline-expr node feeds *into* this binding's RHS. When
        // the binding is a draw, the inline-expr is the measure being
        // drawn from — that final hop into a draw-typed node is the
        // distinguished "stochastic transition" we visually mark.
        // Deps of the inline-expr itself are still plain data flow.
        const inlineEdgeType = (b && b.type === 'draw') ? 'draw' : 'data';
        edges.push({ source: inlineExprId, target: name, edgeType: inlineEdgeType });
        for (const dep of inlineExprDeps) {
          edges.push({ source: dep, target: inlineExprId, edgeType: 'data' });
          visit(dep);
        }
      }

      // Synthetic boundary inputs for placeholder kwargs (varName not bound).
      // Label uses the placeholder syntax (`_foo_`) so the original
      // identifier is visible. Edge targets the expression node when the
      // placeholder is actually used in the body, else the reification.
      const localBoundaries = extractBoundaries(e.value);
      if (localBoundaries) {
        for (const [argName, varName] of localBoundaries) {
          if (!bindings.has(varName)) {
            const synId = name + ':' + argName;
            if (!visited.has(synId)) {
              visited.set(synId, {
                id: synId,
                label: varName,
                type: 'input',
                phase: 'parameterized',
                expr: '',
                line: b.line,
                isBoundary: true,
                isTarget: false,
              });
              const edgeTarget = (inlineExprId && placeholdersInBody.has(varName))
                ? inlineExprId
                : name;
              edges.push({ source: synId, target: edgeTarget });
            }
          }
        }
      }
    }

    // Classify each ref by whether it appears wrapped in `lawof(...)`
    // or `draw(...)` inside the RHS — those are the two ops that lift
    // a value into a different "domain" (lawof: value → measure;
    // draw: measure → stochastic value), and surfacing them as nodes
    // makes the structural type story visible in the graph instead of
    // hiding it inside an expression.
    //
    // Skip the synthesis when the binding ITSELF is the wrapper (e.g.
    // `y = draw(...)` already gets the inline-target rendering for
    // its argument). For nested cases — e.g. `s = 2 * draw(m)` or
    // `w = weighted(0.5, lawof(theta))` — we add a synthetic node
    // between the inner ident and the binding, with the appropriate
    // edge types: the boundary edge into a draw / lawof carries the
    // op's edge type so the renderer can colour it specially.
    const wrappedRefs = collectWrappedRefs(b);
    const calls = new Set(e.callDeps || []);
    for (const dep of e.deps) {
      if (inlineExprDeps && inlineExprDeps.has(dep)) continue;

      // Cross-module member references (spec §04): a `load_module` binding
      // used as `mod.field` inside this RHS renders as explicit `mod.field`
      // member node(s) — not a bare edge to the opaque module node, which
      // hides which member is used. The substitution path stays visible via
      // a membership edge `mod → mod.field`. The member's type/kind/phase
      // come from the linked binding `mod$field`. Aliases (`f_b = common.f_b`)
      // never reach here — collectModuleMemberRefs skips a top-level access.
      const memberFields = memberRefs.get(dep);
      if (memberFields && memberFields.size > 0) {
        for (const field of memberFields) {
          const memberId = dep + '.' + field;
          const lb = linkedBindings && linkedBindings.get(dep + '$' + field);
          const lbt = lb && lb.type;
          const memberCallable = lbt === 'functionof' || lbt === 'fn'
            || lbt === 'kernelof' || lbt === 'bijection' || lbt === 'fchain';
          if (!visited.has(memberId)) {
            visited.set(memberId, {
              id: memberId,
              label: memberId,
              type: lb ? lb.type : 'module',
              kind: lb ? reificationKind(lb, linkedBindings) : undefined,
              phase: lb ? lb.phase : undefined,
              expr: lb ? lb.rhs : '',
              line: -1,
              isBoundary: false,
              isTarget: false,
              // Doc-comment from the resolved member, recursively (an alias
              // member inherits its target's doc, even across module hops).
              doc: effectiveDoc(dep + '$' + field, linkedBindings),
              // Drill-in target for the dbltap handler (spec §04 navigation).
              moduleMember: { module: dep, field },
            });
          }
          edges.push({ source: memberId, target: name,
            edgeType: (memberCallable || calls.has(dep)) ? 'call' : 'data' });
          edges.push({ source: dep, target: memberId, edgeType: 'data' });
        }
        visit(dep, false);
        continue;
      }

      const wrapper = wrappedRefs.wrapped.get(dep);
      const usedDirectly = wrappedRefs.direct.has(dep);

      if (wrapper) {
        // Anonymous synthetic node representing the lawof / draw of
        // this dep. id keyed by (binding, op, dep) — one synthetic
        // per (binding, op, dep) combination even if the dep appears
        // wrapped multiple times (typical visualisation simplicity;
        // genuinely-independent multiple draws of the same measure
        // are rare enough not to add per-occurrence ids).
        const synId = name + ':' + wrapper + ':' + dep;
        if (!visited.has(synId)) {
          visited.set(synId, {
            id: synId,
            label: '',
            type: wrapper,
            // For lawof: the synthetic IS the measure produced by
            // reifying the inner value. Set kind='measure' so the
            // renderer's color-resolution picks the lawof blue.
            kind: wrapper === 'lawof' ? 'measure' : undefined,
            // For draw: the synthetic produces a stochastic value;
            // for lawof: a measure (deterministic per spec
            // §sec:lawof). The renderer's phase-color logic uses
            // these for fill colour on value-typed nodes.
            phase: wrapper === 'draw' ? 'stochastic' : 'fixed',
            expr: wrapper + '(' + dep + ')',
            line: b.line,
            isBoundary: false,
            isTarget: false,
          });
        }
        // Edge classification mirrors the binding-level case:
        //   dep → synthetic: 'draw' if the synthetic is a draw (the
        //                    boundary edge into the draw operation);
        //                    'data' for lawof (no special colour).
        //   synthetic → binding: 'data' — the value or measure simply
        //                        flows into the binding's RHS.
        edges.push({
          source: dep, target: synId,
          edgeType: wrapper === 'draw' ? 'draw' : 'data',
        });
        edges.push({ source: synId, target: name, edgeType: 'data' });
        visit(dep, false);
      }

      // Direct edge: dep used outside any draw/lawof wrapper, OR a
      // defensive fallback when the AST walker missed it (shouldn't
      // happen — e.deps is built from the same AST — but no
      // visualization regression if the dep wasn't classified).
      if (usedDirectly || !wrapper) {
        let edgeType;
        if (calls.has(dep))               edgeType = 'call';
        else if (b && b.type === 'draw')  edgeType = 'draw';
        else                              edgeType = 'data';
        edges.push({ source: dep, target: name, edgeType });
        visit(dep, false);
      }
    }
  }

  visit(nodeName, true);

  const reifications = computeReifications(bindings, visited, nodeName);
  applyScopeLocalPhases(visited, reifications, bindings);
  return { nodes: dissolveInternalIntermediates([...visited.values()], edges),
           edges,
           reifications };
}

/**
 * Multi-LHS rewrites (`a, b = call(...)`) and other analyzer
 * internals introduce synthetic `%`-prefixed intermediates into the
 * binding graph — they're real bindings with real derivations, but
 * they aren't user-meaningful and shouldn't show up in the DAG
 * view. Dissolve them: for every `%`-prefixed node, fan its
 * incoming edges directly into its outgoing edges (so a parent that
 * fed into M now feeds into each of M's consumers), then drop M
 * itself.
 *
 * Idempotent and edge-set-only. Synthetic-boundary nodes
 * (placeholder / hole IDs with `:` from reification scopes) keep
 * their bubbles — those are a different category.
 */
function dissolveInternalIntermediates(nodes: any[], edges: any[]) {
  const internal = new Set<string>();
  for (const n of nodes) {
    if (n.id && n.id.charAt(0) === '%') internal.add(n.id);
  }
  if (internal.size === 0) return nodes;
  // Bucket edges by their internal-node endpoint.
  const incoming = new Map<string, any[]>(); // M -> [edges where target === M]
  const outgoing = new Map<string, any[]>(); // M -> [edges where source === M]
  for (const e of edges) {
    if (internal.has(e.target)) {
      if (!incoming.has(e.target)) incoming.set(e.target, []);
      incoming.get(e.target)!.push(e);
    }
    if (internal.has(e.source)) {
      if (!outgoing.has(e.source)) outgoing.set(e.source, []);
      outgoing.get(e.source)!.push(e);
    }
  }
  // Splice in→out edges for each internal node, preserving the
  // outgoing edge's edgeType (the consumer's "kind of use" is what
  // matters at the user-facing endpoint; the incoming edge's type
  // is an engine-internal detail).
  for (const M of internal) {
    const ins  = incoming.get(M) || [];
    const outs = outgoing.get(M) || [];
    for (const ie of ins) {
      for (const oe of outs) {
        edges.push({ source: ie.source, target: oe.target,
                     edgeType: oe.edgeType || ie.edgeType });
      }
    }
  }
  // Drop edges touching internal nodes, and drop the nodes themselves.
  for (let i = edges.length - 1; i >= 0; i--) {
    const e = edges[i];
    if (internal.has(e.source) || internal.has(e.target)) {
      edges.splice(i, 1);
    }
  }
  return nodes.filter((n: any) => !internal.has(n.id));
}

/**
 * Override each in-bubble node's `phase` field with its scope-local
 * phase. A reification cuts the phase chain at its kwargs (those names
 * are *parameters* — values get supplied at call time, so within the
 * body they're `parameterized` rather than `stochastic`).
 *
 * Nesting: process largest-kernel reifications first, smallest last,
 * so the innermost containing scope wins for any node that's in
 * multiple bubbles. Each scope's boundaries union with its enclosing
 * scopes' boundaries so a parameter of an outer kernel reads as
 * `parameterized` from the inner scope's perspective too (the outer
 * kernel hasn't been "called" yet either).
 *
 * Synthetic boundary nodes (placeholders, holes — IDs containing ':')
 * are skipped: their phase is hardcoded at construction and isn't a
 * function of source bindings.
 */
function applyScopeLocalPhases(visited: Map<string, any>, reifications: any[], bindings: any) {
  if (!reifications || reifications.length === 0) return;

  // Per-reification boundary set. extractBoundaries returns Map<argName, varName>;
  // varName is the body-side identifier that becomes a `%local` reference,
  // i.e. the host-binding name the boundary cuts. That's what
  // computePhasesForScope wants.
  const boundariesOf = new Map();
  for (const r of reifications) {
    const b = bindings.get(r.name);
    if (!b) continue;
    const e = eff(b);
    const m = extractBoundaries(e.value);
    boundariesOf.set(r.name, m ? new Set(m.values()) : new Set());
  }

  // Sort reifications so containers come before contained. Larger
  // kernels enclose smaller ones — sort by kernel size descending.
  // (Two unrelated reifications can have any order; their kernels
  // don't overlap so writes don't conflict.)
  const ordered = reifications.slice().sort((a: any, b: any) => b.kernel.length - a.kernel.length);

  // For each reification, accumulate the union of its own boundaries
  // plus all enclosing reifications' boundaries, then compute scope-
  // local phases under that boundary set.
  for (const r of ordered) {
    const ownBoundaries = boundariesOf.get(r.name) || new Set();
    const allBoundaries = new Set(ownBoundaries);
    // Find enclosing reifications: those whose kernel contains r.name.
    for (const outer of reifications) {
      if (outer === r) continue;
      if (!outer.kernel.includes(r.name)) continue;
      const ob = boundariesOf.get(outer.name);
      if (ob) for (const x of ob) allBoundaries.add(x);
    }
    const scopePhases = computePhasesForScope(bindings, allBoundaries);
    for (const id of r.kernel) {
      if (id.indexOf(':') !== -1) continue;
      const node = visited.get(id);
      if (!node) continue;
      const p = scopePhases.get(id);
      if (p != null) node.phase = p;
    }
  }
}

/**
 * For each lawof/functionof/kernelof (or disintegration-result) binding
 * visible in the sub-DAG, compute the set of visible nodes that belong to
 * its kernel. Boundary inputs stop the trace, so kernels respect the
 * reification semantics rather than naive ancestor walks.
 */
function computeReifications(bindings: any, visited: Map<string, any>, rootName: string) {
  const out: any[] = [];
  for (const [name] of visited) {
    if (name.indexOf(':') !== -1) continue; // skip synthetic nodes
    const b = bindings.get(name);
    if (!b) continue;
    if (b.type !== 'lawof' && b.type !== 'functionof' && b.type !== 'kernelof') continue;
    // fn-like reifications get no bubble — the bare hexagon is enough.
    if (isFnLike(bindings, name)) continue;
    // Disintegration results get a bubble only when they're the inspection
    // target. As an ancestor in someone else's trace, they render as a
    // plain node — the user's source structure is the natural view.
    if (b.disintegrateRole && name !== rootName) continue;

    const kernel = kernelNames(bindings, name);
    const visibleKernel = new Set<any>();
    for (const k of kernel) if (visited.has(k)) visibleKernel.add(k);
    // Include synthetic nodes (anon expression target, placeholder boundaries)
    // belonging to this reification.
    for (const [vid] of visited) {
      if (vid.startsWith(name + ':')) visibleKernel.add(vid);
    }
    if (visibleKernel.size < 2) continue;

    const e = eff(b);
    let boundaryVars = new Set();
    const boundaries = extractBoundaries(e.value);
    if (boundaries) boundaryVars = new Set(boundaries.values());

    // If an anonymous expression target exists, that is THE target of the
    // reification (the inline expression being reified). Otherwise, target
    // deps are the simple-identifier first args.
    let targets;
    const syntheticTargetId = name + ':target';
    if (visited.has(syntheticTargetId)) {
      targets = [syntheticTargetId];
    } else {
      targets = (e.deps || []).filter((d: any) => !boundaryVars.has(d) && visited.has(d));
    }

    const kind = reificationKind(b, bindings);
    out.push({ name, type: b.type, kind, kernel: [...visibleKernel], targets });
  }
  return out;
}

// True for a lawof/functionof/kernelof whose kernel members (other than
// itself) are all "constants in scope" — fixed-phase bindings whose value
// is determined at compile time (literals and computations over literals).
// Such a reification has no meaningful runtime scope to enclose; we render
// it as just the hexagon (like `fn`), with no bubble or synthetic children.
function isFnLike(bindings: any, bindingName: string) {
  const b = bindings.get(bindingName);
  if (!b) return false;
  if (b.type !== 'lawof' && b.type !== 'functionof' && b.type !== 'kernelof') return false;
  const kn = kernelNames(bindings, bindingName);
  for (const n of kn) {
    if (n === bindingName) continue;
    const nb = bindings.get(n);
    if (!nb || nb.phase !== 'fixed') return false;
  }
  return true;
}

function kernelNames(bindings: any, bindingName: string) {
  const binding = bindings.get(bindingName);
  if (!binding) return new Set<string>();
  let boundaryVars = new Set<any>();
  if (binding.type === 'functionof' || binding.type === 'kernelof') {
    const boundaries = extractBoundaries(eff(binding).value);
    if (boundaries) boundaryVars = new Set(boundaries.values());
  }
  const visited = new Set<string>();
  function visit(name: string, useEffective: boolean) {
    if (visited.has(name)) return;
    visited.add(name);
    if (boundaryVars.has(name)) return;
    const b = bindings.get(name);
    if (!b) return;
    // Use effective deps for the root (so synthesized disintegration RHS
    // is honoured), literal deps for everything reached transitively.
    const deps = useEffective ? eff(b).deps : (b.deps || []);
    for (const dep of deps) visit(dep, false);
  }
  visit(bindingName, true);
  return visited;
}


/**
 * Find the binding at the given source position.
 *
 * If `col` is provided and the cursor is within the column range of one of the
 * binding's LHS names (e.g. on `prior` in `forward_kernel, prior = ...`),
 * return that specific binding. Otherwise fall back to the first binding
 * defined on that line.
 *
 * @param {Map} bindings
 * @param {number} line - 0-based line number
 * @param {number} [col] - optional 0-based column number
 */
function findBindingAtLine(bindings: any, line: number, col?: number) {
  if (col != null) {
    for (const b of bindings.values()) {
      const nl = b.nameLoc;
      if (nl && nl.start.line === line && nl.end.line === line
          && col >= nl.start.col && col <= nl.end.col) {
        return b;
      }
    }
  }
  for (const b of bindings.values()) {
    if (b.line === line) return b;
  }
  return null;
}

/**
 * Build a DAG covering every binding in the module ("show module"
 * view). Implemented as the union of computeSubDAG()'s output starting
 * from each leaf binding (those not referenced as a dep by any other
 * binding) — ancestor-walks from leaves cover every reachable
 * binding. If somehow nothing qualifies as a leaf (would only happen
 * with a cyclic binding graph, which the analyzer should already
 * reject), we fall back to every binding as a root.
 *
 * Output has the same shape as computeSubDAG so the renderer doesn't
 * need a separate code path. No node carries `isTarget=true` — the
 * full-module view has no distinguished focus.
 *
 * @param {Map} bindings
 * @returns {{ nodes: object[], edges: object[], reifications: object[] }}
 */
function computeFullDAG(bindings: any, opts?: any) {
  if (!bindings || bindings.size === 0) {
    return { nodes: [], edges: [], reifications: [] };
  }

  // Leaves: bindings that no other binding depends on. Their
  // ancestor-walks (computeSubDAG) cover everything reachable.
  const referenced = new Set();
  for (const b of bindings.values()) {
    if (b && b.deps) for (const d of b.deps) referenced.add(d);
  }
  const allNames = [...bindings.keys()];
  const leaves = allNames.filter(n => !referenced.has(n));
  const roots = leaves.length > 0 ? leaves : allNames;

  // Union sub-DAGs by name / edge-key. A binding visited from two
  // different leaves shouldn't appear twice; nor should an edge.
  const nodesByName = new Map();
  const edges: any[] = [];
  const edgeKeys = new Set();
  const reifications: any[] = [];
  const reifNames = new Set();

  for (const root of roots) {
    const sub = computeSubDAG(bindings, root, opts);
    for (const n of sub.nodes) {
      if (nodesByName.has(n.id)) continue;
      // Drop the per-leaf isTarget flag — there's no global target in
      // a module view. Cloning shallowly so the source object isn't
      // mutated if some other view holds onto it.
      nodesByName.set(n.id, Object.assign({}, n, { isTarget: false }));
    }
    for (const e of sub.edges) {
      const key = e.source + '→' + e.target + '|' + (e.edgeType || '');
      if (edgeKeys.has(key)) continue;
      edgeKeys.add(key);
      edges.push(e);
    }
    for (const r of (sub.reifications || [])) {
      if (reifNames.has(r.name)) continue;
      reifNames.add(r.name);
      reifications.push(r);
    }
  }

  // Each per-leaf sub-DAG had applyScopeLocalPhases run against its
  // local `visited`. In the union, a node may have inherited a phase
  // from whichever leaf saw it first — that may or may not match the
  // node's actual scope membership in the merged graph (e.g. a node
  // outside any bubble in leaf-A but inside a bubble in leaf-B).
  // Reset every non-synthetic node's phase to its global value, then
  // re-apply scope overrides against the merged reifications.
  for (const [id, node] of nodesByName) {
    if (id.indexOf(':') !== -1) continue;
    const b = bindings.get(id);
    if (b && b.phase != null) node.phase = b.phase;
  }
  applyScopeLocalPhases(nodesByName, reifications, bindings);

  return { nodes: [...nodesByName.values()], edges, reifications };
}

module.exports = { computeSubDAG, computeFullDAG, findBindingAtLine };
