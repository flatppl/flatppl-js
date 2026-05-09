'use strict';

const { extractBoundaries, collectDeps, isMeasureExpr, computePhasesForScope } = require('./analyzer');

// Resolve a binding's "effective" RHS view. For most bindings this is
// just the literal RHS; for disintegration results that have a
// synthesized Plan, it returns the Plan's expression so the renderer
// can treat them as bona-fide kernelof/lawof bindings.
function eff(b) {
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
function reificationKind(binding, bindings) {
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

function firstPositionalArg(callExpr) {
  if (!callExpr || !callExpr.args) return null;
  for (const arg of callExpr.args) {
    if (arg.type !== 'KeywordArg') return arg;
  }
  return null;
}

// Collect every Placeholder reference (in `_name_` form) under an AST subtree.
function collectPlaceholders(node, out) {
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
function renderExprShort(node, maxLen) {
  if (maxLen == null) maxLen = 24;
  function r(n) {
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
function collectWrappedRefs(binding) {
  const wrapped = new Map();
  const direct = new Set();
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

function classifyRefUsages(node, wrapped, direct, currentWrapper) {
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

function computeSubDAG(bindings, nodeName) {
  const binding = bindings.get(nodeName);
  if (!binding) return { nodes: [], edges: [] };

  // Disintegration with an Unsupported plan → plain dep trace using the
  // analyzer-recorded deps (the user's literal disintegrate(...) call).
  // Synthesized plans expose their RHS via eff().value/deps below, so the
  // root binding renders like any user-written kernelof/lawof.

  const rootValue = eff(binding).value;
  let boundaryVars = new Set();
  let boundaryLabels = new Map(); // varName -> argName

  if (binding.type === 'functionof' || binding.type === 'kernelof') {
    const boundaries = extractBoundaries(rootValue);
    if (boundaries) {
      boundaryVars = new Set(boundaries.values());
      for (const [argName, varName] of boundaries) {
        boundaryLabels.set(varName, argName);
      }
    }
  }

  const visited = new Map();
  const edges = [];

  function visit(name, useEffective) {
    if (visited.has(name)) return;
    const b = bindings.get(name);
    // Effective-RHS overlays apply only at the inspection root. Transitive
    // visits use the literal RHS so a top-level view still reflects the
    // user's source structure — e.g., descending through `prior2` reaches
    // `joint_model`, not the rewriter's synthesized view of it.
    const e = useEffective ? eff(b) : { value: b && b.node && b.node.value, deps: (b && b.deps) || [], callDeps: (b && b.callDeps) || [] };
    const isBoundary = boundaryVars.has(name);

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
      ? b.diagnostics.filter(d => d.severity === 'error')
      : null;

    visited.set(name, {
      id: name,
      label: boundaryLabels.get(name),
      type: b ? b.type : 'unknown',
      kind: reificationKind(b, bindings),
      phase: b ? b.phase : undefined,
      expr: b ? b.rhs : '',
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
        ? require('./types').show(b.inferredType)
        : undefined,
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
    let inlineExprDeps = null;
    let inlineExprId = null;
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
  return { nodes: [...visited.values()], edges, reifications };
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
function applyScopeLocalPhases(visited, reifications, bindings) {
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
  const ordered = reifications.slice().sort((a, b) => b.kernel.length - a.kernel.length);

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
function computeReifications(bindings, visited, rootName) {
  const out = [];
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
    const visibleKernel = new Set();
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
      targets = (e.deps || []).filter(d => !boundaryVars.has(d) && visited.has(d));
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
function isFnLike(bindings, bindingName) {
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

function kernelNames(bindings, bindingName) {
  const binding = bindings.get(bindingName);
  if (!binding) return new Set();
  let boundaryVars = new Set();
  if (binding.type === 'functionof' || binding.type === 'kernelof') {
    const boundaries = extractBoundaries(eff(binding).value);
    if (boundaries) boundaryVars = new Set(boundaries.values());
  }
  const visited = new Set();
  function visit(name, useEffective) {
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
function findBindingAtLine(bindings, line, col) {
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
function computeFullDAG(bindings) {
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
  const edges = [];
  const edgeKeys = new Set();
  const reifications = [];
  const reifNames = new Set();

  for (const root of roots) {
    const sub = computeSubDAG(bindings, root);
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
