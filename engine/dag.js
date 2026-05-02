'use strict';

const { extractBoundaries, countHoles, collectDeps, isMeasureExpr } = require('./analyzer');

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
    case 'functionof': {
      const firstArg = firstPositionalArg(eff(binding).value);
      return isMeasureExpr(firstArg, bindings) ? 'kernel' : 'function';
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
        visited.set(inlineExprId, {
          id: inlineExprId,
          // Empty label — anonymous bridge node, identifies itself via
          // hover (rendered first-arg expression).
          label: '',
          type: 'call',
          phase: b.phase,
          expr: renderExprShort(firstArg, 200),
          line: b.line,
          isBoundary: false,
          isTarget: false,
        });
        edges.push({ source: inlineExprId, target: name, edgeType: 'data' });
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

    const calls = new Set(e.callDeps || []);
    for (const dep of e.deps) {
      if (inlineExprDeps && inlineExprDeps.has(dep)) continue;
      edges.push({ source: dep, target: name, edgeType: calls.has(dep) ? 'call' : 'data' });
      visit(dep, false);
    }
  }

  visit(nodeName, true);

  const reifications = computeReifications(bindings, visited, nodeName);
  return { nodes: [...visited.values()], edges, reifications };
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

module.exports = { computeSubDAG, findBindingAtLine };
