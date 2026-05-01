'use strict';

const { extractBoundaries, countHoles, collectDeps } = require('./analyzer');

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
 * For lawof/functionof, boundary inputs stop the backwards trace.
 * For nodes tagged with disintegrateRole, synthesize the equivalent
 * lawof-shaped sub-DAG (target/boundary inputs derived from the joint
 * measure's record fields and the disintegration selector).
 *
 * @param {Map} bindings - from analyzer: Map<name, BindingInfo>
 * @param {string} nodeName - the target node name
 * @returns {{ nodes: object[], edges: object[] }}
 */
function computeSubDAG(bindings, nodeName) {
  const binding = bindings.get(nodeName);
  if (!binding) return { nodes: [], edges: [] };

  // Special path: disintegration result.
  if (binding.disintegrateRole) {
    const result = computeDisintegrateSubDAG(bindings, binding);
    if (!result.reifications) result.reifications = [];
    return result;
  }

  let boundaryVars = new Set();
  let boundaryLabels = new Map(); // varName -> argName
  let boundaries = null;

  if (binding.type === 'lawof' || binding.type === 'functionof') {
    boundaries = extractBoundaries(binding.node.value);
    if (boundaries) {
      boundaryVars = new Set(boundaries.values());
      for (const [argName, varName] of boundaries) {
        boundaryLabels.set(varName, argName);
      }
    }
  }

  const visited = new Map();
  const edges = [];

  function visit(name) {
    if (visited.has(name)) return;
    const b = bindings.get(name);
    const isBoundary = boundaryVars.has(name);

    visited.set(name, {
      id: name,
      label: boundaryLabels.get(name),
      type: b ? b.type : 'unknown',
      phase: b ? b.phase : undefined,
      expr: b ? b.rhs : '',
      line: b ? b.line : -1,
      isBoundary,
      isTarget: name === nodeName,
      closedMeasure: b && b.type === 'lawof' && isClosedMeasure(bindings, name),
    });

    if (isBoundary || !b) return;

    // For lawof/functionof, optionally synthesize:
    //   - an anonymous "expression target" node, when the first positional
    //     arg is a compound expression, so the bubble has a clear value-
    //     being-reified that external nodes can tether to;
    //   - boundary input nodes for placeholder kwargs (varName not bound).
    // Skipped for fn-like bindings and for disintegration results
    // (the latter are decompositions of an existing joint, not reifications
    // of a new scope — they render as plain nodes connected from the joint).
    let inlineExprDeps = null;
    let inlineExprId = null;
    if ((b.type === 'lawof' || b.type === 'functionof')
        && !b.disintegrateRole
        && !isFnLike(bindings, name)
        && b.node && b.node.value) {
      const firstArg = firstPositionalArg(b.node.value);
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
      const localBoundaries = extractBoundaries(b.node.value);
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

    const calls = new Set(b.callDeps || []);
    for (const dep of b.deps) {
      if (inlineExprDeps && inlineExprDeps.has(dep)) continue;
      edges.push({ source: dep, target: name, edgeType: calls.has(dep) ? 'call' : 'data' });
      visit(dep);
    }
  }

  visit(nodeName);

  const reifications = computeReifications(bindings, visited);
  return { nodes: [...visited.values()], edges, reifications };
}

/**
 * For each lawof/functionof (or disintegration-result) binding visible in
 * the sub-DAG, compute the set of visible nodes that belong to its kernel.
 * Boundary inputs stop the trace, so kernels respect lawof/functionof
 * semantics rather than naive ancestor walks.
 */
function computeReifications(bindings, visited) {
  const out = [];
  for (const [name] of visited) {
    if (name.indexOf(':') !== -1) continue; // skip synthetic nodes
    const b = bindings.get(name);
    if (!b) continue;
    if (b.type !== 'lawof' && b.type !== 'functionof') continue;
    // Disintegration results are decompositions of a joint measure, not
    // reifications of a new scope. Render them as plain nodes connected
    // from the joint — no bubble.
    if (b.disintegrateRole) continue;
    // fn-like reifications get no bubble — the bare hexagon is enough.
    if (isFnLike(bindings, name)) continue;

    const kernel = kernelNames(bindings, name);
    const visibleKernel = new Set();
    for (const k of kernel) if (visited.has(k)) visibleKernel.add(k);
    // Include synthetic nodes (anon expression target, placeholder boundaries)
    // belonging to this reification.
    for (const [vid] of visited) {
      if (vid.startsWith(name + ':')) visibleKernel.add(vid);
    }
    if (visibleKernel.size < 2) continue;

    let boundaryVars = new Set();
    const boundaries = extractBoundaries(b.node.value);
    if (boundaries) boundaryVars = new Set(boundaries.values());

    // If an anonymous expression target exists, that is THE target of the
    // reification (the inline expression being reified). Otherwise, target
    // deps are the simple-identifier first args.
    let targets;
    const syntheticTargetId = name + ':target';
    if (visited.has(syntheticTargetId)) {
      targets = [syntheticTargetId];
    } else {
      targets = (b.deps || []).filter(d => !boundaryVars.has(d) && visited.has(d));
    }

    out.push({ name, type: b.type, kernel: [...visibleKernel], targets });
  }
  return out;
}

/**
 * A lawof binding is a closed measure if it has no free inputs:
 * no explicit boundary args AND no elementof/external ancestor.
 * (A Markov kernel has at least one free input — explicit or implicit.)
 *
 * For disintegration results: the kernel is always parameterized (it takes
 * the unselected fields as inputs), the prior is closed iff the joint had
 * no inherited boundaries.
 */
function isClosedMeasure(bindings, bindingName) {
  const binding = bindings.get(bindingName);
  if (!binding || binding.type !== 'lawof') return false;
  if (binding.disintegrateRole) {
    if (binding.disintegrateRole.kind === 'kernel') return false;
    const inh = binding.disintegrateRole.inheritedBoundaries;
    return !inh || inh.size === 0;
  }
  const boundaries = extractBoundaries(binding.node.value);
  if (boundaries && boundaries.size > 0) return false;
  const seen = new Set();
  function hasInput(name) {
    if (seen.has(name)) return false;
    seen.add(name);
    const b = bindings.get(name);
    if (!b) return false;
    if (b.type === 'input') return true;
    for (const dep of b.deps) if (hasInput(dep)) return true;
    return false;
  }
  return !hasInput(bindingName);
}

// True for a lawof/functionof whose kernel members (other than itself) are
// all "constants in scope" — fixed-phase bindings whose value is determined
// at compile time (literals and computations over literals). Such a
// reification has no meaningful runtime scope to enclose; we render it as
// just the hexagon (like `fn`), with no bubble or synthetic children.
function isFnLike(bindings, bindingName) {
  const b = bindings.get(bindingName);
  if (!b || (b.type !== 'lawof' && b.type !== 'functionof')) return false;
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
  if (binding.type === 'lawof' || binding.type === 'functionof') {
    const boundaries = extractBoundaries(binding.node.value);
    if (boundaries) boundaryVars = new Set(boundaries.values());
  }
  const visited = new Set();
  function visit(name) {
    if (visited.has(name)) return;
    visited.add(name);
    if (boundaryVars.has(name)) return;
    const b = bindings.get(name);
    if (!b) return;
    for (const dep of b.deps) visit(dep);
  }
  visit(bindingName);
  return visited;
}

/**
 * Synthesize the sub-DAG for a disintegration result.
 *
 * For `kernel, prior = disintegrate(selector, joint)` where
 * `joint = lawof(record(field1 = node1, ..., fieldN = nodeN), [argA = nodeA, ...])`:
 *
 * - Kernel: target = selected fields' nodes; boundaries = unselected fields'
 *   nodes (with field names as labels) + inherited joint boundaries.
 * - Prior: target = unselected fields' nodes; boundaries = inherited joint
 *   boundaries only.
 */
function computeDisintegrateSubDAG(bindings, binding) {
  const role = binding.disintegrateRole;
  const isKernel = role.kind === 'kernel';
  const selected = new Set(role.selectorFields);

  const visited = new Map();
  const edges = [];

  // The disintegration result node itself.
  visited.set(binding.name, {
    id: binding.name,
    label: binding.name,
    type: binding.type,
    phase: binding.phase,
    expr: binding.rhs,
    line: binding.line,
    isBoundary: false,
    isTarget: true,
  });

  if (role.jointKind === 'lawof_record') {
    return computeLawofRecordDisintegration(bindings, binding, role, isKernel, selected, visited, edges);
  }
  if (role.jointKind === 'joint') {
    return computeJointDisintegration(bindings, binding, role, isKernel, selected, visited, edges);
  }
  if (role.jointKind === 'jointchain') {
    return computeJointchainDisintegration(bindings, binding, role, isKernel, selected, visited, edges);
  }

  // Unknown joint kind — return just the result node alone.
  return { nodes: [...visited.values()], edges };
}

/**
 * Tier 1 — joint constructed via `lawof(record(field = node, ...), [bnds...])`.
 * Each field is bound to a module-level identifier; unselected fields become
 * boundary inputs of the kernel (and are absent from the prior).
 */
function computeLawofRecordDisintegration(bindings, binding, role, isKernel, selected, visited, edges) {
  const targetFields = isKernel
    ? role.selectorFields.filter(f => role.jointFields.has(f))
    : [...role.jointFields.keys()].filter(f => !selected.has(f));
  const boundaryFields = isKernel
    ? [...role.jointFields.keys()].filter(f => !selected.has(f))
    : [];

  const boundaryVars = new Set();
  const boundaryLabels = new Map();

  for (const f of boundaryFields) {
    const node = role.jointFields.get(f); // Identifier AST node
    if (node && node.type === 'Identifier') {
      boundaryVars.add(node.name);
      boundaryLabels.set(node.name, f);
    }
  }
  for (const [argName, varName] of role.inheritedBoundaries) {
    boundaryVars.add(varName);
    if (!boundaryLabels.has(varName)) boundaryLabels.set(varName, argName);
  }

  function visit(name) {
    if (visited.has(name)) return;
    const b = bindings.get(name);
    const isBoundary = boundaryVars.has(name);

    visited.set(name, {
      id: name,
      label: boundaryLabels.get(name),
      type: b ? b.type : 'unknown',
      phase: b ? b.phase : undefined,
      expr: b ? b.rhs : '',
      line: b ? b.line : -1,
      isBoundary,
      isTarget: false,
    });

    if (isBoundary || !b) return;

    const calls = new Set(b.callDeps || []);
    for (const dep of b.deps) {
      edges.push({ source: dep, target: name, edgeType: calls.has(dep) ? 'call' : 'data' });
      visit(dep);
    }
  }

  // Trace target fields; ancestors stop at boundaries.
  for (const f of targetFields) {
    const node = role.jointFields.get(f);
    if (!node || node.type !== 'Identifier') continue;
    const v = node.name;
    edges.push({ source: v, target: binding.name, edgeType: 'data' });
    visit(v);
  }

  for (const v of boundaryVars) {
    if (!visited.has(v)) visit(v);
    edges.push({ source: v, target: binding.name, edgeType: 'data' });
  }

  for (const [argName, varName] of role.inheritedBoundaries) {
    if (!bindings.has(varName) && !visited.has(varName)) {
      const synId = binding.name + ':' + argName;
      visited.set(synId, {
        id: synId,
        label: argName,
        type: 'input',
        phase: 'parameterized',
        expr: '',
        line: binding.line,
        isBoundary: true,
        isTarget: false,
      });
      edges.push({ source: synId, target: binding.name, edgeType: 'data' });
    }
  }

  return { nodes: [...visited.values()], edges };
}

/**
 * Tier 2 — joint constructed via `joint(name = M, ...)` keyword form.
 * Components are independent: there are no cross-boundaries between the
 * selected and unselected fields. Each field's contribution to the kernel
 * (or prior) sub-DAG is the set of module-level identifiers referenced by
 * its measure expression.
 */
function computeJointDisintegration(bindings, binding, role, isKernel, selected, visited, edges) {
  const allFields = [...role.jointFields.keys()];
  const myFields = isKernel
    ? allFields.filter(f => selected.has(f))
    : allFields.filter(f => !selected.has(f));

  const definedNames = new Set(bindings.keys());
  const collectedDeps = new Set();
  const collectedCallDeps = new Set();

  for (const f of myFields) {
    const expr = role.jointFields.get(f);
    if (!expr) continue;
    const { deps, callDeps } = collectDeps(expr, definedNames);
    for (const d of deps) collectedDeps.add(d);
    for (const d of callDeps) collectedCallDeps.add(d);
  }

  function visit(name) {
    if (visited.has(name)) return;
    const b = bindings.get(name);
    visited.set(name, {
      id: name,
      type: b ? b.type : 'unknown',
      phase: b ? b.phase : undefined,
      expr: b ? b.rhs : '',
      line: b ? b.line : -1,
      isBoundary: false,
      isTarget: false,
    });
    if (!b) return;
    const calls = new Set(b.callDeps || []);
    for (const dep of b.deps) {
      edges.push({ source: dep, target: name, edgeType: calls.has(dep) ? 'call' : 'data' });
      visit(dep);
    }
  }

  for (const dep of collectedDeps) {
    edges.push({
      source: dep,
      target: binding.name,
      edgeType: collectedCallDeps.has(dep) ? 'call' : 'data',
    });
    visit(dep);
  }

  return { nodes: [...visited.values()], edges };
}

/**
 * Tier 2 — joint constructed via `jointchain(name1 = M, name2 = K, ...)`.
 * The chain order is significant: later kernels may depend on earlier
 * variates. We don't fully model that variate graph; we approximate as
 * follows:
 *  - Selected/unselected component expressions contribute their module-level
 *    deps as ancestors of the kernel/prior (like the joint case).
 *  - Any unselected field appearing BEFORE a selected field in the chain
 *    becomes a synthetic boundary input on the kernel (labelled with the
 *    field name). This is exact for the "trailing-suffix selection" case
 *    (e.g., `disintegrate("c", jointchain(a, b, c))` → kernel needs a, b).
 *  - For prior, we treat unselected fields like an independent joint —
 *    the actual disintegration may be intractable in non-suffix selection
 *    cases, but the visualization is still informative.
 */
function computeJointchainDisintegration(bindings, binding, role, isKernel, selected, visited, edges) {
  const allFieldsOrdered = [...role.jointFields.keys()];
  const myFields = isKernel
    ? allFieldsOrdered.filter(f => selected.has(f))
    : allFieldsOrdered.filter(f => !selected.has(f));

  const definedNames = new Set(bindings.keys());
  const collectedDeps = new Set();
  const collectedCallDeps = new Set();

  for (const f of myFields) {
    const expr = role.jointFields.get(f);
    if (!expr) continue;
    const { deps, callDeps } = collectDeps(expr, definedNames);
    for (const d of deps) collectedDeps.add(d);
    for (const d of callDeps) collectedCallDeps.add(d);
  }

  function visit(name) {
    if (visited.has(name)) return;
    const b = bindings.get(name);
    visited.set(name, {
      id: name,
      type: b ? b.type : 'unknown',
      phase: b ? b.phase : undefined,
      expr: b ? b.rhs : '',
      line: b ? b.line : -1,
      isBoundary: false,
      isTarget: false,
    });
    if (!b) return;
    const calls = new Set(b.callDeps || []);
    for (const dep of b.deps) {
      edges.push({ source: dep, target: name, edgeType: calls.has(dep) ? 'call' : 'data' });
      visit(dep);
    }
  }

  for (const dep of collectedDeps) {
    edges.push({
      source: dep,
      target: binding.name,
      edgeType: collectedCallDeps.has(dep) ? 'call' : 'data',
    });
    visit(dep);
  }

  // For the kernel: add synthetic boundary inputs for any unselected fields
  // that appear before the first selected field in the chain.
  if (isKernel) {
    const firstSelectedIdx = allFieldsOrdered.findIndex(f => selected.has(f));
    if (firstSelectedIdx > 0) {
      for (let i = 0; i < firstSelectedIdx; i++) {
        const f = allFieldsOrdered[i];
        if (selected.has(f)) continue;
        const synId = binding.name + ':' + f;
        if (!visited.has(synId)) {
          visited.set(synId, {
            id: synId,
            label: f,
            type: 'input',
            phase: 'parameterized',
            expr: '',
            line: binding.line,
            isBoundary: true,
            isTarget: false,
          });
          edges.push({ source: synId, target: binding.name, edgeType: 'data' });
        }
      }
    }
  }

  return { nodes: [...visited.values()], edges };
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
