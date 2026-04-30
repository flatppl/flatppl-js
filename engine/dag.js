'use strict';

const { extractBoundaries, countHoles, collectDeps } = require('./analyzer');

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
    return computeDisintegrateSubDAG(bindings, binding);
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
    });

    if (isBoundary || !b) return;

    const calls = new Set(b.callDeps || []);
    for (const dep of b.deps) {
      edges.push({ source: dep, target: name, edgeType: calls.has(dep) ? 'call' : 'data' });
      visit(dep);
    }
  }

  visit(nodeName);

  // Synthetic input nodes for functionof/lawof placeholder boundaries
  if (boundaries) {
    for (const [argName, varName] of boundaries) {
      if (!bindings.has(varName)) {
        const synId = nodeName + ':' + argName;
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
        edges.push({ source: synId, target: nodeName });
      }
    }
  }

  // Synthetic input nodes for fn holes
  if (binding.type === 'fn') {
    const holeCount = countHoles(binding.node.value);
    for (let i = 1; i <= holeCount; i++) {
      const synId = nodeName + ':_' + i;
      visited.set(synId, {
        id: synId,
        label: '_',
        type: 'input',
        phase: 'parameterized',
        expr: '',
        line: binding.line,
        isBoundary: true,
        isTarget: false,
      });
      edges.push({ source: synId, target: nodeName });
    }
  }

  return { nodes: [...visited.values()], edges };
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
