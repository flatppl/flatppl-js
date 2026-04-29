'use strict';

const { extractBoundaries, countHoles } = require('./analyzer');

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
  const targetFields = isKernel
    ? role.selectorFields.filter(f => role.jointFields.has(f))
    : [...role.jointFields.keys()].filter(f => !selected.has(f));
  const boundaryFields = isKernel
    ? [...role.jointFields.keys()].filter(f => !selected.has(f))
    : [];

  // Build boundary sets: (field-derived) + (inherited from joint)
  const boundaryVars = new Set();
  const boundaryLabels = new Map(); // varName -> argName

  for (const f of boundaryFields) {
    const v = role.jointFields.get(f);
    boundaryVars.add(v);
    boundaryLabels.set(v, f);
  }
  for (const [argName, varName] of role.inheritedBoundaries) {
    boundaryVars.add(varName);
    if (!boundaryLabels.has(varName)) boundaryLabels.set(varName, argName);
  }

  const visited = new Map();
  const edges = [];

  // The disintegration result node itself.
  visited.set(binding.name, {
    id: binding.name,
    label: binding.name,
    type: binding.type,
    expr: binding.rhs,
    line: binding.line,
    isBoundary: false,
    isTarget: true,
  });

  function visit(name) {
    if (visited.has(name)) return;
    const b = bindings.get(name);
    const isBoundary = boundaryVars.has(name);

    visited.set(name, {
      id: name,
      label: boundaryLabels.get(name),
      type: b ? b.type : 'unknown',
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

  // Trace targets back; their ancestors stop at boundaries.
  for (const f of targetFields) {
    const v = role.jointFields.get(f);
    if (!v) continue;
    edges.push({ source: v, target: binding.name, edgeType: 'data' });
    visit(v);
  }

  // Boundaries (kernel inputs) — include them, edge to result, but don't trace deeper.
  for (const v of boundaryVars) {
    if (!visited.has(v)) visit(v);
    edges.push({ source: v, target: binding.name, edgeType: 'data' });
  }

  // Synthetic input nodes for inherited boundaries that aren't bound module-locally
  // (e.g., when joint had a placeholder-style boundary).
  for (const [argName, varName] of role.inheritedBoundaries) {
    if (!bindings.has(varName) && !visited.has(varName)) {
      const synId = binding.name + ':' + argName;
      visited.set(synId, {
        id: synId,
        label: argName,
        type: 'input',
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
 * Find the binding whose definition is on the given line number.
 */
function findBindingAtLine(bindings, lineNumber) {
  for (const b of bindings.values()) {
    if (b.line === lineNumber) return b;
  }
  return null;
}

module.exports = { computeSubDAG, findBindingAtLine };
