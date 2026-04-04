'use strict';

const { extractBoundaries, countHoles } = require('./analyzer');

/**
 * Compute the ancestor sub-DAG of a node.
 * For lawof/functionof, boundary inputs stop the backwards trace.
 *
 * @param {Map} bindings - from analyzer: Map<name, BindingInfo>
 * @param {string} nodeName - the target node name
 * @returns {{ nodes: object[], edges: object[] }}
 */
function computeSubDAG(bindings, nodeName) {
  const binding = bindings.get(nodeName);
  if (!binding) return { nodes: [], edges: [] };

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
 * Find the binding whose definition is on the given line number.
 */
function findBindingAtLine(bindings, lineNumber) {
  for (const b of bindings.values()) {
    if (b.line === lineNumber) return b;
  }
  return null;
}

module.exports = { computeSubDAG, findBindingAtLine };
