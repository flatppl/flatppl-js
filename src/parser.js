'use strict';

/**
 * Split a string by commas at parenthesis/bracket depth 0.
 */
function splitTopLevel(str) {
  const parts = [];
  let depth = 0;
  let current = '';
  for (const ch of str) {
    if (ch === '(' || ch === '[') depth++;
    else if (ch === ')' || ch === ']') depth--;
    if (ch === ',' && depth === 0) {
      parts.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  if (current.trim()) parts.push(current.trim());
  return parts;
}

/**
 * For lawof/functionof expressions, extract boundary inputs.
 * E.g. "lawof(record(obs = obs), theta1 = theta1, theta2 = theta2)"
 *   -> { boundaries: { theta1: 'theta1', theta2: 'theta2' } }
 */
function parseBoundaryInputs(rhs) {
  const match = rhs.match(/^(lawof|functionof)\s*\((.+)\)\s*$/s);
  if (!match) return null;

  const inner = match[2];
  const parts = splitTopLevel(inner);

  const boundaries = {};
  for (let i = 1; i < parts.length; i++) {
    const kwMatch = parts[i].match(/^(\w+)\s*=\s*(\w+)$/);
    if (kwMatch) {
      boundaries[kwMatch[1]] = kwMatch[2];
    }
  }
  return { type: match[1], boundaries };
}

/**
 * Parse FlatPPL source text into variable bindings with dependencies.
 */
function parseFlatPPL(text) {
  const lines = text.split('\n');
  const bindings = [];
  const allNames = new Set();

  // First pass: collect variable names and classify nodes
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].replace(/#.*$/, '').trim();
    if (!line) continue;

    const match = line.match(/^([\w]+(?:\s*,\s*[\w]+)*)\s*=\s*(.+)$/);
    if (!match) continue;

    const lhsStr = match[1];
    const rhs = match[2].trim();
    const names = lhsStr.split(/\s*,\s*/).map(n => n.trim());

    let type = 'deterministic';
    if (/^draw\s*\(/.test(rhs)) type = 'stochastic';
    else if (/^elementof\s*\(/.test(rhs)) type = 'input';
    else if (/^lawof\s*\(/.test(rhs)) type = 'lawof';
    else if (/^functionof\s*\(/.test(rhs)) type = 'functionof';
    else if (/^fn\s*\(/.test(rhs)) type = 'fn';
    else if (/^likelihoodof\s*\(/.test(rhs)) type = 'likelihood';
    else if (/^bayesupdate\s*\(/.test(rhs)) type = 'bayesupdate';
    else if (/^load_module\s*\(/.test(rhs)) type = 'module';
    else if (/^load_table\s*\(/.test(rhs)) type = 'table';
    else if (/^\[/.test(rhs) || /^[0-9.eE+\-]+$/.test(rhs) || /^"/.test(rhs)) type = 'literal';

    for (const name of names) allNames.add(name);
    bindings.push({ names, line: i, rhs, type });
  }

  // Second pass: resolve dependencies via identifier matching
  for (const binding of bindings) {
    const deps = new Set();
    const callDeps = new Set();
    const rhs = binding.rhs;
    const re = /\b([a-zA-Z_]\w*)\b/g;
    let m;
    while ((m = re.exec(rhs)) !== null) {
      const ident = m[1];
      // Skip identifiers in keyword-argument position (followed by = but not ==)
      const after = rhs.slice(m.index + m[0].length);
      if (/^\s*=(?!=)/.test(after)) continue;
      if (!allNames.has(ident) || binding.names.includes(ident)) continue;
      deps.add(ident);
      // Identifier immediately followed by ( is in callable position
      if (/^\s*\(/.test(after)) callDeps.add(ident);
    }
    binding.deps = [...deps];
    binding.callDeps = [...callDeps];
  }

  return bindings;
}

/**
 * Create a lookup map from variable name to binding info.
 */
function createBindingMap(bindings) {
  const map = new Map();
  for (const b of bindings) {
    for (const name of b.names) {
      map.set(name, { name, line: b.line, rhs: b.rhs, type: b.type, deps: b.deps, callDeps: b.callDeps });
    }
  }
  return map;
}

/**
 * Compute the ancestor sub-DAG of a node.
 * For lawof/functionof, boundary inputs stop the backwards trace.
 */
function computeSubDAG(bindingMap, nodeName) {
  const binding = bindingMap.get(nodeName);
  if (!binding) return { nodes: [], edges: [] };

  let boundaryVars = new Set();
  let boundaryLabels = new Map(); // varName -> argName
  let parsed = null;
  if (binding.type === 'lawof' || binding.type === 'functionof') {
    parsed = parseBoundaryInputs(binding.rhs);
    if (parsed) {
      boundaryVars = new Set(Object.values(parsed.boundaries));
      for (const [argName, varName] of Object.entries(parsed.boundaries)) {
        boundaryLabels.set(varName, argName);
      }
    }
  }

  const visited = new Map();
  const edges = [];

  function visit(name) {
    if (visited.has(name)) return;
    const b = bindingMap.get(name);
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
  if (parsed) {
    for (const [argName, varName] of Object.entries(parsed.boundaries)) {
      if (!bindingMap.has(varName)) {
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
    const fnMatch = binding.rhs.match(/^fn\s*\((.+)\)\s*$/s);
    if (fnMatch) {
      const inner = fnMatch[1];
      const holeRe = /\b_\b/g;
      let count = 0;
      let m;
      while ((m = holeRe.exec(inner)) !== null) {
        const after = inner.slice(m.index + 1);
        if (/^\s*=(?!=)/.test(after)) continue;
        count++;
        const synId = nodeName + ':_' + count;
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
  }

  return { nodes: [...visited.values()], edges };
}

/**
 * Find the binding whose definition is on the given line number.
 */
function findBindingAtLine(bindings, lineNumber) {
  return bindings.find(b => b.line === lineNumber) || null;
}

module.exports = { parseFlatPPL, createBindingMap, computeSubDAG, findBindingAtLine };
