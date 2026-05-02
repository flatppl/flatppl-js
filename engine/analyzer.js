'use strict';

const { isKnownName, MEASURE_PRODUCING } = require('./builtins');
// Lazy require to avoid a circular load (disintegrate requires analyzer).
let _disintegratePlan = null;
function disintegratePlan(...args) {
  if (!_disintegratePlan) _disintegratePlan = require('./disintegrate').disintegratePlan;
  return _disintegratePlan(...args);
}

/**
 * Determine whether an expression produces a measure (probability measure
 * or general measure) — as opposed to a value, kernel, or function.
 *
 * Measures come from: `lawof(...)`, distribution constructors (Normal, ...),
 * and measure-algebra ops that combine measures (iid, joint, chain, ...).
 *
 * NB: kernels (kernelof, functionof on a measure) are NOT measures — they
 * are functions returning measures.
 *
 * @param {object} node - AST expression
 * @param {Map} bindings - bindings map (for Identifier resolution)
 * @param {Set} [seen] - cycle guard
 */
function isMeasureExpr(node, bindings, seen) {
  if (!node) return false;
  if (!seen) seen = new Set();
  switch (node.type) {
    case 'Identifier': {
      const name = node.name;
      if (seen.has(name)) return false;
      seen.add(name);
      const b = bindings.get(name);
      if (!b) return false;
      if (b.type === 'lawof') return true;
      // 'call'-type bindings can be measure-typed (e.g., theta_dist = Normal(...)).
      if (b.type === 'call' && b.node && b.node.value) {
        return isMeasureExpr(b.node.value, bindings, seen);
      }
      return false;
    }
    case 'CallExpr': {
      if (!node.callee || node.callee.type !== 'Identifier') return false;
      const name = node.callee.name;
      if (name === 'lawof') return true;
      if (MEASURE_PRODUCING.has(name)) return true;
      return false;
    }
    default:
      return false;
  }
}

/**
 * Classify a statement by examining the RHS expression.
 */
function classifyStatement(valueNode) {
  if (!valueNode) return 'call';

  if (valueNode.type === 'CallExpr' && valueNode.callee.type === 'Identifier') {
    const name = valueNode.callee.name;
    switch (name) {
      case 'draw': return 'draw';
      case 'elementof': return 'input';
      case 'external': return 'input';
      case 'lawof': return 'lawof';
      case 'functionof': return 'functionof';
      case 'kernelof': return 'kernelof';
      case 'fn': return 'fn';
      case 'likelihoodof': return 'likelihood';
      case 'bayesupdate': return 'bayesupdate';
      case 'load_module': return 'module';
      case 'standard_module': return 'module';
      case 'load_data': return 'data';
    }
  }

  if (valueNode.type === 'ArrayLiteral' || valueNode.type === 'NumberLiteral'
      || valueNode.type === 'StringLiteral' || valueNode.type === 'TupleLiteral'
      || valueNode.type === 'BoolLiteral') {
    return 'literal';
  }

  return 'call';
}

/**
 * Validate argument structure of special operations.
 * Returns an array of diagnostics.
 */
function validateSpecialOperation(valueNode) {
  if (!valueNode || valueNode.type !== 'CallExpr') return [];
  if (!valueNode.callee || valueNode.callee.type !== 'Identifier') return [];

  const name = valueNode.callee.name;
  const args = valueNode.args;
  const diags = [];

  switch (name) {
    case 'functionof':
    case 'kernelof': {
      // First arg must be a positional expression, rest must be keyword args
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `${name}() requires at least one argument`, loc: valueNode.loc });
        break;
      }
      if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `First argument of ${name}() must be an expression, not a keyword argument`, loc: args[0].loc });
      }
      for (let i = 1; i < args.length; i++) {
        if (args[i].type !== 'KeywordArg') {
          diags.push({ severity: 'error', message: `Arguments after the first in ${name}() must be keyword boundary inputs (name = node)`, loc: args[i].loc });
        }
      }
      break;
    }
    case 'lawof': {
      // Unary: a single positional expression. Boundary keyword args were
      // moved to kernelof() — flag them with a migration hint.
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `lawof() requires exactly one argument`, loc: valueNode.loc });
        break;
      }
      if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `lawof() argument must be an expression, not a keyword argument`, loc: args[0].loc });
      }
      for (let i = 1; i < args.length; i++) {
        diags.push({
          severity: 'error',
          message: `lawof() takes a single argument; for a Markov kernel use kernelof(expr, ...keyword boundaries)`,
          loc: args[i].loc,
        });
      }
      break;
    }
    case 'fn': {
      // Single positional expression
      if (args.length !== 1) {
        diags.push({ severity: 'error', message: `fn() requires exactly one expression argument`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `fn() argument must be an expression, not a keyword argument`, loc: args[0].loc });
      }
      break;
    }
    case 'draw':
    case 'elementof':
    case 'external':
    case 'valueset': {
      // Single positional expression
      if (args.length !== 1) {
        diags.push({ severity: 'error', message: `${name}() requires exactly one argument`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `${name}() argument must be an expression, not a keyword argument`, loc: args[0].loc });
      }
      break;
    }
    case 'load_module': {
      // First arg must be a string, rest are optional keyword args
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `load_module() requires a file path argument`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `First argument of load_module() must be a file path`, loc: args[0].loc });
      }
      for (let i = 1; i < args.length; i++) {
        if (args[i].type !== 'KeywordArg') {
          diags.push({ severity: 'error', message: `Arguments after the file path in load_module() must be keyword substitutions (name = value)`, loc: args[i].loc });
        }
      }
      break;
    }
    case 'standard_module': {
      // Two positional arguments: module name and version string
      if (args.length !== 2) {
        diags.push({ severity: 'error', message: `standard_module() requires exactly two arguments (name, version)`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `standard_module() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }
    case 'load_data': {
      // source (positional or keyword) + valueset (keyword)
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `load_data() requires source and valueset arguments`, loc: valueNode.loc });
      }
      break;
    }
  }

  return diags;
}

/**
 * Walk an expression tree and collect referenced identifiers.
 * Skips keyword argument names (the 'name' in KeywordArg is not a reference).
 * Returns { deps: Set<string>, callDeps: Set<string> }.
 */
function collectDeps(node, definedNames) {
  const deps = new Set();
  const callDeps = new Set();

  function walk(node, isCallee) {
    if (!node) return;

    switch (node.type) {
      case 'Identifier':
        if (definedNames.has(node.name)) {
          deps.add(node.name);
          if (isCallee) callDeps.add(node.name);
        }
        break;
      case 'BinaryExpr':
        walk(node.left, false);
        walk(node.right, false);
        break;
      case 'UnaryExpr':
        walk(node.operand, false);
        break;
      case 'CallExpr':
        walk(node.callee, true);
        for (const arg of node.args) walk(arg, false);
        break;
      case 'IndexExpr':
        walk(node.object, false);
        for (const idx of node.indices) walk(idx, false);
        break;
      case 'FieldAccess':
        walk(node.object, false);
        break;
      case 'ArrayLiteral':
      case 'TupleLiteral':
        for (const el of node.elements) walk(el, false);
        break;
      case 'KeywordArg':
        // Only walk the value, not the keyword name
        walk(node.value, false);
        break;
      // Leaf nodes: NumberLiteral, StringLiteral, BoolLiteral,
      // ConstantRef, SetRef, Placeholder, Hole, SliceAll — no deps
    }
  }

  walk(node, false);
  return { deps, callDeps };
}

/**
 * For functionof/kernelof calls, extract boundary inputs from keyword args.
 * Returns Map<argName, varName> for args after the first positional arg.
 * Placeholders resolve to their inner name. (lawof is unary and has no
 * boundary kwargs.)
 */
function extractBoundaries(valueNode) {
  if (!valueNode || valueNode.type !== 'CallExpr') return null;
  const callee = valueNode.callee;
  if (!callee || callee.type !== 'Identifier') return null;
  if (callee.name !== 'functionof' && callee.name !== 'kernelof') return null;

  const boundaries = new Map();
  const args = valueNode.args;

  // Skip first arg (the expression to reify), process keyword args
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg.type === 'KeywordArg') {
      const argName = arg.name;
      let varName = null;
      if (arg.value.type === 'Identifier') {
        varName = arg.value.name;
      } else if (arg.value.type === 'Placeholder') {
        varName = '_' + arg.value.name + '_'; // full placeholder form for matching
      }
      if (varName) boundaries.set(argName, varName);
    }
  }
  return boundaries.size > 0 ? boundaries : null;
}

/**
 * Extract the field map from a joint-measure expression, when it can be
 * statically resolved.
 *
 * Recognised forms:
 *  - Tier 1 (`lawof_record`): `lawof(record(name1 = node1, ...))`
 *    — each field maps to a module-level node name. lawof is unary now,
 *    so there are no inherited boundaries from this form.
 *  - Tier 2 (`joint`): `joint(name1 = M1, ...)` keyword form
 *    — each field maps to an inline measure expression. Components are
 *    independent (no cross-boundaries between fields).
 *
 * Returns:
 *   { kind: 'lawof_record' | 'joint',
 *     fields: Map<fieldName, AST-expression>,
 *     inheritedBoundaries: Map<argName, varName> }
 * or null if the structure cannot be statically resolved.
 *
 * For 'lawof_record', each field's expression is an Identifier referring to a
 * module-level binding.
 * For 'joint', each field's expression is an arbitrary measure expression
 * (typically a CallExpr like Normal(...) or a measure-algebra construction).
 */
function extractJointFields(valueNode) {
  if (!valueNode || valueNode.type !== 'CallExpr') return null;
  if (!valueNode.callee || valueNode.callee.type !== 'Identifier') return null;

  // ----- Tier 1: lawof(record(...)) (unary) -----
  // Disintegrate operates on joint *measures*, so only unary lawof of a
  // record qualifies as a Tier 1 target. `kernelof(record(...), kwargs...)`
  // produces a kernel, not a joint measure, and cannot be disintegrated.
  if (valueNode.callee.name === 'lawof') {
    if (valueNode.args.length !== 1) return null;
    const firstArg = valueNode.args[0];
    if (firstArg.type !== 'CallExpr' || !firstArg.callee
        || firstArg.callee.type !== 'Identifier'
        || firstArg.callee.name !== 'record') {
      return null;
    }

    const fields = new Map();
    for (const arg of firstArg.args) {
      if (arg.type !== 'KeywordArg') return null; // not statically resolvable
      if (arg.value.type !== 'Identifier') return null; // can't trace back to a node name
      fields.set(arg.name, arg.value);
    }
    return { kind: 'lawof_record', fields, inheritedBoundaries: new Map() };
  }

  // ----- Tier 2: joint(name1 = M1, ...) keyword form -----
  if (valueNode.callee.name === 'joint') {
    if (valueNode.args.length === 0) return null;
    const fields = new Map();
    for (const arg of valueNode.args) {
      if (arg.type !== 'KeywordArg') return null; // positional joint not statically inspectable here
      fields.set(arg.name, arg.value); // arbitrary measure expression
    }
    return { kind: 'joint', fields, inheritedBoundaries: new Map() };
  }

  // ----- Tier 2: jointchain(name1 = M1, name2 = K2, ...) keyword form -----
  // The chain order matters: later fields may depend on earlier fields' variates.
  // We don't try to model that fully here — for disintegration along trailing
  // selected fields, the kernel gets synthesized chain-earlier field labels as
  // boundary inputs (see dag.js).
  if (valueNode.callee.name === 'jointchain') {
    if (valueNode.args.length === 0) return null;
    const fields = new Map();
    for (const arg of valueNode.args) {
      if (arg.type !== 'KeywordArg') return null;
      fields.set(arg.name, arg.value);
    }
    return { kind: 'jointchain', fields, inheritedBoundaries: new Map() };
  }

  return null;
}

/**
 * Detect a disintegrate-decomposition statement and resolve its structure.
 *
 *   kernel_name, prior_name = disintegrate(selector, joint_ref)
 *
 * `selector` may be a string literal or an array of string literals.
 * `joint_ref` must be an Identifier referencing a binding whose RHS is a
 * statically-resolvable joint measure (currently: lawof(record(...))).
 *
 * Returns null if any part doesn't match.
 *
 * @param {object} stmt - AssignStatement
 * @param {Map} bindingMap - already-built bindings map (for joint lookup)
 * @returns {{ kernelName, priorName, selectorFields, jointName, jointFields, inheritedBoundaries } | null}
 */
function detectDisintegration(stmt, bindingMap) {
  if (stmt.type !== 'AssignStatement') return null;
  if (stmt.names.length !== 2) return null;
  if (stmt.value.type !== 'CallExpr') return null;
  if (!stmt.value.callee || stmt.value.callee.type !== 'Identifier') return null;
  if (stmt.value.callee.name !== 'disintegrate') return null;
  if (stmt.value.args.length !== 2) return null;

  const selectorArg = stmt.value.args[0];
  const jointArg = stmt.value.args[1];

  // Parse selector
  let selectorFields = null;
  if (selectorArg.type === 'StringLiteral') {
    selectorFields = [selectorArg.value];
  } else if (selectorArg.type === 'ArrayLiteral') {
    selectorFields = [];
    for (const el of selectorArg.elements) {
      if (el.type !== 'StringLiteral') return null;
      selectorFields.push(el.value);
    }
  } else {
    return null;
  }

  if (jointArg.type !== 'Identifier') return null;
  const jointBinding = bindingMap.get(jointArg.name);
  if (!jointBinding) return null;

  // We no longer pre-screen with extractJointFields — the rewriter will
  // determine structurally whether this disintegration is supported, and
  // selector errors come back as Unsupported reasons. extractJointFields
  // is kept around for selector-error diagnostics on the cases it does
  // recognise (see pass 3 in analyze).
  const jointInfo = extractJointFields(jointBinding.node.value);

  return {
    kernelName: stmt.names[0].name,
    priorName: stmt.names[1].name,
    selectorFields,
    jointName: jointArg.name,
    jointKind:           jointInfo ? jointInfo.kind                : null,
    jointFields:         jointInfo ? jointInfo.fields              : null,
    inheritedBoundaries: jointInfo ? jointInfo.inheritedBoundaries : new Map(),
    selectorLoc: selectorArg.loc,
  };
}

/**
 * Attach an "effective RHS" view to a binding so the DAG renderer can
 * treat it as if its source were `effectiveValue` instead of the
 * statement's literal RHS. Used by disintegration to render synthesized
 * kernel/prior expressions naturally.
 */
function attachEffectiveRhs(binding, effectiveValue, definedNames) {
  binding.effectiveValue = effectiveValue;
  const { deps, callDeps } = collectDeps(effectiveValue, definedNames);
  // Self-references are never deps for rendering purposes.
  for (const n of binding.names || [binding.name]) {
    deps.delete(n);
    callDeps.delete(n);
  }
  binding.effectiveDeps = [...deps];
  binding.effectiveCallDeps = [...callDeps];
}

/**
 * For a Plan.delegate disintegration result, mirror the delegate target's
 * RHS view onto this binding so it renders identically — same kernelof/
 * lawof structure, same boundaries, same ancestor trace. The binding
 * keeps its own identity (LHS name, source location) but shares the
 * target's effective semantics.
 */
function attachDelegate(binding, targetName, bindings) {
  const target = bindings.get(targetName);
  if (!target || !target.node || !target.node.value) return;
  binding.effectiveValue    = target.node.value;
  binding.effectiveDeps     = [...(target.deps || [])];
  binding.effectiveCallDeps = [...(target.callDeps || [])];
}

/**
 * Compute the phase of every binding via ancestor analysis, per spec
 * (`docs/04-design.md#phases`).
 *
 *  - `draw(...)` self → stochastic
 *  - `elementof(...)` self → parameterized
 *  - `external(...)` self → fixed (despite being an "input")
 *  - any other binding → max of its dependencies' phases, where
 *    stochastic > parameterized > fixed
 *
 * @param {Map} bindings
 * @returns {Map<string, 'fixed' | 'parameterized' | 'stochastic'>}
 */
function computePhases(bindings) {
  const phases = new Map();
  const visiting = new Set();

  function calleeName(b) {
    const v = b && b.node && b.node.value;
    if (v && v.type === 'CallExpr' && v.callee && v.callee.type === 'Identifier') {
      return v.callee.name;
    }
    return null;
  }

  function maxPhase(a, b) {
    if (a === 'stochastic' || b === 'stochastic') return 'stochastic';
    if (a === 'parameterized' || b === 'parameterized') return 'parameterized';
    return 'fixed';
  }

  function phaseOf(name) {
    if (phases.has(name)) return phases.get(name);
    if (visiting.has(name)) return 'fixed'; // cycle (shouldn't occur in valid code)
    visiting.add(name);

    const b = bindings.get(name);
    if (!b) {
      visiting.delete(name);
      phases.set(name, 'fixed');
      return 'fixed';
    }

    const cn = calleeName(b);
    let phase;
    if (cn === 'draw') {
      phase = 'stochastic';
    } else if (cn === 'elementof') {
      phase = 'parameterized';
    } else if (cn === 'external') {
      phase = 'fixed';
    } else {
      phase = 'fixed';
      for (const dep of b.deps) {
        phase = maxPhase(phase, phaseOf(dep));
        if (phase === 'stochastic') break;
      }
    }

    visiting.delete(name);
    phases.set(name, phase);
    return phase;
  }

  for (const name of bindings.keys()) phaseOf(name);
  return phases;
}

/**
 * Validate that all literal integer indices in `IndexExpr` nodes are >= 1.
 * FlatPPL uses 1-based indexing throughout (arrays, tables, tuples), so a
 * literal `x[0]` or `x[-1]` is always invalid regardless of the container's
 * type. Runtime expressions are not checked.
 *
 * @param {object} node - root expression node
 * @param {Diagnostic[]} diagnostics - mutable, appended to
 */
function validateIndexing(node, diagnostics) {
  function checkIndex(idx) {
    // Direct integer literal: x[0], x[2.5] (non-integer is a type error elsewhere)
    if (idx.type === 'NumberLiteral'
        && Number.isInteger(idx.value) && idx.value <= 0) {
      diagnostics.push({
        severity: 'error',
        message: `Invalid index ${idx.value}: FlatPPL uses 1-based indexing (indices start at 1)`,
        loc: idx.loc,
      });
      return;
    }
    // Negated literal: x[-1] parses as UnaryExpr('-', NumberLiteral(1))
    if (idx.type === 'UnaryExpr' && idx.op === '-'
        && idx.operand.type === 'NumberLiteral'
        && Number.isInteger(idx.operand.value) && idx.operand.value > 0) {
      diagnostics.push({
        severity: 'error',
        message: `Invalid index -${idx.operand.value}: FlatPPL uses 1-based indexing (indices start at 1)`,
        loc: idx.loc,
      });
    }
  }

  function walk(node) {
    if (!node) return;
    if (node.type === 'IndexExpr') {
      walk(node.object);
      for (const i of node.indices) {
        checkIndex(i);
        walk(i);
      }
      return;
    }
    if (node.type === 'CallExpr') { walk(node.callee); for (const a of node.args) walk(a); return; }
    if (node.type === 'BinaryExpr') { walk(node.left); walk(node.right); return; }
    if (node.type === 'UnaryExpr') { walk(node.operand); return; }
    if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') {
      for (const e of node.elements) walk(e);
      return;
    }
    if (node.type === 'FieldAccess') { walk(node.object); return; }
    if (node.type === 'KeywordArg') { walk(node.value); return; }
    // Leaves: NumberLiteral, StringLiteral, BoolLiteral, ConstantRef, SetRef,
    // Identifier, Placeholder, Hole, SliceAll — no recursion.
  }
  walk(node);
}

/**
 * Validate hole (`_`) and placeholder (`_name_`) usage according to the spec:
 *  - `_` is only valid inside `fn(...)`.
 *  - `_name_` is only valid inside `functionof(...)` or `kernelof(...)`.
 *    (lawof is unary now and cannot bind placeholders.)
 *
 * Scope is determined by the nearest enclosing special operation.
 *
 * @param {object} node - root expression node
 * @param {Diagnostic[]} diagnostics - mutable, appended to
 */
function validateHolesAndPlaceholders(node, diagnostics) {
  // scope can be: 'normal', 'fn', 'reify' (functionof/kernelof)
  function walk(node, scope) {
    if (!node) return;
    switch (node.type) {
      case 'Hole':
        if (scope !== 'fn') {
          diagnostics.push({
            severity: 'error',
            message: "Hole '_' may only appear inside fn(...)",
            loc: node.loc,
          });
        }
        return;
      case 'Placeholder':
        if (scope !== 'reify') {
          diagnostics.push({
            severity: 'error',
            message: `Placeholder '_${node.name}_' may only appear inside functionof(...) or kernelof(...)`,
            loc: node.loc,
          });
        }
        return;
      case 'CallExpr': {
        let inner = scope;
        if (node.callee && node.callee.type === 'Identifier') {
          if (node.callee.name === 'fn') inner = 'fn';
          else if (node.callee.name === 'functionof' || node.callee.name === 'kernelof') inner = 'reify';
        }
        walk(node.callee, scope);
        for (const a of node.args) walk(a, inner);
        return;
      }
      case 'BinaryExpr':
        walk(node.left, scope);
        walk(node.right, scope);
        return;
      case 'UnaryExpr':
        walk(node.operand, scope);
        return;
      case 'ArrayLiteral':
      case 'TupleLiteral':
        for (const e of node.elements) walk(e, scope);
        return;
      case 'IndexExpr':
        walk(node.object, scope);
        for (const i of node.indices) walk(i, scope);
        return;
      case 'FieldAccess':
        walk(node.object, scope);
        return;
      case 'KeywordArg':
        walk(node.value, scope);
        return;
      // Identifier, NumberLiteral, StringLiteral, BoolLiteral, ConstantRef,
      // SetRef, SliceAll: nothing to do.
    }
  }
  walk(node, 'normal');
}

/**
 * For fn() calls, count hole arguments in the expression.
 */
function countHoles(valueNode) {
  if (!valueNode || valueNode.type !== 'CallExpr') return 0;
  if (!valueNode.callee || valueNode.callee.name !== 'fn') return 0;

  let count = 0;
  function walk(node) {
    if (!node) return;
    if (node.type === 'Hole') { count++; return; }
    if (node.type === 'CallExpr') {
      walk(node.callee);
      for (const a of node.args) walk(a);
    }
    if (node.type === 'BinaryExpr') { walk(node.left); walk(node.right); }
    if (node.type === 'UnaryExpr') walk(node.operand);
    if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') for (const e of node.elements) walk(e);
    if (node.type === 'IndexExpr') { walk(node.object); for (const i of node.indices) walk(i); }
    if (node.type === 'FieldAccess') walk(node.object);
    if (node.type === 'KeywordArg') walk(node.value);
  }

  // Walk the first argument of fn()
  for (const arg of valueNode.args) walk(arg);
  return count;
}

/**
 * Reconstruct the RHS expression as source text from the original source.
 */
function sliceSource(source, loc) {
  const lines = source.split('\n');
  if (loc.start.line === loc.end.line) {
    return lines[loc.start.line].slice(loc.start.col, loc.end.col);
  }
  let result = lines[loc.start.line].slice(loc.start.col);
  for (let i = loc.start.line + 1; i < loc.end.line; i++) {
    result += '\n' + lines[i];
  }
  result += '\n' + lines[loc.end.line].slice(0, loc.end.col);
  return result;
}

/**
 * Find all identifier references in an expression and their locations.
 * Used by definition/hover providers to find what's under the cursor.
 */
function collectIdentRefs(node) {
  const refs = [];
  function walk(node) {
    if (!node) return;
    if (node.type === 'Identifier') { refs.push(node); return; }
    if (node.type === 'CallExpr') { walk(node.callee); for (const a of node.args) walk(a); }
    if (node.type === 'BinaryExpr') { walk(node.left); walk(node.right); }
    if (node.type === 'UnaryExpr') walk(node.operand);
    if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') for (const e of node.elements) walk(e);
    if (node.type === 'IndexExpr') { walk(node.object); for (const i of node.indices) walk(i); }
    if (node.type === 'FieldAccess') walk(node.object);
    if (node.type === 'KeywordArg') walk(node.value);
  }
  walk(node);
  return refs;
}

/**
 * Analyze a parsed AST.
 * Returns { bindings, diagnostics, symbols }.
 *
 * @param {object} ast - Program AST node
 * @param {string} source - original source text (for expression slicing)
 */
function analyze(ast, source) {
  const diagnostics = [];
  const bindings = new Map();
  const symbols = [];
  const definedNames = new Set();

  // First pass: collect all defined names
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;
    for (const nameNode of stmt.names) {
      if (definedNames.has(nameNode.name)) {
        diagnostics.push({
          severity: 'error',
          message: `Duplicate variable name '${nameNode.name}'`,
          loc: nameNode.loc,
        });
      }
      definedNames.add(nameNode.name);
    }
  }

  // Second pass: classify, extract deps, build bindings
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;

    const stmtType = classifyStatement(stmt.value);
    diagnostics.push(...validateSpecialOperation(stmt.value));
    validateHolesAndPlaceholders(stmt.value, diagnostics);
    validateIndexing(stmt.value, diagnostics);
    const { deps, callDeps } = collectDeps(stmt.value, definedNames);
    const rhs = sliceSource(source, stmt.value.loc);

    // Remove self-references
    for (const nameNode of stmt.names) {
      deps.delete(nameNode.name);
      callDeps.delete(nameNode.name);
    }

    // Check for undefined references
    const refs = collectIdentRefs(stmt.value);
    for (const ref of refs) {
      if (!definedNames.has(ref.name) && !isKnownName(ref.name)) {
        diagnostics.push({
          severity: 'warning',
          message: `Undefined variable '${ref.name}'`,
          loc: ref.loc,
        });
      }
    }

    // Build binding info for each name
    for (const nameNode of stmt.names) {
      const info = {
        name: nameNode.name,
        names: stmt.names.map(n => n.name),
        line: stmt.loc.start.line,
        rhs,
        type: stmtType,
        deps: [...deps],
        callDeps: [...callDeps],
        node: stmt,
        nameLoc: nameNode.loc,
      };
      bindings.set(nameNode.name, info);

      // Build symbol for outline
      const kindMap = {
        draw: 'Variable', input: 'Variable', call: 'Variable',
        lawof: 'Function', functionof: 'Function', kernelof: 'Function', fn: 'Function',
        likelihood: 'Variable', bayesupdate: 'Variable',
        literal: 'Constant', module: 'Module', table: 'Variable',
      };
      symbols.push({
        name: nameNode.name,
        kind: kindMap[stmtType] || 'Variable',
        type: stmtType,
        loc: stmt.loc,
        nameLoc: nameNode.loc,
      });
    }
  }

  // Third pass: detect disintegrate-decompositions, tag results, validate selectors.
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;
    const info = detectDisintegration(stmt, bindings);
    if (!info) continue;

    // Validate selector fields exist in the joint's record (when the joint
    // is in a form whose fields we can statically enumerate). For positional
    // jointchain or other forms, the rewriter's Unsupported reason carries
    // the equivalent diagnostic — emitted below.
    if (info.jointFields) {
      for (const field of info.selectorFields) {
        if (!info.jointFields.has(field)) {
          diagnostics.push({
            severity: 'error',
            message: `disintegrate: selector field '${field}' not found in joint measure '${info.jointName}'`,
            loc: info.selectorLoc,
          });
        }
      }
    }

    // Compute the structural-disintegration Plan first; downstream tagging
    // depends on whether the rewriter could resolve the joint structurally.
    const jointBinding = bindings.get(info.jointName);
    let plan = null;
    if (jointBinding && jointBinding.node && jointBinding.node.value) {
      plan = disintegratePlan(
        jointBinding.node.value, info.selectorFields, bindings,
        { seen: new Set(), source: info.jointName });
    }

    const kernel = bindings.get(info.kernelName);
    const prior  = bindings.get(info.priorName);

    // Only tag the result bindings as kernel/prior of a structural
    // disintegration when the rewriter actually resolved one. Unsupported
    // plans fall back to the plain dep trace via the literal RHS.
    const resolved = plan && (plan.kind === 'synthesized' || plan.kind === 'delegate');
    if (resolved) {
      if (kernel) {
        kernel.type = 'kernelof';
        kernel.disintegrateRole = { kind: 'kernel', ...info };
        kernel.disintegratePlan = plan;
      }
      if (prior) {
        prior.type = 'lawof';
        prior.disintegrateRole = { kind: 'prior', ...info };
        prior.disintegratePlan = plan;
      }

      if (plan.kind === 'synthesized') {
        if (kernel) attachEffectiveRhs(kernel, plan.kernel, definedNames);
        if (prior)  attachEffectiveRhs(prior,  plan.prior,  definedNames);
      } else /* delegate */ {
        // Render the result identically to the delegate target — this is
        // the "the disintegration recovered an existing binding" case.
        if (kernel) attachDelegate(kernel, plan.kernel.binding, bindings);
        if (prior)  attachDelegate(prior,  plan.prior.binding,  bindings);
      }
    } else if (plan) {
      // Keep the Plan around (even Unsupported) so the renderer or a
      // future diagnostic surface can read its reason. Don't change the
      // binding's type — fall back to plain dep trace.
      if (kernel) kernel.disintegratePlan = plan;
      if (prior)  prior.disintegratePlan  = plan;
    }
  }

  // Fourth pass: compute phases (stochastic | parameterized | fixed) by
  // ancestor analysis, per spec.
  const phases = computePhases(bindings);
  for (const [name, phase] of phases) {
    const b = bindings.get(name);
    if (b) b.phase = phase;
  }

  return { bindings, diagnostics, symbols };
}

/**
 * Test whether a string is a valid public/private/auto-generated binding name.
 *
 * Rejects: reserved names (self, base), bare `_`, placeholder pattern `_x_`,
 * and any name that doesn't match one of the canonical regular-expression
 * patterns from the spec (`docs/04-design.md#sec:binding-names`).
 */
function isValidBindingName(name) {
  if (typeof name !== 'string' || name.length === 0) return false;
  if (name === 'self' || name === 'base') return false;
  if (name === '_') return false; // discard, not a renameable target
  // Public:        ^[A-Za-z][A-Za-z0-9_]*$
  // Private:       ^_[A-Za-z]([A-Za-z0-9_]*[A-Za-z0-9])?$
  // Auto-gen:      ^__[A-Za-z0-9][A-Za-z0-9_]*$
  return /^[A-Za-z][A-Za-z0-9_]*$/.test(name)
      || /^_[A-Za-z]([A-Za-z0-9_]*[A-Za-z0-9])?$/.test(name)
      || /^__[A-Za-z0-9][A-Za-z0-9_]*$/.test(name);
}

/**
 * Test whether a string is a valid placeholder source token (with surrounding
 * underscores, e.g. `_par_`).
 */
function isValidPlaceholderText(text) {
  return typeof text === 'string'
      && /^_[A-Za-z]([A-Za-z0-9_]*[A-Za-z0-9])?_$/.test(text);
}

/**
 * Plan a rename action at a given cursor position.
 *
 * Walks the AST, identifies what's under the cursor, and returns enough info
 * for a rename provider to act on. Returns null when the position isn't a
 * renameable target (e.g. on a literal, a comment, or a bare-`_` LHS).
 *
 * @param {object} ast - parsed Program AST
 * @param {Map} bindings - analyzer bindings map
 * @param {number} line - 0-based cursor line
 * @param {number} col - 0-based cursor column
 * @returns {{ kind: 'binding', oldName: string, targetLoc, locs: Loc[] }
 *         | { kind: 'placeholder', oldName: string, targetLoc, locs: Loc[] }
 *         | null}
 *
 * For 'binding': `locs` includes the binding's defining nameLoc plus every
 *   Identifier reference site in any other statement's RHS.
 * For 'placeholder': `oldName` is the placeholder *inner* name (without the
 *   surrounding underscores). `locs` includes every Placeholder node that
 *   shares the same nearest enclosing `functionof`/`lawof` scope.
 *   Each loc covers the full `_name_` source span.
 */
function planRename(ast, bindings, line, col) {
  function inLoc(loc) {
    return loc && loc.start.line <= line && line <= loc.end.line
        && (loc.start.line < line || col >= loc.start.col)
        && (line < loc.end.line || col <= loc.end.col);
  }

  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;

    // LHS names — direct binding references.
    for (const nameNode of stmt.names) {
      if (!inLoc(nameNode.loc)) continue;
      const name = nameNode.name;
      if (name === '_') return null; // discard binding, can't rename
      if (!bindings.has(name)) return null;
      return planBindingRename(ast, bindings, name);
    }

    // RHS expression — could be an identifier reference or a placeholder.
    const target = findCursorTargetInExpr(stmt.value, inLoc);
    if (target) {
      if (target.kind === 'identifier') {
        if (!bindings.has(target.name)) return null;
        return planBindingRename(ast, bindings, target.name);
      }
      if (target.kind === 'placeholder' && target.scope) {
        return planPlaceholderRename(target.scope, target.name, target.loc);
      }
    }
  }
  return null;
}

function planBindingRename(ast, bindings, name) {
  const binding = bindings.get(name);
  if (!binding) return null;

  const locs = [binding.nameLoc];
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;
    const refs = collectIdentRefs(stmt.value);
    for (const ref of refs) {
      if (ref.name === name) locs.push(ref.loc);
    }
  }
  return { kind: 'binding', oldName: name, targetLoc: binding.nameLoc, locs };
}

function planPlaceholderRename(scopeCallExpr, name, targetLoc) {
  const locs = [];
  function walk(node) {
    if (!node) return;
    if (node.type === 'Placeholder') {
      if (node.name === name) locs.push(node.loc);
      return;
    }
    if (node.type === 'CallExpr') {
      // Stop at NESTED functionof/kernelof — those are different placeholder scopes.
      if (node !== scopeCallExpr
          && node.callee && node.callee.type === 'Identifier'
          && (node.callee.name === 'functionof' || node.callee.name === 'kernelof')) {
        return;
      }
      walk(node.callee);
      for (const a of node.args) walk(a);
      return;
    }
    if (node.type === 'BinaryExpr') { walk(node.left); walk(node.right); return; }
    if (node.type === 'UnaryExpr') { walk(node.operand); return; }
    if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') {
      for (const e of node.elements) walk(e);
      return;
    }
    if (node.type === 'IndexExpr') {
      walk(node.object);
      for (const i of node.indices) walk(i);
      return;
    }
    if (node.type === 'FieldAccess') { walk(node.object); return; }
    if (node.type === 'KeywordArg') { walk(node.value); return; }
  }
  for (const a of scopeCallExpr.args) walk(a);
  return { kind: 'placeholder', oldName: name, targetLoc, locs };
}

/**
 * Find a renameable AST node at the cursor position within an expression.
 * Tracks the nearest enclosing functionof/kernelof CallExpr as the
 * placeholder scope.
 */
function findCursorTargetInExpr(root, inLoc) {
  let result = null;
  function walk(node, scope) {
    if (!node || result) return;
    if (node.type === 'Identifier' && inLoc(node.loc)) {
      result = { kind: 'identifier', name: node.name, loc: node.loc };
      return;
    }
    if (node.type === 'Placeholder' && inLoc(node.loc)) {
      result = { kind: 'placeholder', name: node.name, loc: node.loc, scope };
      return;
    }
    if (node.type === 'CallExpr') {
      let inner = scope;
      if (node.callee && node.callee.type === 'Identifier'
          && (node.callee.name === 'functionof' || node.callee.name === 'kernelof')) {
        inner = node;
      }
      walk(node.callee, scope);
      for (const a of node.args) walk(a, inner);
      return;
    }
    if (node.type === 'BinaryExpr') { walk(node.left, scope); walk(node.right, scope); return; }
    if (node.type === 'UnaryExpr') { walk(node.operand, scope); return; }
    if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') {
      for (const e of node.elements) walk(e, scope);
      return;
    }
    if (node.type === 'IndexExpr') {
      walk(node.object, scope);
      for (const i of node.indices) walk(i, scope);
      return;
    }
    if (node.type === 'FieldAccess') { walk(node.object, scope); return; }
    if (node.type === 'KeywordArg') { walk(node.value, scope); return; }
  }
  walk(root, null);
  return result;
}

/**
 * Find the chain of enclosing AST node ranges at a given cursor position,
 * ordered from innermost to outermost.
 *
 * Used by SelectionRangeProvider to power "Expand Selection" (Shift+Alt+→).
 *
 * @param {object} ast - parsed Program AST
 * @param {number} line - 0-based cursor line
 * @param {number} col - 0-based cursor column
 * @returns {Array<Loc>} innermost first
 */
function findEnclosingRanges(ast, line, col) {
  function inLoc(loc) {
    return loc && loc.start.line <= line && line <= loc.end.line
        && (loc.start.line < line || col >= loc.start.col)
        && (line < loc.end.line || col <= loc.end.col);
  }

  const ranges = []; // outermost first; we'll reverse at the end

  function walk(node) {
    if (!node) return;
    // Program has no .loc — descend into body without recording a range.
    if (node.type === 'Program') {
      for (const s of node.body) walk(s);
      return;
    }
    if (!node.loc || !inLoc(node.loc)) return;
    ranges.push(node.loc);
    // Recurse into children — the deepest matching node is appended last.
    switch (node.type) {
      case 'AssignStatement':
        for (const n of node.names) walk(n);
        walk(node.value);
        break;
      case 'CallExpr':
        walk(node.callee);
        for (const a of node.args) walk(a);
        break;
      case 'BinaryExpr':
        walk(node.left);
        walk(node.right);
        break;
      case 'UnaryExpr':
        walk(node.operand);
        break;
      case 'ArrayLiteral':
      case 'TupleLiteral':
        for (const e of node.elements) walk(e);
        break;
      case 'IndexExpr':
        walk(node.object);
        for (const i of node.indices) walk(i);
        break;
      case 'FieldAccess':
        walk(node.object);
        break;
      case 'KeywordArg':
        walk(node.value);
        break;
      // Leaf nodes have no children to walk.
    }
  }

  walk(ast);
  return ranges.reverse(); // innermost first
}

module.exports = {
  analyze, classifyStatement, collectDeps,
  extractBoundaries, extractJointFields, detectDisintegration,
  countHoles, validateHolesAndPlaceholders,
  computePhases, isMeasureExpr,
  collectIdentRefs, sliceSource,
  planRename, isValidBindingName, isValidPlaceholderText,
  findEnclosingRanges,
};
