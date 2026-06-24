'use strict';

// Module linking (spec §04 Module composition; §11 "modules are not
// flattened in FlatPIR, though tooling may flatten them internally").
//
// The CANONICAL representation keeps modules separate — each compiles to
// its own LoweredModule, cross-module access lowers to `(%ref alias x)`
// (done in lower.ts), and cross-module type inference reads the loaded
// module's types (typeinfer.ts). This pass is the engine-INTERNAL
// flattening that makes cross-module MATERIALISATION work with zero
// changes to the by-name materialiser / density / sampler: it splices a
// loaded module's bindings into one combined module under namespaced
// names, rewriting refs and rewiring load-time substitutions, then the
// caller re-analyzes the combined AST and builds derivations as usual.
//
// This is the classic linker / symbol-resolution model (and mirrors the
// in-tree `alias-resolution` "resolve once, downstream is uniform"
// discipline + `deriveAppliedKernel`'s "synthesise bindings, re-derive"
// pattern). Naming: each `load_module` BINDING is one module INSTANCE
// (the same file loaded twice = two instances, distinct substitutions),
// prefixed by the path of aliases from the root: a binding `x` of the
// module loaded as `m` in the root becomes `m$x`; if that module itself
// loads `s`, its `y` becomes `m$s$y`. The root (primary) module keeps its
// names unprefixed, so the combined graph is a drop-in superset.

const AST = require('./ast.ts');

// Instance prefix of a sub-module loaded as `alias` from an instance at
// `prefix`. The root is '' (unprefixed); a sub-load gets `prefix$alias`.
function _childPrefix(prefix: string, alias: string): string {
  return prefix === '' ? alias : prefix + '$' + alias;
}
// Namespaced name of a binding `name` inside the instance at `prefix`.
function _pfx(prefix: string, name: string): string {
  return prefix === '' ? name : prefix + '$' + name;
}

function _isInputCall(v: any): boolean {
  return v && v.type === 'CallExpr' && v.callee && v.callee.type === 'Identifier'
    && (v.callee.name === 'elementof' || v.callee.name === 'external');
}

// Rewrite the module-level references inside an expression AST into the
// linked namespace of the instance at `prefix`.
//   - `localNames`: this module's binding names (each becomes `prefix$name`).
//     Lambda formals / `_x_` placeholders / built-in names are NOT in this
//     set, so they pass through untouched (formals stay local, builtins
//     stay bare).
//   - `userAliases`: alias → childPrefix for the user `load_module` aliases
//     in this module. `mod.x` (member access on such an alias) rewrites to
//     the loaded binding `childPrefix$x`. A standard-module alias is NOT in
//     this map, so `hep.foo` keeps its FieldAccess shape over the prefixed
//     alias and resolves through the standard-module registry on re-analyze.
function _rewriteRefs(node: any, prefix: string,
  localNames: Set<string>, userAliases: Map<string, string>): any {
  if (node == null || typeof node !== 'object') return node;
  if (Array.isArray(node)) {
    return node.map((n) => _rewriteRefs(n, prefix, localNames, userAliases));
  }
  // Cross-module member access on a user module alias: `mod.x` → the
  // loaded binding's namespaced name.
  if (node.type === 'FieldAccess' && node.object && node.object.type === 'Identifier'
      && userAliases.has(node.object.name)) {
    return AST.Identifier(_pfx(userAliases.get(node.object.name)!, node.field), node.loc);
  }
  if (node.type === 'Identifier') {
    return localNames.has(node.name)
      ? AST.Identifier(_pfx(prefix, node.name), node.loc)
      : node;
  }
  const out: Record<string, any> = {};
  for (const k in node) {
    out[k] = (k === 'loc') ? node[k] : _rewriteRefs(node[k], prefix, localNames, userAliases);
  }
  return out;
}

// Splice one module instance into `out` (an array of AssignStatements).
// `subs` maps an input name of THIS instance to the substitution-value AST
// (already rewritten into the importer's namespace) bound by the enclosing
// `load_module(input = value)` — null for the root.
function _linkInstance(prefix: string, ast: any, moduleRegistry: any,
  modules: Map<string, any>, out: any[], subs: Map<string, any> | null,
  stackPaths: Set<string>): void {
  const stmts = (ast && ast.body) || [];
  const reg = moduleRegistry || {};

  const localNames = new Set<string>();
  for (const s of stmts) {
    if (s.type === 'AssignStatement') for (const nm of s.names) localNames.add(nm.name);
  }
  // User `load_module` aliases (alias → childPrefix). Standard-module
  // aliases are deliberately excluded — see _rewriteRefs.
  const userAliases = new Map<string, string>();
  for (const s of stmts) {
    if (s.type !== 'AssignStatement' || s.names.length !== 1) continue;
    const e = reg[s.names[0].name];
    if (e && e.kind === 'load_module') {
      userAliases.set(s.names[0].name, _childPrefix(prefix, s.names[0].name));
    }
  }
  const rewrite = (n: any) => _rewriteRefs(n, prefix, localNames, userAliases);

  for (const s of stmts) {
    if (s.type !== 'AssignStatement') continue;
    const aliasName = s.names.length === 1 ? s.names[0].name : null;
    const e = aliasName ? reg[aliasName] : null;

    // A user `load_module` binding: recurse into the loaded instance and
    // emit nothing for the module binding itself (member refs to it were
    // already rewritten to the loaded bindings' namespaced names).
    if (e && e.kind === 'load_module') {
      const dep = e.path != null ? modules.get(e.path) : null;
      if (!dep) continue;            // unresolved/missing dep — already diagnosed
      // Dependency cycle (already diagnosed by the bundle compiler): the
      // path is on the active instance stack. Don't recurse — that would
      // splice forever.
      if (stackPaths.has(e.path)) continue;
      const childSubs = new Map<string, any>();
      for (const arg of (s.value.args || [])) {
        if (arg.type === 'KeywordArg') childSubs.set(arg.name, rewrite(arg.value));
      }
      const childStack = new Set(stackPaths); childStack.add(e.path);
      _linkInstance(_childPrefix(prefix, aliasName!), dep.ast,
        dep.loweredModule && dep.loweredModule.moduleRegistry, modules, out, childSubs, childStack);
      continue;
    }

    // A substituted input: replace `mu = elementof(...)` with an alias to
    // the importer's substitution value (spec §04 load-time substitution).
    if (aliasName && subs && subs.has(aliasName) && _isInputCall(s.value)) {
      const subStmt = AST.AssignStatement(
        [AST.Identifier(_pfx(prefix, aliasName), s.names[0].loc)],
        subs.get(aliasName), s.loc);
      if (s.doc) subStmt.doc = s.doc;   // keep the binding's doc through linking
      out.push(subStmt);
      continue;
    }

    // An ordinary binding (incl. an UNsubstituted input, and a
    // standard_module load): emit under namespaced name(s) with refs
    // rewritten.
    const newNames = s.names.map((nm: any) => AST.Identifier(_pfx(prefix, nm.name), nm.loc));
    const stmt = AST.AssignStatement(newNames, rewrite(s.value), s.loc);
    if (s.doc) stmt.doc = s.doc;         // keep the binding's doc through linking (spec §04)
    out.push(stmt);
  }
}

// Build a single combined Program AST that flattens the primary module
// and all of its transitively-loaded dependencies (spec §11 internal
// flattening). The caller re-analyzes it to obtain a linked bindings map
// for derivation building / materialisation.
function linkModules(primary: any, modules: Map<string, any>): any {
  const out: any[] = [];
  _linkInstance('', primary.ast,
    primary.loweredModule && primary.loweredModule.moduleRegistry, modules, out, null, new Set());
  return AST.Program(out, []);
}

module.exports = { linkModules };
