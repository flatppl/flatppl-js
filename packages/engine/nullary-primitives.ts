'use strict';

// The nullary form of the measure-kernel evaluation primitives (spec §07
// §sec:measure-eval-prims). A measure is a nullary kernel and takes no
// kernel_input, so `builtin_logdensityof(M, x)`, `builtin_sample(rngstate, M,
// n, …)` and the four transports omit it.
//
// A closed built-in measure `K(args)` is the kernel `K` applied to the input
// `record(args)`, so this pass rewrites the nullary call to the kernel form
// `builtin_*(K, record(args), …)` before the rest of the pipeline sees it. Every
// consumer of the primitives then handles one form. Like `restrict`, the rewrite
// runs on the AST at the top of `analyze`.
//
// A broadcast holds the measure constant (spec §04). Records are no broadcast
// operands, so `builtin_logdensityof.(M, xs)` maps a per-cell function:
// `broadcast(fn(builtin_logdensityof(K, record(args), _)), xs)`. A broadcast
// `builtin_sample` has no such form here and is left as written.
//
// Anything else in the kernel slot is left as written: a bare kernel name, a
// composite measure, a measure-input kernel's own refusal in `lower.ts`.

const AST = require('./ast.ts');
const { FLATPDL_KERNEL_TAGS, PARTICLE_PHYSICS_DISTRIBUTIONS } = require('./builtins.ts');

// The kernel argument index of each primitive (spec §07 table).
const KERNEL_SLOT: Record<string, number> = {
  builtin_logdensityof: 0,
  builtin_touniform: 0,
  builtin_fromuniform: 0,
  builtin_tonormal: 0,
  builtin_fromnormal: 0,
  builtin_sample: 1,
};

function isCallTo(node: any, name: string): boolean {
  return node && node.type === 'CallExpr' && node.callee?.type === 'Identifier'
    && node.callee.name === name;
}

function hasKeyword(args: any[]): boolean {
  return args.some((a: any) => a && a.type === 'KeywordArg');
}

// The ordered parameter names of a built-in measure kernel.
function paramNames(name: string): string[] | null {
  const iface = require('./types.ts').builtinInterface(name);
  if (iface) return iface.inputs.map((p: any) => p.name);
  const entry = require('./sampler-registry.ts').REGISTRY[name];
  if (entry && Array.isArray(entry.params)) return entry.params;
  return require('./builtin-param-names.ts').builtinParamNames(name);
}

function expandNullaryPrimitives(ast: any): void {
  if (!ast || !ast.body) return;
  const values = new Map<string, any>();
  for (const s of ast.body) {
    if (s.type === 'AssignStatement' && s.names.length === 1) values.set(s.names[0].name, s.value);
  }

  // The kernel name and constructor call of a closed built-in measure, through
  // bare-name aliases.
  function builtinMeasure(node: any, seen: Set<string>): { name: string, call: any } | null {
    if (!node) return null;
    if (node.type === 'Identifier' && values.has(node.name) && !seen.has(node.name)) {
      seen.add(node.name);
      return builtinMeasure(values.get(node.name), seen);
    }
    if (node.type !== 'CallExpr' || !node.callee) return null;
    const callee = node.callee;
    if (callee.type === 'Identifier' && FLATPDL_KERNEL_TAGS.has(callee.name)
        && !values.has(callee.name)) {
      return { name: callee.name, call: node };
    }
    // A §09 member `alias.Dist(...)` on a standard module. Its kernel tag is the
    // bare member name, as the determiniser emits it.
    if (callee.type === 'FieldAccess' && callee.object?.type === 'Identifier'
        && PARTICLE_PHYSICS_DISTRIBUTIONS.has(callee.field)
        && isCallTo(values.get(callee.object.name), 'standard_module')) {
      return { name: callee.field, call: node };
    }
    return null;
  }

  // The kernel input `record(...)` of constructor call `call` (spec §04
  // calling convention): keywords by name, positionals in declared order, and a
  // sole record argument of a multi-input kernel as the record itself.
  function inputRecord(name: string, call: any): any {
    const args = call.args || [];
    const positional = args.filter((a: any) => a.type !== 'KeywordArg');
    const keywords = args.filter((a: any) => a.type === 'KeywordArg');
    const names = paramNames(name);
    if (!names) return null;
    if (names.length !== 1 && positional.length === 1 && keywords.length === 0
        && isCallTo(positional[0], 'record')) {
      return positional[0];
    }
    if (positional.length > names.length) return null;
    const fields = positional.map((a: any, i: number) => AST.KeywordArg(names[i], a, a.loc))
      .concat(keywords);
    return AST.CallExpr(AST.Identifier('record', call.loc), fields, call.loc);
  }

  function rewrite(node: any): void {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { for (const child of node) rewrite(child); return; }
    for (const key in node) if (key !== 'loc') rewrite(node[key]);
    if (node.type !== 'CallExpr' || node.callee?.type !== 'Identifier') return;
    const args = node.args || [];
    if (hasKeyword(args)) return;
    const head = node.callee.name;
    if (head in KERNEL_SLOT) {
      const slot = KERNEL_SLOT[head];
      const measure = builtinMeasure(args[slot], new Set());
      const input = measure && inputRecord(measure.name, measure.call);
      if (!input) return;
      node.args = [...args.slice(0, slot), AST.Identifier(measure.name, args[slot].loc), input,
        ...args.slice(slot + 1)];
      return;
    }
    if (head === 'broadcast' && args.length === 3 && args[0].type === 'Identifier'
        && KERNEL_SLOT[args[0].name] === 0) {
      const measure = builtinMeasure(args[1], new Set());
      const input = measure && inputRecord(measure.name, measure.call);
      if (!input) return;
      const loc = node.loc;
      const cell = AST.CallExpr(AST.Identifier(args[0].name, args[0].loc),
        [AST.Identifier(measure.name, args[1].loc), input, AST.Hole(loc)], loc);
      node.args = [AST.CallExpr(AST.Identifier('fn', loc), [cell], loc), args[2]];
    }
  }

  for (const s of ast.body) if (s.type === 'AssignStatement') rewrite(s.value);
}

module.exports = { expandNullaryPrimitives };
