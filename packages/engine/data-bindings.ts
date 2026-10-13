'use strict';

// One load-time expansion, before type/shape inference. The original binding
// AST remains the source view; only its effective RHS contains loaded values.
const dataload = require('./dataload.ts');
const lower = require('./lower.ts');
const fixedEval = require('./fixed-eval.ts');
const { resolveModulePath } = require('./module-resolve.ts');

function expandDataBindings(bindings: Map<string, any>, loweredModule: any,
  options: { sources: Record<string, string>; modulePath?: string },
  diagnostics: any[], attach: (binding: any, rhs: any) => void): boolean {
  const parsed = new Map<string, any>();
  const moduleNames = new Set(Object.keys(loweredModule.moduleRegistry || {}));
  const lowerCtx = { moduleNames, bindingNames: new Set(bindings.keys()) };
  const resolveSet = (ir: any) => ir.ns === 'self' ? loweredModule.bindings.get(ir.name)?.rhs : undefined;

  function load(ir: any, resolve: (ir: any) => any): any {
    const schema = dataload.resolveSchema(ir.kwargs?.valueset || ir.args?.[1], resolve, resolveSet);
    // An opaque placeholder is well-formed even if its file isn't parseable.
    // Its actionable refusal is retained on the source binding for access.
    if (dataload.containsAnything(schema)) return undefined;
    const source = resolve(ir.kwargs?.source || ir.args?.[0]);
    if (typeof source !== 'string') throw new Error('load_data: source must be a fixed string path');
    const path = resolveModulePath(options.modulePath || '<entry>', source);
    const text = options.sources[path];
    if (text === undefined) throw new Error(`load_data: data source not found in bundle: '${path}'`);
    if (!parsed.has(path)) parsed.set(path, dataload.parseByExtension(path, text));
    return dataload.valueExpression(parsed.get(path), schema, ir.loc, dataload.detectFormat(path) !== 'json');
  }
  const resolve = fixedEval.makeResolver({ loweredModule, resolveData: (ir: any, evalIR: any) => {
    const rhs = load(ir, evalIR);
    return rhs ? evalIR(lower.lowerExpr(rhs, lowerCtx)) : undefined;
  } });

  function rewrite(node: any): any {
    if (!node || typeof node !== 'object') return node;
    if (node.type === 'CallExpr' && node.callee?.type === 'Identifier'
        && node.callee.name === 'load_data') {
      const ir = lower.lowerExpr(node, lowerCtx);
      if (ir.kind === 'call' && ir.op === 'load_data' && !ir.target) return load(ir, resolve) || node;
    }
    let result = node;
    for (const key of Object.keys(node)) {
      if (key === 'loc') continue;
      const value = node[key];
      const next = Array.isArray(value) ? value.map(rewrite) : rewrite(value);
      const changed = Array.isArray(value) ? next.some((v: any, i: number) => v !== value[i]) : next !== value;
      if (changed) {
        if (result === node) result = { ...node };
        result[key] = next;
      }
    }
    return result;
  }
  let changed = false;
  for (const [name, binding] of bindings) {
    const original = binding.effectiveValue || binding.node?.value;
    if (!original) continue;
    try {
      const rhs = rewrite(original);
      if (rhs !== original) {
        attach(binding, rhs);
        binding.loadedData = true;
        changed = true;
      }
    } catch (e: any) {
      diagnostics.push({ severity: 'error', message: e.message,
        loc: binding.node?.value?.loc, binding: name });
    }
  }
  return changed;
}

module.exports = { expandDataBindings };
