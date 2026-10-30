'use strict';

// Surface binding shared by broadcast and ksuperpose. Per-cell ordinary calls
// use one argument form, even when the special operation uses MixedArgs.
const AST = require('./ast.ts');

// MixedArgs belongs to broadcast/ksuperpose, not to the per-cell ordinary
// call. Resolve positional inputs onto their declared names before lowering.
function callableParamNames(kernel: any, bindings: any): string[] | null {
  if (kernel?.type === 'Identifier') {
    const t = bindings.get(kernel.name)?.inferredType;
    if (t?.inputs) return t.inputs.map((p: any) => p.name);
    const iface = require('./types.ts').builtinInterface(kernel.name);
    if (iface?.inputs) return iface.inputs.map((p: any) => p.name);
  }
  if (kernel?.type === 'CallExpr' && ['functionof', 'kernelof'].includes(kernel.callee?.name)) {
    return kernel.args.slice(1).map((arg: any) => arg.name);
  }
  return null;
}

function bindMixedArgs(args: any[], kernel: any, bindings: any): any[] {
  if (!args.some(a => a.type === 'KeywordArg') || args.every(a => a.type === 'KeywordArg')) return args;
  const names = callableParamNames(kernel, bindings);
  if (!names) return args;
  let i = 0;
  const seen = new Set<string>();
  return args.map(a => {
    const name = a.type === 'KeywordArg' ? a.name : names[i++];
    if (!name || seen.has(name)) throw new Error('duplicate or excess callable argument');
    seen.add(name);
    return a.type === 'KeywordArg' ? a : AST.KeywordArg(name, a, a.loc);
  });
}

module.exports = { bindMixedArgs };
