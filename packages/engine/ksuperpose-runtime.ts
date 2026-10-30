'use strict';

// Value-aware direct ksuperpose expansion. Preserve operand expressions and
// use the same weighted/superpose derivations as the static AST expansion.
const AST = require('./ast.ts');

function _isSelfRef(ir: any): boolean {
  return !!(ir && ir.kind === 'ref' && ir.ns === 'self');
}

function _litInt(i: number): any {
  return { kind: 'lit', value: i, numType: 'integer' };
}

function _getCell(ir: any, shape: number[], index: number[]): any {
  if (!shape.length) return ir;
  return { kind: 'call', op: 'get', args: [ir,
    ...require('./ksuperpose-expand.ts').componentIndex(shape, index).map(_litInt)] };
}

// A synthesised binding. It carries no AST: this pass sets the derivation
// itself, so nothing downstream re-classifies it from source. `phase: null`
// keeps it out of the fixed-phase dead-end sweep, which would otherwise
// report every component as an engine gap.
function _synthBinding(name: string, ir: any, deps: string[], loc: any): any {
  return {
    name,
    names: [name],
    line: 0,
    rhs: null,
    type: 'call',
    deps,
    callDeps: [],
    bodyDeps: [],
    paramSourceDeps: [],
    node: AST.AssignStatement([name], null, loc),
    nameLoc: loc,
    phase: null,
    inferredType: null,
    ir,
  };
}

// Infer outer shape first; only ask the fixed resolver when shape inference
// left a dimension dynamic. Keep the operand IR, never freeze its values.
function _classifyArg(ir: any, bindings: any, fixedValues: any, refuse: any): any {
  if (ir?.kind === 'lit') return { shape: [] };
  if (ir?.kind === 'call' && ir.op === 'vector') return { shape: [ir.args?.length || 0] };
  const t = ir?.meta?.type || (_isSelfRef(ir) && bindings.get(ir.name)?.inferredType);
  const cls = require('./ksuperpose-expand.ts').classifyType(t);
  if (cls?.invalid) { refuse('ksuperpose: records and tuples are not allowed as broadcast operands (spec §04)'); return null; }
  if (cls?.shape && cls.shape.every((n: any) => typeof n === 'number')) return cls;
  if (_isSelfRef(ir) && fixedValues.has(ir.name)) {
    const slot = require('./broadcast-shape.ts').classifyAxisStructure(fixedValues.get(ir.name));
    return { shape: slot.coll ? slot.outerShape : [], table: slot.kind === 'table' };
  }
  refuse('ksuperpose: operand has no statically-known shape and no resolvable value; bind it to a name with a known shape');
  return null;
}

// The distribution op a `ksuperpose` kernel argument denotes, or null.
//
// The synthesised component is a distribution CALL, so the kernel must be a
// builtin measure kernel. A reified kernel (`m -> Normal(mu = m, …)`, a
// `kernelof`, a user function) would need its body substituted per component,
// which is what the static rewrite gets for free by rewriting the AST before
// lowering — hence the remedy in the refusal is a statically-known N, not a
// different kernel.
function _kernelOp(kernelIR: any, bindings: any): string | null {
  if (!_isSelfRef(kernelIR)) return null;
  const name = kernelIR.name;
  // A ref to a real binding is a user-defined or reified kernel, never a
  // builtin: builtin distribution names lower to a self-ref with no binding.
  if (bindings.has(name)) return null;
  const orchestrator = require('./orchestrator.ts');
  const known = orchestrator.SAMPLEABLE_DISTRIBUTIONS;
  return (known && known.has(name)) ? name : null;
}

// Expand one binding whose IR is a direct `ksuperpose` call with a runtime
// N. Mutates `bindings` and `derivations`; pushes at most one diagnostic.
// Returns true when the binding was expanded.
function _expandOne(
  name: string, binding: any, call: any,
  bindings: any, derivations: any, fixedValues: any, diagnostics: any[],
): boolean {
  const loc = (binding.node && binding.node.loc) || null;
  let refused = false;
  const refuse = (message: string) => {
    if (refused) return;
    refused = true;
    diagnostics.push({ severity: 'error', message, loc });
  };

  const callArgs = call.args || [];
  if (callArgs.length < 2) return false;
  const kernelIR = callArgs[0];
  const weightsIR = callArgs[1];
  const wName = _isSelfRef(weightsIR) ? weightsIR.name : null;
  const kernelOp = _kernelOp(kernelIR, bindings);

  const specs: any[] = [];
  const posArgs = callArgs.slice(2);
  const kwargs = call.kwargs || {};
  const names = require('./types.ts').builtinInterface(kernelOp)?.inputs.map((p: any) => p.name) || [];
  const mixed = Object.keys(kwargs).length > 0;
  for (const [i, ir] of [weightsIR, ...posArgs].entries()) {
    const cls = _classifyArg(ir, bindings, fixedValues, refuse);
    if (!cls) return false;
    specs.push({ key: i && mixed ? names[i - 1] : null, ir, cls });
  }
  for (const k of Object.keys(kwargs)) {
    if (names.includes(k) && names.indexOf(k) < posArgs.length) { refuse('ksuperpose: duplicate kernel input ' + k); return false; }
    const cls = _classifyArg(kwargs[k], bindings, fixedValues, refuse);
    if (!cls) return false;
    specs.push({ key: k, ir: kwargs[k], cls });
  }
  let shape: number[];
  try { shape = require('./broadcast-shape.ts').mergeBroadcastShapes(specs.map(s => s.cls.shape)); }
  catch (e: any) { refuse('ksuperpose: ' + e.message); return false; }
  const N = shape.reduce((a, b) => a * b, 1);
  if (N === 0) {
    derivations[name] = { kind: 'superpose', fromNames: [] };
    return true;
  }
  if (kernelOp == null) {
    refuse('ksuperpose: with a runtime component count the mixture is built '
      + 'as a call to a builtin measure kernel, so a reified or user-defined '
      + 'kernel cannot be used here (spec §06 allows it, and the '
      + 'statically-known-N spelling supports it). Give the weight vector a '
      + 'statically-known length.');
    return false;
  }

  const index = shape.map(() => 0);
  const weightedNames: string[] = [];
  for (let i = 1; i <= N; i++) {
    const compIR: any = { kind: 'call', op: kernelOp };
    const cArgs: any[] = [];
    const cKwargs: Record<string, any> = {};
    let hasKw = false;
    for (const s of specs.slice(1)) {
      const value = _getCell(s.ir, s.cls.shape, index);
      if (s.key == null) cArgs.push(value);
      else { cKwargs[s.key] = value; hasKw = true; }
    }
    if (posArgs.length === 1 && !mixed && specs[1].cls.table
        && !require('./types.ts').acceptsPositional(require('./types.ts').builtinInterface(kernelOp), 1)) {
      const columns = specs[1].cls.columns || bindings.get(posArgs[0].name)?.inferredType?.columns;
      for (const key of Object.keys(columns || {})) {
        cKwargs[key] = { kind: 'call', op: 'get_field', args: [cArgs[0], { kind: 'lit', value: key }] };
      }
      cArgs.length = 0;
      hasKw = true;
    }
    if (cArgs.length > 0) compIR.args = cArgs;
    if (hasKw) compIR.kwargs = cKwargs;

    const compName = `__ksrt_${name}_c${i}`;
    const wtName = `__ksrt_${name}_w${i}`;
    const weightIR = _getCell(weightsIR, specs[0].cls.shape, index);
    const compDeps = specs.filter((s: any) => _isSelfRef(s.ir)).map((s: any) => s.ir.name);
    bindings.set(compName, _synthBinding(compName, compIR, compDeps, loc));
    bindings.set(wtName, _synthBinding(
      wtName, { kind: 'call', op: 'weighted', args: [weightIR, { kind: 'ref', ns: 'self', name: compName }] },
      [compName, ...(wName ? [wName] : [])], loc));
    derivations[compName] = { kind: 'sample', distIR: compIR };
    derivations[wtName] = { kind: 'weighted', from: compName, weightIR };
    weightedNames.push(wtName);
    for (let a = shape.length - 1; a >= 0; a--) {
      if (++index[a] < shape[a]) break;
      index[a] = 0;
    }
  }
  derivations[name] = { kind: 'superpose', fromNames: weightedNames };
  return true;
}

// Expand every direct ksuperpose call the static rewrite left behind.
//
// Returns `{ changed, refused }`. `refused` names the mixture bindings that
// got a located refusal; the caller suppresses the generic fixed-phase
// dead-end message for them and for everything above them, which would
// otherwise follow a precise diagnosis with "this is an engine gap".
function expandRuntimeKsuperpose(
  bindings: any, derivations: any, fixedValues: any, diagnostics: any[],
): { changed: boolean; refused: Set<string> } {
  // Snapshot the names: the loop adds bindings, and a synthesised one is never
  // itself a direct ksuperpose call.
  const names: string[] = Array.from(bindings.keys()).map(String);
  let changed = false;
  const refused = new Set<string>();
  for (const name of names) {
    if (derivations[name]) continue;
    const binding = bindings.get(name);
    if (!binding || !binding.ir) continue;
    const call = binding.ir.kind === 'call' && binding.ir.op === 'ksuperpose' ? binding.ir : null;
    if (!call) continue;
    const before = diagnostics.length;
    if (_expandOne(name, binding, call, bindings, derivations,
      fixedValues, diagnostics)) {
      changed = true;
    } else if (diagnostics.length > before) {
      refused.add(name);
    }
  }
  return { changed, refused };
}

module.exports = { expandRuntimeKsuperpose };
