'use strict';

// §04/§07: dependency discovery is shared by compilation and portable export.
// Source spans belong to arguments, so export never rewrites comments or names.
const { tokenize } = require('./tokenizer.ts');
const { parse } = require('./parser.ts');
import type { LoadDependency } from './bundle-types';

/** Resolve fixed string aliases without evaluating a model (spec §04). */
function fixedStringAliases(ast: any): Record<string, string> {
  const definitions = new Map<string, any>();
  for (const statement of ast.body || []) {
    if (statement.type === 'AssignStatement' && statement.names?.length === 1) {
      definitions.set(statement.names[0].name, statement.value);
    }
  }
  const result: Record<string, string> = Object.create(null);
  const visiting = new Set<string>();
  function resolve(name: string): string | undefined {
    if (Object.prototype.hasOwnProperty.call(result, name)) return result[name];
    if (visiting.has(name)) return undefined;
    visiting.add(name);
    const value = definitions.get(name);
    const text = value?.type === 'StringLiteral' ? value.value
      : value?.type === 'Identifier' ? resolve(value.name) : undefined;
    visiting.delete(name);
    if (typeof text === 'string') result[name] = text;
    return text;
  }
  for (const name of definitions.keys()) resolve(name);
  return result;
}

/** Find builtin loads, retaining the original fixed argument and its span. */
function loadDependencies(ast: any): LoadDependency[] {
  const aliases = fixedStringAliases(ast);
  const shadowed = new Set<string>();
  for (const statement of ast.body || []) {
    if (statement.type === 'AssignStatement') {
      for (const name of statement.names || []) shadowed.add(name.name);
    }
  }
  const result: LoadDependency[] = [];
  function visit(node: any): void {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { for (const child of node) visit(child); return; }
    if (node.type === 'CallExpr') {
      const callee = node.callee;
      const name = callee?.type === 'Identifier' && (!shadowed.has(callee.name) || node.builtin)
        ? callee.name : callee?.type === 'FieldAccess' && callee.object?.type === 'Identifier'
          && callee.object.name === 'base' ? callee.field : null;
      if (name === 'load_module' || name === 'load_data') {
        // load_module keywords substitute inputs of the loaded module, even
        // when an input is named source. Only its positional argument locates it.
        const namedSource = name === 'load_data'
          ? node.args?.find((a: any) => a.type === 'KeywordArg' && a.name === 'source')?.value : undefined;
        const argument = namedSource || node.args?.find((a: any) => a.type !== 'KeywordArg');
        result.push({
          source: argument?.type === 'StringLiteral' ? argument.value
            : argument?.type === 'Identifier' ? aliases[argument.name] ?? null : null,
          kind: name === 'load_module' ? 'modules' : 'data',
          loc: node.loc, argumentLoc: argument?.loc,
        });
      }
    }
    for (const key of Object.keys(node)) if (key !== 'loc') visit(node[key]);
  }
  visit(ast);
  return result;
}

/** Parse dependencies. Strict export refuses syntax errors and unknown paths. */
function sourceDependencies(source: string, strict = false): LoadDependency[] {
  const tokenized = tokenize(source);
  const parsed = parse(tokenized.tokens);
  if (strict) {
    const error = [...tokenized.diagnostics, ...parsed.diagnostics].find((d: any) => d.severity === 'error');
    if (error) throw new Error(`Cannot bundle invalid FlatPPL: ${error.message}`);
  }
  const result = loadDependencies(parsed.ast);
  if (strict && result.some(dep => dep.source === null)) {
    throw new Error('Cannot bundle a load_module/load_data source that is not a fixed string or string alias');
  }
  return result;
}

module.exports = { fixedStringAliases, loadDependencies, sourceDependencies };
