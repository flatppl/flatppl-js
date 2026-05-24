'use strict';

// AST node constructors for FlatPPL.
// Every node carries `loc: { start: {line, col}, end: {line, col} }` (0-based).

function loc(startLine: number, startCol: number, endLine: number, endCol: number) {
  return { start: { line: startLine, col: startCol }, end: { line: endLine, col: endCol } };
}

// Loc marker for nodes that didn't come from the parser — produced by the
// engine itself (rewriters, lowerings). The `synthetic: true` flag lets
// downstream code distinguish provenance; `source` is an optional tag
// pointing back to whatever produced the node (e.g. a binding name).
// `start`/`end` carry sentinel positions so any consumer that expects a
// well-formed loc still works.
function synthLoc(source: any) {
  return {
    start: { line: -1, col: -1 },
    end: { line: -1, col: -1 },
    synthetic: true,
    source: source || null,
  };
}

// --- Statements ---

function Program(body: any, comments: any) {
  return { type: 'Program', body, comments: comments || [] };
}

function AssignStatement(names: any, value: any, loc: any) {
  return { type: 'AssignStatement', names, value, loc };
}

function ErrorStatement(text: any, loc: any) {
  return { type: 'ErrorStatement', text, loc };
}

function Comment(text: any, loc: any) {
  return { type: 'Comment', text, loc };
}

// --- Expressions ---

function Identifier(name: any, loc: any) {
  return { type: 'Identifier', name, loc };
}

function NumberLiteral(value: any, raw: any, loc: any) {
  return { type: 'NumberLiteral', value, raw, loc };
}

function StringLiteral(value: any, raw: any, loc: any) {
  return { type: 'StringLiteral', value, raw, loc };
}

function BoolLiteral(value: any, loc: any) {
  return { type: 'BoolLiteral', value, loc };
}

function ConstantRef(name: any, loc: any) {
  return { type: 'ConstantRef', name, loc };
}

function SetRef(name: any, loc: any) {
  return { type: 'SetRef', name, loc };
}

function Placeholder(name: any, loc: any) {
  return { type: 'Placeholder', name, loc };
}

function Hole(loc: any) {
  return { type: 'Hole', loc };
}

// AxisRef: `.name` axis label used by `aggregate` (spec §05 Axis names).
// Distinguished from FieldAccess (which is `.name` AFTER a postfix-able
// expression — those have a parent object). An AxisRef appears at the
// start of a Primary; it carries no parent.
function AxisRef(name: any, loc: any) {
  return { type: 'AxisRef', name, loc };
}

function ArrayLiteral(elements: any, loc: any) {
  return { type: 'ArrayLiteral', elements, loc };
}

function TupleLiteral(elements: any, loc: any) {
  return { type: 'TupleLiteral', elements, loc };
}

function BinaryExpr(op: any, left: any, right: any, loc: any) {
  return { type: 'BinaryExpr', op, left, right, loc };
}

function UnaryExpr(op: any, operand: any, loc: any) {
  return { type: 'UnaryExpr', op, operand, loc };
}

function CallExpr(callee: any, args: any, loc: any) {
  return { type: 'CallExpr', callee, args, loc };
}

function IndexExpr(object: any, indices: any, loc: any, indexOp?: any) {
  // indexOp picks the lowering target for `xs[i]`. FlatPPL/FlatPPJ
  // use 'get' (1-based); FlatPPY uses 'get0' (0-based). Defaults to
  // 'get' so any code constructing an IndexExpr without specifying
  // continues to lower as before.
  return { type: 'IndexExpr', object, indices, loc, indexOp: indexOp || 'get' };
}

function FieldAccess(object: any, field: any, loc: any) {
  return { type: 'FieldAccess', object, field, loc };
}

function KeywordArg(name: any, value: any, loc: any) {
  return { type: 'KeywordArg', name, value, loc };
}

function SliceAll(loc: any) {
  return { type: 'SliceAll', loc };
}

module.exports = {
  loc, synthLoc,
  Program, AssignStatement, ErrorStatement, Comment,
  Identifier, NumberLiteral, StringLiteral, BoolLiteral,
  ConstantRef, SetRef, Placeholder, Hole, AxisRef,
  ArrayLiteral, TupleLiteral, BinaryExpr, UnaryExpr,
  CallExpr, IndexExpr, FieldAccess,
  KeywordArg, SliceAll,
};
