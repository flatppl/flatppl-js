'use strict';

// AST node constructors for FlatPPL.
// Every node carries `loc: { start: {line, col}, end: {line, col} }` (0-based).

function loc(startLine, startCol, endLine, endCol) {
  return { start: { line: startLine, col: startCol }, end: { line: endLine, col: endCol } };
}

// --- Statements ---

function Program(body, comments) {
  return { type: 'Program', body, comments: comments || [] };
}

function AssignStatement(names, value, loc) {
  return { type: 'AssignStatement', names, value, loc };
}

function ErrorStatement(text, loc) {
  return { type: 'ErrorStatement', text, loc };
}

function Comment(text, loc) {
  return { type: 'Comment', text, loc };
}

// --- Expressions ---

function Identifier(name, loc) {
  return { type: 'Identifier', name, loc };
}

function NumberLiteral(value, raw, loc) {
  return { type: 'NumberLiteral', value, raw, loc };
}

function StringLiteral(value, raw, loc) {
  return { type: 'StringLiteral', value, raw, loc };
}

function BoolLiteral(value, loc) {
  return { type: 'BoolLiteral', value, loc };
}

function ConstantRef(name, loc) {
  return { type: 'ConstantRef', name, loc };
}

function SetRef(name, loc) {
  return { type: 'SetRef', name, loc };
}

function Placeholder(name, loc) {
  return { type: 'Placeholder', name, loc };
}

function Hole(loc) {
  return { type: 'Hole', loc };
}

function ArrayLiteral(elements, loc) {
  return { type: 'ArrayLiteral', elements, loc };
}

function TupleLiteral(elements, loc) {
  return { type: 'TupleLiteral', elements, loc };
}

function BinaryExpr(op, left, right, loc) {
  return { type: 'BinaryExpr', op, left, right, loc };
}

function UnaryExpr(op, operand, loc) {
  return { type: 'UnaryExpr', op, operand, loc };
}

function CallExpr(callee, args, loc) {
  return { type: 'CallExpr', callee, args, loc };
}

function IndexExpr(object, indices, loc) {
  return { type: 'IndexExpr', object, indices, loc };
}

function FieldAccess(object, field, loc) {
  return { type: 'FieldAccess', object, field, loc };
}

function KeywordArg(name, value, loc) {
  return { type: 'KeywordArg', name, value, loc };
}

function SliceAll(loc) {
  return { type: 'SliceAll', loc };
}

module.exports = {
  loc,
  Program, AssignStatement, ErrorStatement, Comment,
  Identifier, NumberLiteral, StringLiteral, BoolLiteral,
  ConstantRef, SetRef, Placeholder, Hole,
  ArrayLiteral, TupleLiteral, BinaryExpr, UnaryExpr,
  CallExpr, IndexExpr, FieldAccess,
  KeywordArg, SliceAll,
};
