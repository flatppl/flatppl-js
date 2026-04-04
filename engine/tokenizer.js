'use strict';

// Token types
const T = {
  IDENT: 'IDENT',
  NUMBER: 'NUMBER',
  STRING: 'STRING',
  PLACEHOLDER: 'PLACEHOLDER', // _name_
  HOLE: 'HOLE',               // bare _

  LPAREN: 'LPAREN', RPAREN: 'RPAREN',
  LBRACKET: 'LBRACKET', RBRACKET: 'RBRACKET',
  COMMA: 'COMMA',
  DOT: 'DOT',
  COLON: 'COLON',
  EQUALS: 'EQUALS',     // =
  EQEQ: 'EQEQ',        // ==
  NEQ: 'NEQ',           // !=
  LT: 'LT', GT: 'GT',
  LTE: 'LTE', GTE: 'GTE',
  PLUS: 'PLUS', MINUS: 'MINUS',
  STAR: 'STAR', SLASH: 'SLASH',

  NEWLINE: 'NEWLINE',
  COMMENT: 'COMMENT',
  EOF: 'EOF',
};

function token(type, value, startLine, startCol, endLine, endCol) {
  return {
    type, value,
    loc: { start: { line: startLine, col: startCol }, end: { line: endLine, col: endCol } },
  };
}

function isAlpha(ch) { return /[a-zA-Z]/.test(ch); }
function isDigit(ch) { return /[0-9]/.test(ch); }
function isAlphaNum(ch) { return /[a-zA-Z0-9]/.test(ch); }
function isIdentStart(ch) { return isAlpha(ch) || ch === '_'; }
function isIdentChar(ch) { return isAlphaNum(ch) || ch === '_'; }
function isWhitespace(ch) { return ch === ' ' || ch === '\t' || ch === '\r'; }

/**
 * Tokenize FlatPPL source text into an array of tokens.
 * Returns { tokens: Token[], diagnostics: Diagnostic[] }.
 */
function tokenize(source) {
  const tokens = [];
  const diagnostics = [];
  let pos = 0;
  let line = 0;
  let col = 0;
  let depth = 0; // paren/bracket nesting depth

  function peek(offset) { return source[pos + (offset || 0)] || ''; }
  function advance() {
    const ch = source[pos++];
    if (ch === '\n') { line++; col = 0; } else { col++; }
    return ch;
  }
  function at(offset) { return pos + (offset || 0) < source.length ? source[pos + (offset || 0)] : ''; }

  while (pos < source.length) {
    const startLine = line, startCol = col;
    const ch = at();

    // Whitespace (not newline)
    if (isWhitespace(ch)) {
      advance();
      continue;
    }

    // Newline
    if (ch === '\n') {
      advance();
      if (depth === 0) {
        tokens.push(token(T.NEWLINE, '\n', startLine, startCol, line, col));
      }
      continue;
    }

    // Comment
    if (ch === '#') {
      let text = '';
      while (pos < source.length && at() !== '\n') {
        text += advance();
      }
      tokens.push(token(T.COMMENT, text, startLine, startCol, line, col));
      continue;
    }

    // String literal
    if (ch === '"') {
      advance(); // opening quote
      let value = '';
      let raw = '"';
      while (pos < source.length && at() !== '"' && at() !== '\n') {
        if (at() === '\\' && pos + 1 < source.length) {
          raw += advance(); // backslash
          raw += advance(); // escaped char
          const esc = raw[raw.length - 1];
          if (esc === 'n') value += '\n';
          else if (esc === 't') value += '\t';
          else if (esc === '\\') value += '\\';
          else if (esc === '"') value += '"';
          else value += esc;
        } else {
          const c = advance();
          raw += c;
          value += c;
        }
      }
      if (at() === '"') {
        raw += advance(); // closing quote
      } else {
        diagnostics.push({ severity: 'error', message: 'Unterminated string literal', loc: { start: { line: startLine, col: startCol }, end: { line, col } } });
      }
      tokens.push(token(T.STRING, value, startLine, startCol, line, col));
      continue;
    }

    // Number literal
    if (isDigit(ch) || (ch === '.' && isDigit(at(1)))) {
      let num = '';
      while (pos < source.length && isDigit(at())) num += advance();
      if (at() === '.' && at(1) !== '.') {
        num += advance(); // dot
        while (pos < source.length && isDigit(at())) num += advance();
      }
      if (at() === 'e' || at() === 'E') {
        num += advance(); // e/E
        if (at() === '+' || at() === '-') num += advance();
        while (pos < source.length && isDigit(at())) num += advance();
      }
      tokens.push(token(T.NUMBER, num, startLine, startCol, line, col));
      continue;
    }

    // Identifier, placeholder, hole, or keyword
    if (isIdentStart(ch)) {
      let ident = '';
      const iStart = pos;
      while (pos < source.length && isIdentChar(at())) ident += advance();

      // Distinguish: bare _ (hole) vs _name_ (placeholder) vs regular ident
      if (ident === '_') {
        tokens.push(token(T.HOLE, '_', startLine, startCol, line, col));
      } else if (ident.length >= 3 && ident[0] === '_' && ident[ident.length - 1] === '_'
                 && isAlpha(ident[1])) {
        // _name_ pattern — placeholder
        const innerName = ident.slice(1, -1);
        tokens.push(token(T.PLACEHOLDER, innerName, startLine, startCol, line, col));
      } else {
        tokens.push(token(T.IDENT, ident, startLine, startCol, line, col));
      }
      continue;
    }

    // Two-character operators
    if (ch === '=' && at(1) === '=') {
      advance(); advance();
      tokens.push(token(T.EQEQ, '==', startLine, startCol, line, col));
      continue;
    }
    if (ch === '!' && at(1) === '=') {
      advance(); advance();
      tokens.push(token(T.NEQ, '!=', startLine, startCol, line, col));
      continue;
    }
    if (ch === '<' && at(1) === '=') {
      advance(); advance();
      tokens.push(token(T.LTE, '<=', startLine, startCol, line, col));
      continue;
    }
    if (ch === '>' && at(1) === '=') {
      advance(); advance();
      tokens.push(token(T.GTE, '>=', startLine, startCol, line, col));
      continue;
    }

    // Single-character tokens
    const singleMap = {
      '(': T.LPAREN, ')': T.RPAREN,
      '[': T.LBRACKET, ']': T.RBRACKET,
      ',': T.COMMA, '.': T.DOT, ':': T.COLON,
      '=': T.EQUALS,
      '<': T.LT, '>': T.GT,
      '+': T.PLUS, '-': T.MINUS,
      '*': T.STAR, '/': T.SLASH,
    };
    if (singleMap[ch]) {
      const tt = singleMap[ch];
      advance();
      if (tt === T.LPAREN || tt === T.LBRACKET) depth++;
      else if (tt === T.RPAREN || tt === T.RBRACKET) depth = Math.max(0, depth - 1);
      tokens.push(token(tt, ch, startLine, startCol, line, col));
      continue;
    }

    // Unknown character
    const uch = advance();
    diagnostics.push({
      severity: 'error',
      message: `Unexpected character '${uch}'`,
      loc: { start: { line: startLine, col: startCol }, end: { line, col } },
    });
  }

  tokens.push(token(T.EOF, '', line, col, line, col));
  return { tokens, diagnostics };
}

module.exports = { T, tokenize };
