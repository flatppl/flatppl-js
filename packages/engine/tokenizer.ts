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
  SEMI: 'SEMI',         // ;  (FlatPPJ semicolon kwargs)
  DOT: 'DOT',
  COLON: 'COLON',
  EQUALS: 'EQUALS',     // =
  TILDE: 'TILDE',       // ~  (FlatPPL/FlatPPJ tilde bindings)
  EQEQ: 'EQEQ',        // ==
  NEQ: 'NEQ',           // !=
  LT: 'LT', GT: 'GT',
  LTE: 'LTE', GTE: 'GTE',
  PLUS: 'PLUS', MINUS: 'MINUS',
  STAR: 'STAR', SLASH: 'SLASH',
  CARET: 'CARET',       // ^  (FlatPPL/FlatPPJ exponentiation)
  AMPAMP: 'AMPAMP',     // && (FlatPPL/FlatPPJ logical AND)
  PIPEPIPE: 'PIPEPIPE', // || (FlatPPL/FlatPPJ logical OR)
  BANG: 'BANG',         // !  (FlatPPL/FlatPPJ logical NOT)

  NEWLINE: 'NEWLINE',
  COMMENT: 'COMMENT',
  EOF: 'EOF',
};

function token(type, value, startLine, startCol, endLine, endCol): any {
  return {
    type, value,
    loc: { start: { line: startLine, col: startCol }, end: { line: endLine, col: endCol } },
  };
}

function isAlpha(ch) { return /[a-zA-Z]/.test(ch); }
function isDigit(ch) { return /[0-9]/.test(ch); }
function isHexDigit(ch) { return /[0-9a-fA-F]/.test(ch); }
function isAlphaNum(ch) { return /[a-zA-Z0-9]/.test(ch); }
function isIdentStart(ch) { return isAlpha(ch) || ch === '_'; }
function isIdentChar(ch) { return isAlphaNum(ch) || ch === '_'; }
function isWhitespace(ch) { return ch === ' ' || ch === '\t' || ch === '\r'; }

/**
 * Tokenize FlatPPL source text into an array of tokens.
 * Returns { tokens: Token[], diagnostics: Diagnostic[] }.
 *
 * `variant` is the surface-syntax variant (see ./variants). The base
 * tokenizer is variant-agnostic; per-variant token additions live in
 * later commits in this series. `variant` is accepted (and ignored)
 * starting now so call sites can pass it without churn.
 */
function tokenize(source, variant) {  // eslint-disable-line no-unused-vars
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
  function at(offset?) { return pos + (offset || 0) < source.length ? source[pos + (offset || 0)] : ''; }

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
          const escStartLine = line, escStartCol = col;
          raw += advance(); // backslash
          raw += advance(); // escaped char
          const esc = raw[raw.length - 1];
          if (esc === 'n') value += '\n';
          else if (esc === 't') value += '\t';
          else if (esc === 'r') value += '\r';
          else if (esc === '0') value += '\0';
          else if (esc === '\\') value += '\\';
          else if (esc === '"') value += '"';
          else {
            diagnostics.push({
              severity: 'error',
              message: `Invalid escape sequence '\\${esc}'`,
              loc: { start: { line: escStartLine, col: escStartCol }, end: { line, col } },
            });
            value += esc;
          }
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

    // Number literal (decimal, hex, with optional underscores between digits)
    if (isDigit(ch) || (ch === '.' && isDigit(at(1)))) {
      let num = '';
      // Hex: 0x HexDigit (_? HexDigit)*  — at least one hex digit
      // required. Lowercase `0x` only, per spec §05 HexIntLit (matches
      // Julia/Rust; the uppercase `0X` C-family form is intentionally
      // not canonical FlatPPL).
      if (ch === '0' && at(1) === 'x') {
        num += advance(); // 0
        num += advance(); // x
        if (!isHexDigit(at())) {
          diagnostics.push({
            severity: 'error',
            message: 'Hex literal requires at least one hex digit after 0x',
            loc: { start: { line: startLine, col: startCol }, end: { line, col } },
          });
        }
        while (pos < source.length && (isHexDigit(at()) || (at() === '_' && isHexDigit(at(1))))) {
          num += advance();
        }
        tokens.push(token(T.NUMBER, num, startLine, startCol, line, col));
        continue;
      }
      // Decimal integer part
      while (pos < source.length && (isDigit(at()) || (at() === '_' && isDigit(at(1))))) {
        num += advance();
      }
      // Fractional part
      if (at() === '.' && at(1) !== '.') {
        num += advance(); // dot
        while (pos < source.length && (isDigit(at()) || (at() === '_' && isDigit(at(1))))) {
          num += advance();
        }
      }
      // Exponent
      if (at() === 'e' || at() === 'E') {
        num += advance(); // e/E
        if (at() === '+' || at() === '-') num += advance();
        while (pos < source.length && (isDigit(at()) || (at() === '_' && isDigit(at(1))))) {
          num += advance();
        }
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

    // Two-character operators. Order matters: check `==` before `=`,
    // `!=` before `!`, `<=` before `<`, `>=` before `>`, `&&` before
    // any bitwise-and (which FlatPPL doesn't have but we'd want to
    // reject cleanly), and `||` before any bitwise-or.
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
    if (ch === '&' && at(1) === '&') {
      advance(); advance();
      tokens.push(token(T.AMPAMP, '&&', startLine, startCol, line, col));
      continue;
    }
    if (ch === '|' && at(1) === '|') {
      advance(); advance();
      tokens.push(token(T.PIPEPIPE, '||', startLine, startCol, line, col));
      continue;
    }

    // Dot-prefixed (broadcast) operators, spec §05 Broadcasting syntax:
    // `.+ .- .* ./ .^ .< .> .<= .>= .== .!= .&& .|| .!`. Each is a
    // single token recognized by maximal munch (longest match — the
    // 3-char forms are tried before their 2-char prefixes). They carry
    // the SAME token type and value as the plain operator plus a
    // `dotted:true` marker, so the precedence parser matches them with
    // zero change and only the desugaring branches inspect the marker.
    //
    // Reaching here means `ch === '.'` was NOT consumed by the number
    // branch (which already ate `.`+digit, so `1./x` munched `1.` as
    // the real literal `1.0` — the Julia rule, spec §05). `.(`
    // (dot-call) and `.name` (field access) are left as a bare DOT for
    // the parser, since `(` and identifier-starts are not operator
    // chars below.
    if (ch === '.') {
      const DOT_OPS_3 = {
        '.==': T.EQEQ, '.!=': T.NEQ, '.<=': T.LTE, '.>=': T.GTE,
        '.&&': T.AMPAMP, '.||': T.PIPEPIPE,
      };
      const DOT_OPS_2 = {
        '.<': T.LT, '.>': T.GT, '.+': T.PLUS, '.-': T.MINUS,
        '.*': T.STAR, './': T.SLASH, '.^': T.CARET, '.!': T.BANG,
      };
      const three = ch + at(1) + at(2);
      const two = ch + at(1);
      let lexeme = null, ttype = null;
      if (DOT_OPS_3[three]) { lexeme = three; ttype = DOT_OPS_3[three]; }
      else if (DOT_OPS_2[two]) { lexeme = two; ttype = DOT_OPS_2[two]; }
      if (lexeme) {
        for (let k = 0; k < lexeme.length; k++) advance();
        const tok = token(ttype, lexeme.slice(1), startLine, startCol, line, col);
        tok.dotted = true;
        tokens.push(tok);
        continue;
      }
    }

    // Single-character tokens
    const singleMap = {
      '(': T.LPAREN, ')': T.RPAREN,
      '[': T.LBRACKET, ']': T.RBRACKET,
      ',': T.COMMA, ';': T.SEMI,
      '.': T.DOT, ':': T.COLON,
      '=': T.EQUALS, '~': T.TILDE,
      '<': T.LT, '>': T.GT,
      '+': T.PLUS, '-': T.MINUS,
      '*': T.STAR, '/': T.SLASH,
      '^': T.CARET, '!': T.BANG,
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
