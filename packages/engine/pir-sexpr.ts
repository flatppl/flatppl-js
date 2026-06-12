'use strict';

// =====================================================================
// pir-sexpr.ts — FlatPIR S-expression printer & reader (spec §11)
// =====================================================================
//
// The engine's in-memory FlatPIR representation is JSON-shaped (see
// pir.ts and lower.ts). Spec §11 defines a canonical S-expression
// surface form for FlatPIR (.flatpir files). This module bridges the
// two:
//
//   toSexpr(loweredModule, opts?) -> string
//     Prints a LoweredModule as a `(%module ...)` form. Indent-pretty
//     by default; pass { compact: true } for a single-line form.
//
//   fromSexpr(text) -> { module, diagnostics }
//     Parses a `.flatpir` text into a LoweredModule. Designed to
//     round-trip toSexpr output and to consume hand-written .flatpir.
//
// Not yet: cross-module loading; %meta READER reconstruction —
// emission is opt-in via `toSexpr(mod, { meta: true })` (see
// `_metaToSexpr` below), while the reader parses-and-skips %meta
// forms rather than converting them back into engine `meta.type`
// objects (consumers re-run inference today).
//
// Spec mapping (high level):
//   IR { kind: 'lit', value: x }      → bare literal (number / string /
//                                       true / false / null)
//   IR { kind: 'ref', ns, name }      → (%ref <ns> <name>)
//   IR { kind: 'hole' }               → _
//   IR { kind: 'const', name }        → bare symbol (pi, inf, im, etc.)
//   IR { kind: 'axis', name }         → (%axis <name>)
//   IR { kind: 'call', op, args,
//         kwargs, fields, assigns }   → (<op> [%meta]? args... %kwarg...
//                                              %field... %assign...)
//   IR { kind: 'call', op:'functionof',
//         params, paramKwargs,
//         paramSources, body }        → (functionof <output> %specinputs
//                                          ((<name> <ref>) ...))
//                                       | (functionof <output> %autoinputs
//                                          %deferred)
//                                       (spec §11 "Reified callables" —
//                                       fixed arity; see
//                                       _reificationToSexpr)
//   IR { kind: 'call', target:{ns,name},
//         args, kwargs }              → (%call (%ref <ns> <name>) args... %kwarg...)
//                                       (call to a user-defined callable)

const pir = require('./pir.ts');

// =====================================================================
// Printer
// =====================================================================

function toSexpr(mod: any, opts?: any) {
  opts = opts || {};
  const indent = opts.compact ? null : '  ';
  const lines: string[] = ['(%module'];
  // %public list
  const publics = Array.from(mod.publicSet || []);
  if (publics.length > 0) {
    lines.push((indent || '') + '(%public ' + publics.join(' ') + ')');
  }
  // bindings
  for (const [name, binding] of mod.bindings) {
    // %meta emission (opt-in; bare FlatPIR is the canonical pre-inference
    // shape — annotated output is for tooling export). The OUTERMOST call
    // of each binding's RHS carries the binding-level phase; inner calls
    // carry type-only slots (phase %deferred) where typeinfer annotated.
    const mopts = opts.meta
      ? { meta: true, outerPhase: binding.phase || null }
      : null;
    const rhsText = _exprToSexpr(binding.rhs, indent ? '  ' : '', mopts);
    // Optional `(%doc <markup> <line>...)` trailing sub-form (spec
    // §11 Documentation). Absent when the binding has no doc OR when
    // the doc has zero content lines (canonical form for undocumented).
    const docText = _docToSexpr(binding.doc);
    const bodyText = docText ? rhsText + ' ' + docText : rhsText;
    if (indent) {
      lines.push('');  // blank line between bindings
      lines.push(indent + '(%bind ' + name + ' ' + bodyText + ')');
    } else {
      lines.push('(%bind ' + name + ' ' + bodyText + ')');
    }
  }
  lines.push(')');
  return indent ? lines.join('\n') : lines.join(' ');
}

function _exprToSexpr(e: any, ind: string, mopts?: any): string {
  if (e == null) return 'null';
  if (typeof e !== 'object') return _atomToSexpr(e);
  switch (e.kind) {
    case 'lit':   return _atomToSexpr(e.value, e.numType);
    case 'hole':  return '_';
    case 'const': return String(e.name);
    case 'axis': {
      // Bare axis (no variance marker) → `(%axis <name>)`. Variance-
      // marked axes inside `metricsum(...)` lower to per-variance heads
      // per spec §11 (FlatPIR): `(%uaxis <name>)` for upper /
      // contravariant, `(%laxis <name>)` for lower / covariant.
      if (e.variance === 'upper') return '(%uaxis ' + e.name + ')';
      if (e.variance === 'lower') return '(%laxis ' + e.name + ')';
      return '(%axis ' + e.name + ')';
    }
    case 'ref':   return '(%ref ' + e.ns + ' ' + e.name + ')';
    case 'call':  return _callToSexpr(e, ind, mopts);
    default:
      throw new Error('pir-sexpr: unknown IR kind ' + JSON.stringify(e.kind));
  }
}

// Build the optional `(%doc <markup> <line>...)` sub-form. Returns
// the string OR null when the binding has no documented content.
// Per spec §11: `(%doc md)` with zero content lines is semantically
// the same as omitting the sub-form, and the absent form is canonical
// — so we elide both no-doc bindings and zero-content `%doc md` ones.
function _docToSexpr(doc: any): string | null {
  if (!doc) return null;
  const lines = Array.isArray(doc.lines) ? doc.lines : [];
  if (lines.length === 0) return null;
  const markup = typeof doc.markup === 'string' ? doc.markup : 'md';
  const parts = ['%doc', markup];
  for (const ln of lines) parts.push(JSON.stringify(ln));
  return '(' + parts.join(' ') + ')';
}

// ---------------------------------------------------------------------
// %meta emission (spec §11 "Type and phase annotations") — opt-in via
// toSexpr(mod, {meta: true}). The TYPE slot renders the engine type the
// typeinfer pass wrote on the call (`meta.type`); the PHASE slot carries
// the binding-level phase on the OUTERMOST call of each binding's RHS
// (the spec explicitly allows annotating only the outermost call) and
// %deferred on inner calls. Anything without a clean spec category
// (rngstate, type vars, set markers, callables with unknown inputs)
// downgrades to %deferred — semantically equivalent to omission, never
// wrong. A call where both slots would be %deferred emits NO %meta.

function _metaToSexpr(e: any, mopts: any): string | null {
  const t = e.meta && e.meta.type;
  const phase = (mopts && mopts.outerPhase) ? '%' + mopts.outerPhase : '%deferred';
  const tStr = _typeToSexpr(t);
  if (tStr === '%deferred' && phase === '%deferred') return null;
  return '(%meta ' + tStr + ' ' + phase + ')';
}

function _typeToSexpr(t: any): string {
  if (!t || t.kind === 'deferred') return '%deferred';
  switch (t.kind) {
    case 'failed':
      return '(%failed ' + JSON.stringify(String(t.reason || 'inference failed')) + ')';
    case 'any': return '%any';
    case 'scalar': return '(%scalar ' + t.prim + ')';
    case 'array': {
      const dims = Array.isArray(t.shape) ? t.shape : [];
      if (typeof t.rank !== 'number') return '%deferred';
      const shape = dims.map((d: any) =>
        (typeof d === 'number' ? String(d) : '%dynamic')).join(' ');
      return '(%array ' + t.rank + ' (' + shape + ') ' + _typeToSexpr(t.elem) + ')';
    }
    case 'tvector':
      return '(%tvector '
        + (typeof t.length === 'number' ? String(t.length) : '%dynamic')
        + ' ' + _typeToSexpr(t.elem) + ')';
    case 'record': {
      const fields = t.fields || {};
      const entries = Object.keys(fields).map(
        (k) => '(' + k + ' ' + _typeToSexpr(fields[k]) + ')');
      return '(%record ' + entries.join(' ') + ')';
    }
    case 'table': {
      const cols = t.columns || {};
      const entries = Object.keys(cols).map(
        (k) => '(' + k + ' ' + _typeToSexpr(cols[k]) + ')');
      const nrows = typeof t.nrows === 'number' ? String(t.nrows) : '%dynamic';
      return '(%table (%columns ' + entries.join(' ') + ') (%nrows ' + nrows + '))';
    }
    case 'tuple': {
      const elems = Array.isArray(t.elems) ? t.elems : [];
      return '(%tuple ' + elems.map(_typeToSexpr).join(' ') + ')';
    }
    case 'measure': return '(%measure (%domain ' + _typeToSexpr(t.domain) + '))';
    case 'function':
    case 'kernel': {
      const names = Array.isArray(t.inputs)
        ? t.inputs.map((i: any) => i && i.name).filter(Boolean) : [];
      if (names.length === 0) return '%deferred';   // inputs unknown
      return '(%' + t.kind + ' (%inputs ' + names.join(' ') + '))';
    }
    // likelihood: the spec category requires (%inputs …) AND (%obstype
    // <type>); the engine's likelihood type carries no obstype, so a
    // conformant emission isn't possible yet — downgrade.
    default: return '%deferred';
  }
}

function _atomToSexpr(v: any, numType?: string) {
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (v === null) return 'null';
  if (typeof v === 'string') return JSON.stringify(v);
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) {
      if (v === Infinity)  return 'inf';
      if (v === -Infinity) return '(neg inf)';
      return 'nan';
    }
    // Spec §11: a literal's type IS its lexical form on the wire
    // (`3` integer, `1.0` real). An integer-VALUED real lit must
    // therefore carry a decimal point — emitting `3.0` as `3` would
    // change its type. The reader reconstructs `numType` from the
    // same lexical rule, so the tag round-trips.
    if (numType === 'real' && Number.isInteger(v)) return v.toFixed(1);
    return String(v);
  }
  // Anything else (an object that's not an IR node) — emit as JSON
  // string for now; spec-strict callers can post-process.
  return JSON.stringify(v);
}

function _callToSexpr(e: any, ind: string, mopts?: any): string {
  // Reified callables take the spec §11 fixed-arity form, not the
  // generic call shape. (kernelof is accepted defensively; the lowerer
  // canonicalises it to functionof so it never appears internally.)
  if (!e.target && (e.op === 'functionof' || e.op === 'kernelof') && e.body) {
    return _reificationToSexpr(e, ind, mopts);
  }
  // Head shape (spec §11): a built-in call uses a bare op symbol as the
  // head; a user-defined-callable call (lower.ts emits `target:{ns,name}`
  // and no `op`) uses `(%call (%ref <ns> <name>) …)`.
  const parts: string[] = [];
  if (e.target) {
    parts.push('%call');
    parts.push('(%ref ' + e.target.ns + ' ' + e.target.name + ')');
  } else if (e.callee) {
    // Expression-headed user call (spec §11, b070d0a): the head is an
    // expression that must evaluate to a user-defined callable — e.g.
    // an inline reification `(%call (functionof …) 2.5)`. Ref heads
    // keep the `target` form above (unchanged on the wire).
    parts.push('%call');
    parts.push(_exprToSexpr(e.callee, ind, mopts ? { meta: true } : null));
  } else {
    parts.push(e.op);
  }
  // Optional `(%meta <type> <phase>)` immediately after the head (spec
  // §11 — calls and only calls). Skipped when both slots would be
  // %deferred (a missing annotation is equivalent).
  const inner = mopts ? { meta: true } : null;
  if (mopts) {
    const m = _metaToSexpr(e, mopts);
    if (m) parts.push(m);
  }
  // Positional args
  if (e.args) {
    for (const a of e.args) parts.push(_exprToSexpr(a, ind, inner));
  }
  // kwargs
  if (e.kwargs) {
    const names = Object.keys(e.kwargs);
    for (const k of names) {
      parts.push('(%kwarg ' + k + ' ' + _exprToSexpr(e.kwargs[k], ind, inner) + ')');
    }
  }
  // record / joint / cartprod / table fields
  if (e.fields) {
    for (const f of e.fields) {
      parts.push('(%field ' + f.name + ' ' + _exprToSexpr(f.value, ind, inner) + ')');
    }
  }
  // load_module / standard_module assigns
  if (e.assigns) {
    for (const a of e.assigns) {
      parts.push('(%assign ' + a.name + ' ' + _exprToSexpr(a.value, ind, inner) + ')');
    }
  }
  return '(' + parts.join(' ') + ')';
}

// ---------------------------------------------------------------------
// Reified callables (spec §11 "Reified callables") — fixed arity after
// the head: output expression, input-origin tag, input list.
//
//   (functionof <output> %specinputs ((<name> <ref>) ...))
//   (functionof <output> %autoinputs %deferred)
//
// Each input entry pairs the callable's CALL-NAME with the node it
// reifies (the old `(%params <node-names>)` form was lossy — it dropped
// the call-names). The engine-internal IR is spec-shaped (lower.ts:
// `%local` is placeholders-only; identifier-form boundary refs are plain
// self/module refs), so the body is emitted VERBATIM — no namespace
// translation at the serialization boundary.
//
// JS never emits a FILLED %autoinputs list — that is inference
// metadata (strippable like %meta), and the engine's auto-trace
// (parametric-leaf discovery for a boundary-less reification) is not
// implemented; `inferReification`'s inputs come from authored params
// only (TODO §11).
function _reificationToSexpr(e: any, ind: string, mopts?: any): string {
  const params: string[] = e.params || [];
  const paramKwargs: string[] = e.paramKwargs || [];
  const paramSources: any[] = e.paramSources || [];
  const isPlaceholderName = (p: string) => /^_.*_$/.test(p);
  const sourceOf = (i: number) => {
    const s = paramSources[i];
    if (s && s.kind) return s;
    // Defensive default mirroring lower.ts conventions: `_x_`-class
    // names are placeholders, everything else identifier bindings.
    return isPlaceholderName(params[i])
      ? { kind: 'placeholder', name: params[i] }
      : { kind: 'binding', name: params[i] };
  };
  const callNameOf = (i: number) => {
    if (paramKwargs[i]) return paramKwargs[i];
    // Placeholder `_x_` declares call-name `x`; identifier-form keeps
    // its own name (the lower.ts `functionof(e, a = a)` convention).
    const p = params[i];
    return isPlaceholderName(p) ? p.slice(1, -1) : p;
  };
  const parts: string[] = [e.op === 'kernelof' ? 'kernelof' : 'functionof'];
  // Optional %meta after the head (the spec's annotated example puts the
  // reification's own kernel/function annotation here).
  if (mopts) {
    const m = _metaToSexpr(e, mopts);
    if (m) parts.push(m);
  }
  parts.push(_exprToSexpr(e.body, ind, mopts ? { meta: true } : null));
  if (params.length === 0) {
    parts.push('%autoinputs', '%deferred');
  } else {
    const entries = params.map((p: string, i: number) => {
      const s = sourceOf(i);
      const ref = s.kind === 'placeholder'
        ? '(%ref %local ' + s.name + ')'
        : '(%ref ' + (s.ns || 'self') + ' ' + (s.name || p) + ')';
      return '(' + callNameOf(i) + ' ' + ref + ')';
    });
    parts.push('%specinputs', '(' + entries.join(' ') + ')');
  }
  return '(' + parts.join(' ') + ')';
}

// =====================================================================
// Reader
// =====================================================================
//
// Tokenizer that recognises:
//   ( )          — list parens
//   ;… <eol>     — line comments
//   "..."        — string literal (JSON escapes)
//   123 / 1.5e2  — number literal (int vs real by lexical form)
//   true false   — boolean literal
//   _            — hole
//   %name        — structural keyword (%module, %ref, %kwarg, %field,
//                  %assign, %bind, %public, %meta, %axis, %local,
//                  %specinputs, %autoinputs, %deferred, %fixed,
//                  %parameterized, %stochastic, …)
//   name         — bare symbol (built-in op name, set name, constant)
//
// The parser builds the same JSON-shaped IR that lower.ts emits, plus
// a list of bindings into a LoweredModule shape consumable by the rest
// of the engine.

function fromSexpr(text: any) {
  const tokens = _tokenize(text);
  const diagnostics: any[] = [];
  let pos = 0;
  function peek() { return tokens[pos]; }
  function eat() { return tokens[pos++]; }
  function eof() { return pos >= tokens.length; }

  function readForm(): any {
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unexpected EOF' });
      return null;
    }
    const t = peek();
    if (t.type === '(') return readList();
    if (t.type === ')') {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unexpected )' });
      eat();
      return null;
    }
    eat();
    return _atomFromToken(t);
  }

  function readList(): any {
    eat(); // '('
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unclosed (' });
      return null;
    }
    const head = peek();
    // Head of list determines the form.
    if (head.type === ')') {
      eat();
      return [];  // empty list — invalid in spec, but parse defensively
    }
    if (head.type === 'sym' || head.type === '%kw' || head.type === 'str') {
      if (head.value === '%module')   return readModuleBody();
      if (head.value === '%bind')     return readBindForm();
      if (head.value === '%public')   return readPublicForm();
      if (head.value === '%ref')      return readRefForm();
      if (head.value === '%axis')     return readAxisForm();
      // Variance-marked axes (spec §11): `(%uaxis <name>)` is the
      // contravariant / upper-variance form; `(%laxis <name>)` is the
      // covariant / lower. Both lower to the same `{kind:'axis'}` IR
      // shape, with the `variance` field set, mirroring the AST
      // representation.
      if (head.value === '%uaxis')    return readUAxisForm();
      if (head.value === '%laxis')    return readLAxisForm();
      if (head.value === '%kwarg')    return readKwargForm();
      if (head.value === '%field')    return readFieldForm();
      if (head.value === '%assign')   return readAssignForm();
      if (head.value === '%meta')     return readMetaForm();
      if (head.value === '%call')     return readCallForm();
      // Reified callables have a fixed-arity tail (spec §11):
      // output, input-origin tag, input list.
      if (head.value === 'functionof' || head.value === 'kernelof') {
        eat();
        return readReificationTail(head.value);
      }
    }
    // Default: parse as call (head is op name).
    eat();
    const op = head.value;
    const callShape: any = { kind: 'call', op, args: [], kwargs: {} };
    while (!eof() && peek().type !== ')') {
      const part = readForm();
      _absorbCallPart(callShape, part);
    }
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unclosed list (op ' + op + ')' });
    } else {
      eat();  // ')'
    }
    if (Object.keys(callShape.kwargs).length === 0) delete callShape.kwargs;
    if (callShape.args.length === 0) delete callShape.args;
    return callShape;
  }

  function readModuleBody(): any {
    eat(); // %module
    const mod = pir.loweredModule();
    while (!eof() && peek().type !== ')') {
      const sub = readForm();
      if (sub == null) continue;
      if (sub.__kind === 'public') {
        for (const n of sub.names) mod.publicSet.add(n);
      } else if (sub.__kind === 'bind') {
        mod.bindings.set(sub.name,
          pir.loweredBinding(sub.name, sub.rhs, { doc: sub.doc || null }));
      } else {
        diagnostics.push({ severity: 'error', message: 'pir-sexpr: stray form inside %module' });
      }
    }
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unclosed %module' });
    } else {
      eat();
    }
    return { __kind: 'module', module: mod };
  }

  function readBindForm(): any {
    eat(); // %bind
    const nameTok = eat();
    if (!nameTok || nameTok.type !== 'sym') {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: %bind expects a name' });
      return { __kind: 'bind', name: '?', rhs: null };
    }
    const rhs = readForm();
    // Optional trailing `(%doc <markup> <line>...)` sub-form.
    let doc: any = null;
    if (peek() && peek().type === '(') {
      // Peek inside: head must be the `%doc` keyword (lexer emits
      // `%`-prefixed names as type '%kw').
      const save = pos;
      eat(); // (
      const headTok = peek();
      if (headTok && headTok.type === '%kw' && headTok.value === '%doc') {
        eat(); // %doc
        const markupTok = peek();
        const markup = (markupTok && markupTok.type === 'sym')
          ? (eat() && markupTok.value) : 'md';
        const lines: string[] = [];
        while (!eof() && peek().type !== ')') {
          const t = eat();
          if (t.type === 'str') lines.push(t.value);
          else diagnostics.push({
            severity: 'error',
            message: `pir-sexpr: %doc line must be a string, got ${t.type}`,
          });
        }
        if (peek() && peek().type === ')') eat();
        doc = { markup, lines };
      } else {
        // Not a %doc form — rewind so the outer loop sees the `(`.
        pos = save;
      }
    }
    if (peek() && peek().type === ')') eat();
    return { __kind: 'bind', name: nameTok.value, rhs, doc };
  }

  function readPublicForm(): any {
    eat(); // %public
    const names: string[] = [];
    while (!eof() && peek().type !== ')') {
      const t = eat();
      if (t.type === 'sym') names.push(t.value);
    }
    if (peek() && peek().type === ')') eat();
    return { __kind: 'public', names };
  }

  function readRefForm(): any {
    eat(); // %ref
    const ns = eat();
    const name = eat();
    if (peek() && peek().type === ')') eat();
    return { kind: 'ref',
      ns: ns ? ns.value : '?',
      name: name ? name.value : '?' };
  }

  function readAxisForm(): any {
    eat(); // %axis
    const name = eat();
    if (peek() && peek().type === ')') eat();
    return { kind: 'axis', name: name ? name.value : '?' };
  }

  function readUAxisForm(): any {
    eat(); // %uaxis
    const name = eat();
    if (peek() && peek().type === ')') eat();
    return { kind: 'axis', name: name ? name.value : '?', variance: 'upper' };
  }

  function readLAxisForm(): any {
    eat(); // %laxis
    const name = eat();
    if (peek() && peek().type === ')') eat();
    return { kind: 'axis', name: name ? name.value : '?', variance: 'lower' };
  }

  function readKwargForm(): any {
    eat(); // %kwarg
    const name = eat();
    const value = readForm();
    if (peek() && peek().type === ')') eat();
    return { __kind: 'kwarg',
      name: name ? name.value : '?',
      value };
  }

  function readFieldForm(): any {
    eat(); // %field
    const name = eat();
    const value = readForm();
    if (peek() && peek().type === ')') eat();
    return { __kind: 'field',
      name: name ? name.value : '?',
      value };
  }

  function readAssignForm(): any {
    eat(); // %assign
    const name = eat();
    const value = readForm();
    if (peek() && peek().type === ')') eat();
    return { __kind: 'assign',
      name: name ? name.value : '?',
      value };
  }

  // (functionof <output> %specinputs ((<name> <ref>) ...)) |
  // (functionof <output> %autoinputs (%deferred | ((<name> <ref>) ...)))
  // — spec §11 "Reified callables", fixed arity after the head. Called
  // with the head symbol already eaten. An optional %meta may appear
  // immediately after the head, before the output.
  function readReificationTail(op: string): any {
    const callShape: any = { kind: 'call', op: 'functionof' };
    let body: any = null;
    // Output (+ optional leading %meta).
    while (!eof() && peek().type !== ')') {
      const t = peek();
      if (t.type === '%kw'
          && (t.value === '%specinputs' || t.value === '%autoinputs')) break;
      const part = readForm();
      if (part && part.__kind === 'meta') {
        callShape.meta = { type: part.type, phase: part.phase };
        continue;
      }
      if (body == null) { body = part; continue; }
      diagnostics.push({ severity: 'error',
        message: 'pir-sexpr: ' + op + ' has more than one output form '
          + 'before the input-origin tag' });
    }
    if (body == null) {
      diagnostics.push({ severity: 'error',
        message: 'pir-sexpr: ' + op + ' is missing its output expression' });
    }
    // Input-origin tag.
    let tag: string | null = null;
    if (!eof() && peek().type === '%kw'
        && (peek().value === '%specinputs' || peek().value === '%autoinputs')) {
      tag = eat().value;
    } else {
      diagnostics.push({ severity: 'error',
        message: 'pir-sexpr: ' + op + ' requires an input-origin tag '
          + '(%specinputs | %autoinputs)' });
    }
    // Input list (or %deferred for a pre-inference %autoinputs).
    let entries: Array<{ name: string; ref: any }> | null = null;
    if (!eof() && peek().type === '%kw' && peek().value === '%deferred') {
      eat();
      if (tag === '%specinputs') {
        diagnostics.push({ severity: 'error',
          message: 'pir-sexpr: %specinputs requires an input list '
            + '(%deferred is only valid under %autoinputs)' });
      }
    } else if (!eof() && peek().type === '(') {
      entries = readInputEntries(op);
    } else if (tag != null) {
      diagnostics.push({ severity: 'error',
        message: 'pir-sexpr: ' + op + ' ' + tag
          + ' requires an input list or %deferred' });
    }
    if (!eof() && peek().type === ')') {
      eat();
    } else {
      diagnostics.push({ severity: 'error',
        message: 'pir-sexpr: unclosed ' + op });
    }
    // Reconstruct the engine-internal arrays for an AUTHORED boundary
    // (%specinputs). A filled %autoinputs list is inference METADATA
    // (spec: strippable, like %meta) — the engine re-infers; parse and
    // drop. Entry refs map: `%local` ⇒ placeholder source; `self` (or
    // a module ns) ⇒ identifier binding source.
    if (tag === '%specinputs' && entries) {
      const params: string[] = [];
      const paramKwargs: string[] = [];
      const paramSources: any[] = [];
      for (const en of entries) {
        paramKwargs.push(en.name);
        params.push(en.ref.name);
        if (en.ref.ns === '%local') {
          paramSources.push({ kind: 'placeholder', name: en.ref.name });
        } else {
          const src: any = { kind: 'binding', name: en.ref.name };
          if (en.ref.ns !== 'self') src.ns = en.ref.ns;  // module-sourced (lossless)
          paramSources.push(src);
        }
      }
      // The body is kept verbatim — engine-internal IR is spec-shaped
      // (lower.ts: `%local` is placeholders-only; identifier-form
      // boundary refs stay plain self/module refs), so post-read modules
      // are indistinguishable from processSource output with no
      // namespace translation.
      callShape.params = params;
      callShape.paramKwargs = paramKwargs;
      callShape.paramSources = paramSources;
    }
    // kernelof(x, …) ≡ functionof(lawof(x), …) (spec §04): apply the
    // same lowering rule lower.ts applies at the surface ingest point,
    // so post-read modules carry ONE uniform reification form — op
    // 'kernelof' never exists in engine-internal IR. (JS output is
    // therefore the functionof∘lawof image, valid FlatPIR though not
    // the syntactic image of a kernelof-authored source — see
    // ARCHITECTURE "Engine-internal vs spec FlatPIR".)
    if (op === 'kernelof' && body) {
      body = { kind: 'call', op: 'lawof', args: [body] };
    }
    callShape.body = body;
    return callShape;
  }

  // ((<name> <ref>) (<name> <ref>) ...) — the reification input list.
  // Plain 2-lists, deliberately NOT %-keyword forms (spec §11).
  function readInputEntries(op: string): Array<{ name: string; ref: any }> {
    eat(); // outer '('
    const entries: Array<{ name: string; ref: any }> = [];
    while (!eof() && peek().type !== ')') {
      if (peek().type !== '(') {
        diagnostics.push({ severity: 'error',
          message: 'pir-sexpr: ' + op + ' input entry must be a '
            + '(<name> <ref>) pair' });
        eat();
        continue;
      }
      eat(); // inner '('
      const nameTok = eat();
      const ref = readForm();
      if (!nameTok || nameTok.type !== 'sym' || !ref || ref.kind !== 'ref') {
        diagnostics.push({ severity: 'error',
          message: 'pir-sexpr: malformed ' + op + ' input entry '
            + '(expected (<name> (%ref <ns> <node>)))' });
      } else {
        entries.push({ name: nameTok.value, ref });
      }
      if (!eof() && peek().type === ')') eat();
    }
    if (!eof() && peek().type === ')') eat();
    return entries;
  }

  function readMetaForm(): any {
    // Spec §11: (%meta <type> <phase>). We round-trip the raw forms
    // but don't currently re-feed them to the inference pipeline.
    eat(); // %meta
    const typ = readForm();
    const ph  = readForm();
    if (peek() && peek().type === ')') eat();
    return { __kind: 'meta', type: typ, phase: ph };
  }

  function readCallForm(): any {
    // (%call <callable> [%meta]? <args/%kwarg…>) — a call to a user-defined
    // callable (spec §11, b070d0a). The head is an EXPRESSION that must
    // evaluate to a user-defined callable: a `(%ref …)` in the common case
    // (reconstructing the engine's `target:{ns,name}` ref-head form), or an
    // inline callable expression such as a reification — kept as a `callee`
    // child node. The "must evaluate to a callable" condition is
    // inference's job; the reader stays liberal. Remaining parts are
    // ordinary positional args / %kwarg entries (and an optional %meta,
    // which may appear before or after the head).
    eat(); // %call
    const callShape: any = { kind: 'call', args: [], kwargs: {} };
    let headDone = false;
    while (!eof() && peek().type !== ')') {
      const part = readForm();
      if (part == null) continue;
      if (part.__kind === 'meta') {
        callShape.meta = { type: part.type, phase: part.phase };
        continue;
      }
      if (!headDone) {
        headDone = true;
        if (part.kind === 'ref') callShape.target = { ns: part.ns, name: part.name };
        else callShape.callee = part;
        continue;
      }
      _absorbCallPart(callShape, part);
    }
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unclosed %call' });
    } else {
      eat();  // ')'
    }
    if (Object.keys(callShape.kwargs).length === 0) delete callShape.kwargs;
    if (callShape.args.length === 0) delete callShape.args;
    return callShape;
  }

  function _absorbCallPart(call: any, part: any) {
    if (part == null) return;
    if (part.__kind === 'kwarg')  { call.kwargs[part.name] = part.value; return; }
    if (part.__kind === 'field')  { (call.fields ||= []).push({ name: part.name, value: part.value }); return; }
    if (part.__kind === 'assign') { (call.assigns ||= []).push({ name: part.name, value: part.value }); return; }
    if (part.__kind === 'meta')   { call.meta = { type: part.type, phase: part.phase }; return; }
    // Reified callables (the only body-carrying form) never reach this
    // absorb loop — they parse through readReificationTail; aggregate
    // and every other op are plain args-based.
    call.args.push(part);
  }

  // ---- Top-level: parse a sequence of forms (typically one %module) --

  const out: any = { module: null, diagnostics };
  while (!eof()) {
    const f = readForm();
    if (f && f.__kind === 'module') {
      out.module = f.module;
    }
  }
  if (!out.module) {
    diagnostics.push({ severity: 'warning', message: 'pir-sexpr: no %module form found' });
  }
  return out;
}

function _atomFromToken(t: any) {
  if (t.type === 'str') return { kind: 'lit', value: t.value };
  if (t.type === 'num') {
    return { kind: 'lit', value: t.value, numType: t.real ? 'real' : 'integer' };
  }
  if (t.type === 'bool') return { kind: 'lit', value: t.value };
  if (t.type === '_') return { kind: 'hole' };
  if (t.type === 'sym' || t.type === '%kw') {
    return { kind: 'const', name: t.value };
  }
  return null;
}

// ----- Tokenizer ------------------------------------------------------

function _tokenize(text: any) {
  const tokens: any[] = [];
  const s = String(text);
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; continue; }
    if (c === ';') {
      // Comment to EOL
      while (i < s.length && s[i] !== '\n') i++;
      continue;
    }
    if (c === '(') { tokens.push({ type: '(' }); i++; continue; }
    if (c === ')') { tokens.push({ type: ')' }); i++; continue; }
    if (c === '"') {
      let j = i + 1;
      let str = '';
      while (j < s.length && s[j] !== '"') {
        if (s[j] === '\\' && j + 1 < s.length) {
          const esc = s[j + 1];
          if (esc === 'n')      { str += '\n'; j += 2; continue; }
          if (esc === 't')      { str += '\t'; j += 2; continue; }
          if (esc === 'r')      { str += '\r'; j += 2; continue; }
          if (esc === '\\')     { str += '\\'; j += 2; continue; }
          if (esc === '"')      { str += '"';  j += 2; continue; }
          str += esc; j += 2; continue;
        }
        str += s[j]; j++;
      }
      tokens.push({ type: 'str', value: str });
      i = j + 1;
      continue;
    }
    // Atom — read until whitespace or paren.
    let j = i;
    while (j < s.length) {
      const cc = s[j];
      if (cc === ' ' || cc === '\t' || cc === '\n' || cc === '\r'
          || cc === '(' || cc === ')' || cc === ';') break;
      j++;
    }
    const tok = s.slice(i, j);
    i = j;
    if (tok === '_')          { tokens.push({ type: '_' }); continue; }
    if (tok === 'true')       { tokens.push({ type: 'bool', value: true }); continue; }
    if (tok === 'false')      { tokens.push({ type: 'bool', value: false }); continue; }
    // Numeric literal: matches integer or real (incl. scientific notation).
    // The lexical form carries the TYPE (spec §11: `3` integer, `1.0`
    // real) — record it so the lit reconstructs `numType`.
    if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(tok)) {
      tokens.push({ type: 'num', value: Number(tok), real: /[.eE]/.test(tok) });
      continue;
    }
    if (tok[0] === '%') {
      tokens.push({ type: '%kw', value: tok });
      continue;
    }
    tokens.push({ type: 'sym', value: tok });
  }
  return tokens;
}

module.exports = {
  toSexpr,
  fromSexpr,
  // Exported for tests
  _internal: { _tokenize },
};
