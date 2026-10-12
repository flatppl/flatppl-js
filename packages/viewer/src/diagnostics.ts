// Source diagnostics shared by the viewer and host editor. Parser recovery
// nodes are implementation details; report the preceding source error instead.

export interface SourceDiagnostic {
  severity: string;
  message: string;
  loc?: { start: { line: number; col: number } };
}

/** Human-readable punctuation and end-of-source names for parser messages. */
export function formatDiagnosticMessage(message: string): string {
  if (message === "Unexpected token '' (EOF)") return 'The source ends before this expression is complete.';
  if (message === "Undefined variable '__error__'") return 'This expression is incomplete.';
  if (!/^(Expected |Unexpected token )/.test(message)) return message;
  const names: Record<string, string> = {
    LPAREN: "'('", RPAREN: "')'", LBRACKET: "'['", RBRACKET: "']'",
    LBRACE: "'{'", RBRACE: "'}'", COMMA: "','", EQUALS: "'='",
    IDENT: 'a name', NEWLINE: 'a new line', EOF: 'the end of the source',
  };
  return message.replace(/\b(LPAREN|RPAREN|LBRACKET|RBRACKET|LBRACE|RBRACE|COMMA|EQUALS|IDENT|NEWLINE|EOF)\b/g,
    (name) => names[name]).replace(/ ''$/, '').replace(', got the end of the source', ' before the end of the source');
}

/** Parser/tokenizer errors are not binned onto bindings by the analyzer.
 * Keep them separately so a recovery binding cannot mask the real problem.
 * Unrelated binding errors still allow the user to inspect a valid binding. */
export function sourceErrorsFor(result: {
  bindings: Map<string, { diagnostics?: SourceDiagnostic[] }>;
  diagnostics: SourceDiagnostic[];
}): SourceDiagnostic[] {
  const bound = new Set<SourceDiagnostic>();
  for (const binding of result.bindings.values()) {
    for (const diagnostic of binding.diagnostics || []) bound.add(diagnostic);
  }
  const errors = result.diagnostics.filter((diagnostic) => diagnostic.severity === 'error' && !bound.has(diagnostic));
  // An expected delimiter is a more useful repair than the parser's earlier
  // generic EOF diagnostic at that same point.
  return errors.filter((diagnostic) => diagnostic.message !== "Unexpected token '' (EOF)"
    || !errors.some((other) => /^Expected (RPAREN|RBRACKET|RBRACE), got EOF/.test(other.message)
      && other.loc?.start.line === diagnostic.loc?.start.line && other.loc?.start.col === diagnostic.loc?.start.col));
}
