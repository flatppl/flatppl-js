// Regression guard for the trusted-MarkdownString command-URI injection
// (a `.flatppl`/embedded doc-comment is attacker-controlled and rendered in
// the hover). A trusted MarkdownString makes `command:`/`javascript:` links
// in that doc-comment EXECUTABLE on hover+click. The hover content needs no
// trust, so the source must never set `isTrusted = true`. extension.ts is the
// tracked source of truth (extension.js is a gitignored build artifact).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const SRC = fs.readFileSync(path.join(__dirname, '..', 'extension.ts'), 'utf8');

test('hover MarkdownString is never trusted (no command-URI injection)', () => {
  // No `<anything>.isTrusted = true` — trusted markdown enables command: links
  // from attacker-controlled doc-comments.
  assert.ok(
    !/\.isTrusted\s*=\s*true\b/.test(SRC),
    'extension.ts must not set a MarkdownString isTrusted = true (command-URI injection risk)',
  );
});
