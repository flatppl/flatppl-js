// @ts-nocheck — Node runs tests directly; the viewer build targets the browser.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderDoc } from './markdown.ts';

const render = (source: string) => renderDoc({ markup: 'md', lines: source.split('\n') })!;

test('Markdown displays raw HTML as text at every HTML token boundary', () => {
  // These are inert renderer probes. No browser executes the returned HTML.
  for (const source of [
    '<button onclick="void 0">label</button>',
    'inline <img src=x onerror="void 0"> text',
    '<script>void 0</script>',
    '<svg><a href="javascript:void(0)">label</a></svg>',
    '<!-- comment -->',
  ]) {
    const html = render(source);
    assert.doesNotMatch(html, /<(?:button|img|script|svg|a\b|!--)/i);
    assert.match(html, /&lt;/);
  }
});

test('Markdown drops active and obscured link and image destinations', () => {
  for (const href of [
    'javascript:void(0)', 'JaVaScRiPt:void(0)', 'vbscript:0',
    'data:text/html,label', 'file:///tmp/label',
    'jav&#x61;script:void(0)', 'javascript&colon;void(0)',
    'java&Tab;script:void(0)', 'java&NewLine;script:void(0)',
    '&#106;avascript:void(0)',
  ]) {
    for (const prefix of ['', '!']) {
      const html = render(`${prefix}[label](${href})`);
      assert.doesNotMatch(html, /<(?:a|img)\b/i, href);
      assert.match(html, /label/);
    }
  }
});

test('Markdown retains safe destinations, formatting, code, and math', () => {
  const html = render([
    '[**web**](https://example.org/?a=1&amp;b=2 "title")',
    '[mail](mailto:reader@example.org)', '[local](../guide.md#part)',
    '[anchor](#part)', '![*image*](./plot.png "plot")',
    '', '| a | b |', '| - | - |', '| 1 | 2 |', '',
    '`<tag>`', '', '```html', '<script>void 0</script>', '```',
    '', '$x^2$',
  ].join('\n'));
  assert.match(html, /href="https:\/\/example\.org\/\?a=1&amp;b=2"/);
  assert.match(html, /<strong>web<\/strong>/);
  assert.match(html, /href="mailto:reader@example.org"/);
  assert.match(html, /href="\.\.\/guide.md#part"/);
  assert.match(html, /href="#part"/);
  assert.match(html, /<img src="\.\/plot.png" alt="image" title="plot">/);
  assert.match(html, /<table>/);
  assert.match(html, /<code>&lt;tag&gt;<\/code>/);
  assert.match(html, /&lt;script&gt;void 0&lt;\/script&gt;/);
  assert.match(html, /<math/);
  assert.doesNotMatch(html, /<script>/);
});
