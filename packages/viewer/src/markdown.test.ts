// @ts-nocheck — Node runs tests directly; the viewer build targets the browser.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderDoc, documentHtml } from './markdown.ts';

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

// Every tag the corpus below may emit: CommonMark/GFM plus Temml's MathML.
const DOCUMENT_TAGS = new Set([
  'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'img', 'em', 'strong', 'del',
  'code', 'pre', 'blockquote', 'ul', 'ol', 'li', 'hr', 'br',
  'table', 'thead', 'tbody', 'tr', 'th', 'td', 'span',
  'math', 'mrow', 'mi', 'mo', 'mn', 'ms', 'mtext', 'mspace', 'msup', 'msub',
  'msubsup', 'mfrac', 'msqrt', 'mroot', 'mstyle', 'mpadded', 'mphantom',
  'menclose', 'mtable', 'mtr', 'mtd', 'munder', 'mover', 'munderover',
  'semantics', 'annotation',
]);

/** Tags and decoded attributes of renderer output. Text escapes `<`, so each
 *  `<` before a letter opens a tag. Refuse a tag the scan cannot read. */
function elements(html: string) {
  const decode = (v: string) => v.replace(/&(#x[\da-f]+|#\d+|quot|amp|lt|gt);/gi, (_, e) =>
    e[0] === '#' ? String.fromCodePoint(e[1].toLowerCase() === 'x' ? parseInt(e.slice(2), 16) : +e.slice(1))
      : { quot: '"', amp: '&', lt: '<', gt: '>' }[e.toLowerCase()]);
  const value = String.raw`(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?`;
  const tags = [...html.matchAll(new RegExp(String.raw`<([a-zA-Z][\w-]*)((?:\s+[^\s=>/]+` + value + String.raw`)*)\s*/?>`, 'g'))];
  assert.equal(tags.length, html.match(/<[a-zA-Z]/g)?.length ?? 0, 'unreadable tag in ' + html);
  return tags.map(m => ({
    tag: m[1].toLowerCase(),
    attrs: [...m[2].matchAll(new RegExp(String.raw`\s+([^\s=>/]+)` + value, 'g'))]
      .map(a => ({ name: a[1].toLowerCase(), value: decode(a[2] ?? a[3] ?? a[4] ?? '') })),
  }));
}

test('Markdown output keeps only document tags, inert attributes, and web destinations', () => {
  // Inert renderer probes, one per vector class. No browser executes them.
  const corpus = {
    'raw HTML block': '<iframe srcdoc="x"></iframe>\n\n<style>p{}</style>\n\n<form action="javascript:void 0"><button>b</button></form>',
    'raw HTML handlers': '<details open ontoggle="void 0">x</details>\n\n<object data="x"></object><embed src="x">',
    'nested foreign content': '<math><mtext><table><mglyph><style><img src=x onerror="void 0"></style></mglyph></table></mtext></math>',
    'HTML in heading and table': '# h <img src=x onerror="void 0">\n\n| a |\n| - |\n| <svg onload="void 0"> |',
    'fence info string': '```"><img src=x onerror="void 0">\ncode\n```',
    'link title': '[t](https://example.org "q\\" onmouseover=\\"void 0")',
    'reference definition': '[r][d]\n\n[d]: javascript:void(0)',
    'scheme autolink': '<javascript:void(0)> <data:text/html,x>',
    'tab in a bracketed destination': '[t](<java\tscript:void(0)>)',
    'image destination quote': '![i](x" onerror="void 0)',
    'math HTML commands': '$\\href{javascript:void(0)}{h}$ $\\htmlClass{c}{x}$ $\\htmlStyle{color:red}{x}$ $\\htmlId{i}{x}$ $\\htmlData{a=b}{x}$',
    'math URL and image commands': '$\\url{javascript:void(0)}$ $\\includegraphics{javascript:void(0)}$',
    'math color argument': '$\\color{red;background:url(x)}{c}$',
    'math text': '$\\text{<img src=x onerror="void 0">}$ $$\\unicode{<svg onload="void 0">}$$',
    'math fence breakout': '```math\n\\text{</math><img src=x onerror="void 0">}\n```',
  };
  for (const [vector, source] of Object.entries(corpus)) {
    for (const { tag, attrs } of elements(render(source))) {
      assert.ok(DOCUMENT_TAGS.has(tag), `${vector}: <${tag}>`);
      for (const { name, value } of attrs) {
        assert.doesNotMatch(name, /^on/, `${vector}: ${name} on <${tag}>`);
        if (name === 'href' || name === 'src') {
          const scheme = /^\s*([a-z][a-z\d+.-]*):/i.exec(value.replace(/[\t\n\r]/g, ''));
          assert.ok(!scheme || /^(?:https?|mailto)$/i.test(scheme[1]), `${vector}: ${name}="${value}"`);
        }
      }
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

test('Markdown renders a ```math fence as display math (the Markdown export convention)', () => {
  const html = render('Before\n\n```math\n\\mu \\sim \\mathcal{N}(0, 1)\n```\n\nAfter');
  assert.match(html, /<math[^>]*display="block"/);
  assert.ok(!html.includes('<code'), 'the fence is not shown as a code block');
  assert.match(render('```js\nlet x = 1;\n```'), /<code class="language-js">/);
});

test('document images require web host consent before emitting a network URL', (t) => {
  const requests = [];
  let allow = false;
  t.after(() => { delete globalThis.window; delete globalThis.document; });
  globalThis.window = { __FLATPPL_CONFIG__: {
    approveDocumentImage(url) { requests.push(url); return allow; },
  } };
  globalThis.document = {
    baseURI: 'https://live.flatppl.org/',
    createTextNode: text => text,
    createElement(tag) {
      assert.equal(tag, 'template');
      // Model the inert template boundary for the one image in this fixture.
      // The browser parses attributes before the approval callback sees them.
      let html, replacement;
      const image = {
        alt: 'plot', getAttribute: () => html.match(/src="([^"]+)"/)[1],
        removeAttribute() {},
        setAttribute(_key, url) { replacement = html.match(/<img[^>]*>/)[0].replace(/src="[^"]+"/, 'src="' + url + '"'); },
        replaceWith(text) { replacement = text; },
      };
      return {
        set innerHTML(value) { html = value; },
        get innerHTML() { return html.replace(/<img[^>]*>/, replacement); },
        content: { querySelectorAll: () => [image] },
      };
    },
  };
  assert.doesNotMatch(render('![plot](https://example.org/plot.png)'), /<img/);
  allow = true;
  const html = render('![plot](//example.org/plot.png "title")');
  assert.match(html, /<img src="https:\/\/example.org\/plot.png" alt="plot" title="title">/);
  allow = false;
  assert.equal(documentHtml('<article><img src="https://example.org/plot.png" alt="plot"></article>'),
    '<article>plot</article>');
  assert.deepEqual(requests, Array(3).fill('https://example.org/plot.png'));
});
