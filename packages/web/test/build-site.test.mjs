// Tests for the deploy-specific site overlay (build-site.mjs): the pages
// listed in site/site.json become standalone HTML pages in dist/ plus the
// gallery's footer links. The overlay is what lets another deployment of
// @flatppl/web carry its own legal notice — so the contract (manifest →
// pages + links, escaping, absence handling) is pinned here.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import {
  sitePagesFromConfig,
  renderSitePage,
  loadSiteConfig,
  buildSite,
} from '../build-site.mjs';

test('sitePagesFromConfig derives the output file and footer link per page', () => {
  const pages = sitePagesFromConfig({
    pages: [
      { title: 'Legal Notice', source: 'legal-notice.md' },
      { title: 'Privacy Policy', source: 'privacy-policy.md' },
    ],
  });
  assert.deepEqual(pages.map((p) => p.out), ['legal-notice.html', 'privacy-policy.html']);
  assert.deepEqual(pages.map((p) => ({ label: p.title, href: p.out })), [
    { label: 'Legal Notice', href: 'legal-notice.html' },
    { label: 'Privacy Policy', href: 'privacy-policy.html' },
  ]);
});

test('sitePagesFromConfig rejects a page without title or source', () => {
  assert.throws(() => sitePagesFromConfig({ pages: [{ title: 'x' }] }), /source/);
  assert.throws(() => sitePagesFromConfig({ pages: [{ source: 'x.md' }] }), /title/);
  assert.throws(() => sitePagesFromConfig({ pages: [{ title: 'x', source: 'sub/x.md' }] }), /basename/);
});

test('renderSitePage escapes the title and wires the shared footer', () => {
  const html = renderSitePage({
    title: 'A <b>title</b> & more',
    bodyHtml: '<p>body</p>',
    footerLinks: [{ label: 'Legal <Notice>', href: 'legal-notice.html' }],
  });
  assert.match(html, /<title>A &lt;b&gt;title&lt;\/b&gt; &amp; more/);
  assert.match(html, /<h1>A &lt;b&gt;title&lt;\/b&gt; &amp; more<\/h1>/);
  assert.ok(html.includes('<p>body</p>'));
  assert.match(html, /<a href="legal-notice.html">Legal &lt;Notice&gt;<\/a>/);
  // Back-link to the gallery and the standalone stylesheet (NOT style.css,
  // whose html/body overflow:hidden would stop a long page from scrolling).
  assert.match(html, /href="\.\/"/);
  assert.match(html, /href="page\.css"/);
  assert.ok(!html.includes('style.css'));
});

test('loadSiteConfig returns no pages when the overlay has no manifest', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'flatppl-site-'));
  try {
    assert.deepEqual(await loadSiteConfig(dir), { pages: [] });
    assert.deepEqual(await loadSiteConfig(join(dir, 'does-not-exist')), { pages: [] });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('buildSite renders every manifest page to dist/ and returns the footer links', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'flatppl-site-'));
  const siteDir = join(dir, 'site');
  const distDir = join(dir, 'dist');
  try {
    await mkdir(siteDir);
    await mkdir(distDir);
    await writeFile(join(siteDir, 'site.json'), JSON.stringify({
      pages: [{ title: 'Legal Notice', source: 'legal-notice.md' }],
    }));
    await writeFile(join(siteDir, 'legal-notice.md'),
      '## Provider\n\nThe provider is [X](https://example.org).\n');
    const links = await buildSite({ siteDir, distDir });
    assert.deepEqual(links, [{ label: 'Legal Notice', href: 'legal-notice.html' }]);
    const { readFile } = await import('node:fs/promises');
    const html = await readFile(join(distDir, 'legal-notice.html'), 'utf8');
    assert.match(html, /<h2[^>]*>Provider<\/h2>/);
    assert.match(html, /<a href="https:\/\/example\.org">X<\/a>/);
    assert.match(html, /<h1>Legal Notice<\/h1>/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('buildSite with an empty overlay yields no links and writes nothing', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'flatppl-site-'));
  const distDir = join(dir, 'dist');
  try {
    await mkdir(distDir);
    const links = await buildSite({ siteDir: join(dir, 'missing'), distDir });
    assert.deepEqual(links, []);
    const { readdir } = await import('node:fs/promises');
    assert.deepEqual(await readdir(distDir), []);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
