// @flatppl/web — gallery footer.
//
// Renders the footer links (Legal Notice, …) into #app-footer from
// __FLATPPL_CONFIG__.footerLinks. The links are deployment-specific —
// a legal notice names the provider of THIS deployment — so the shell
// never hard-codes them: build.mjs bakes the site overlay's page list
// into build-flags.js, and a host that embeds the shell in a page of
// its own sets footerLinks in its bootstrap instead (see build-site.mjs
// for both routes). Without links the footer stays hidden and its grid
// row collapses.
//
// Lives on globalThis as window.FlatPPLWebFooter.

'use strict';

(function (globalScope: any) {
  interface FooterLink { label: string; href: string; }

  /** Validate the configured links: each needs a non-empty string
      label and href; anything else is dropped (a malformed bootstrap
      shouldn't take the whole footer down, nor render `undefined`). */
  function normalizeLinks(raw: any): FooterLink[] {
    if (!Array.isArray(raw)) return [];
    const out: FooterLink[] = [];
    for (const l of raw) {
      if (l && typeof l.label === 'string' && l.label.trim()
            && typeof l.href === 'string' && l.href.trim()) {
        out.push({ label: l.label, href: l.href });
      }
    }
    return out;
  }

  /** Fill #app-footer from the config and unhide it when there is
      anything to show. Returns the rendered links. */
  function install(config: any): FooterLink[] {
    const el = document.getElementById('app-footer');
    if (!el) return [];
    const links = normalizeLinks(config && config.footerLinks);
    el.textContent = '';
    for (const l of links) {
      const a = document.createElement('a');
      a.href = l.href;
      a.textContent = l.label;   // textContent, never innerHTML — the
                                 // labels come from a config file
      el.appendChild(a);
    }
    el.hidden = links.length === 0;
    return links;
  }

  globalScope.FlatPPLWebFooter = { install: install, normalizeLinks: normalizeLinks };
})(typeof window !== 'undefined' ? window : globalThis);
