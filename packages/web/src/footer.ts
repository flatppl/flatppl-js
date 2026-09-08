// @flatppl/web — gallery footer.
//
// Adapts the shared theme footer to deployment-specific pages. The release
// fragment supplies the Legal Notice link;
// a deployment overlay may retarget a matching label to its local page or add
// another deployment-owned link.
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

  /** Apply configured link overrides while preserving the shared footer. */
  function install(config: any): FooterLink[] {
    const el = document.querySelector('.fp-shell-footer');
    if (!el) return [];
    const links = normalizeLinks(config && config.footerLinks);
    const anchors = Array.from(el.querySelectorAll('a')) as HTMLAnchorElement[];
    for (const l of links) {
      const existing = anchors.find((a) => a.textContent === l.label);
      if (existing) {
        existing.href = l.href;
        continue;
      }
      const anchor = document.createElement('a');
      anchor.href = l.href;
      anchor.textContent = l.label;
      el.appendChild(anchor);
    }
    return links;
  }

  globalScope.FlatPPLWebFooter = { install: install, normalizeLinks: normalizeLinks };
})(typeof window !== 'undefined' ? window : globalThis);
