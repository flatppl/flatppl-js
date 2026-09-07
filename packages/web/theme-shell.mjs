import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const HEADER_MARKER = '<!-- flatppl-theme:header -->';
const FOOTER_MARKER = '<!-- flatppl-theme:footer -->';

function escapeAttribute(value) {
  return String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
}

function escapeText(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Insert the pinned shared shell fragments into a host page template. */
export async function insertThemeShell(html, { themeDir, mainTarget, footerLinks = [] }) {
  let header = await readFile(join(themeDir, 'header.html'), 'utf8');
  let footer = await readFile(join(themeDir, 'footer.html'), 'utf8');
  header = header.replace('href="#main-content"', `href="#${mainTarget}"`);
  for (const link of footerLinks) {
    const label = escapeText(link.label);
    const replacement = `<a href="${escapeAttribute(link.href)}">${label}</a>`;
    const labelEnd = footer.indexOf(`>${label}</a>`);
    if (labelEnd >= 0) {
      const start = footer.lastIndexOf('<a ', labelEnd);
      footer = footer.slice(0, start) + replacement
        + footer.slice(labelEnd + label.length + 5);
    } else {
      footer = footer.replace('</footer>', `  ${replacement}\n</footer>`);
    }
  }
  if (!html.includes(HEADER_MARKER) || !html.includes(FOOTER_MARKER)) {
    throw new Error('theme shell markers are missing from the host template');
  }
  return html.replace(HEADER_MARKER, header.trim())
    .replace(FOOTER_MARKER, footer.trim());
}
