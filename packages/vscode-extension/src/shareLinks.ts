// Playground share links in the extension host, with the playground's codec
// and limits (packages/share). No vscode import: the host passes the engine's
// bundle API and the libzstd binary. build-vendor.mjs bundles this module.

import { createShareCore } from '../../share/src/core.ts';
import { LINK_LIMITS, ShareLimitError, shareAddress, type ShareMember, type ShareView } from '../../share/src/format.ts';
import { archiveMembers, createShareQr, makeShareLinks, membersArchive, type ShareLinks } from '../../share/src/links.ts';

export { LINK_LIMITS };

export const NOT_A_SHARE = 'This link names a model on the website instead of carrying it. Open it in the playground.';

interface BundleApi {
  readArchive(bytes: Uint8Array, options: { id: string }): Promise<any>;
  writeArchive(files: Record<string, Uint8Array>, options: object): Promise<Uint8Array>;
  resolveBundleBytes(primaryPath: string, primaryBytes: Uint8Array, readBytes: (path: string) => Promise<Uint8Array | null>,
    options: { strict?: boolean; limits?: object }): Promise<any>;
  archiveMemberId(archiveId: string, path: string): string;
}

/** A decoded share: its rebuilt STORE ZIP and the member and binding to show. */
export interface OpenedShare {
  archive: Uint8Array;
  model: string | null;
  target: string | null;
  link: string;
}

const CHECK_ID = '/flatppl-share/check.flatppl.zip';

export function createShareService(api: BundleApi, wasm: () => Promise<BufferSource>) {
  const core = createShareCore(wasm);

  /**
   * Open a link as a playground visitor does: rebuild its ZIP, open it under
   * the browser limits, select its view and resolve the view's dependencies.
   */
  async function check(token: string): Promise<void> {
    const { view, members } = await core.decode(token);
    const zip = await membersArchive(members, api);
    const primary = view.model ? api.archiveMemberId(CHECK_ID, view.model) : CHECK_ID;
    const selected = members.find(member => member.name === view.model && member.kind === 'file');
    await api.resolveBundleBytes(primary, view.model ? selected?.bytes ?? new Uint8Array() : zip,
      async path => path === CHECK_ID ? zip : null, { strict: true, limits: LINK_LIMITS });
  }

  return {
    /** Links for a bundle. A bundle that would not open gets an error naming the limit or the dependency. */
    async create(members: ShareMember[], view: ShareView, baseUrl: string): Promise<ShareLinks> {
      const links = makeShareLinks(await core.encode(view, members), baseUrl);
      if (links.tooLong) throw new ShareLimitError('The link exceeds 65,536 characters.');
      try { await check(links.url.slice(links.url.indexOf('#share=') + 7)); }
      catch (error: any) {
        const { maxMemberBytes, maxTotalBytes, maxEntries, maxDepth } = LINK_LIMITS;
        const limits = /limit/i.test(error.message) ? ` The playground opens links with at most ${maxMemberBytes / 2 ** 20} MiB`
          + ` per member, ${maxTotalBytes / 2 ** 20} MiB in all, ${maxEntries} entries and ${maxDepth} nesting levels.` : '';
        throw new Error('The playground would not open this link: ' + error.message + '.' + limits);
      }
      return links;
    },
    qr: createShareQr,
    /** The members of a ZIP bundle, as a share stores them. */
    members: (zip: Uint8Array) => archiveMembers(zip, api),
    /** Decode a link, a QR link or a bare token, offline. A route's member and binding take precedence. */
    async open(input: string): Promise<OpenedShare> {
      const address = shareAddress(input);
      if (!address) throw new Error(NOT_A_SHARE);
      const { link, view, members } = await core.decode(address.token);
      const target = address.target === undefined ? view.target : address.target;
      return { archive: await membersArchive(members, api), model: (address.model ?? view.model) || null,
        target: target || null, link };
    },
  };
}

const escapeHtml = (text: string) => text.replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);

/** The QR webview. The SVG comes from the QR library, so it carries no script. */
export function shareQrHtml(url: string, qr: { svg: string; version: number; size: number }, nonce: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}';">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style nonce="${nonce}">
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 16px; }
  .qr svg { display: block; width: min(80vmin, 480px); height: auto; }
  textarea { width: 100%; max-width: 640px; height: 5em; word-break: break-all; font-family: var(--vscode-editor-font-family);
    color: var(--vscode-input-foreground); background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border, transparent); }
  button { margin: 8px 8px 0 0; padding: 4px 12px; border: none; cursor: pointer;
    color: var(--vscode-button-foreground); background: var(--vscode-button-background); }
  button:hover { background: var(--vscode-button-hoverBackground); }
</style>
</head>
<body>
<div class="qr" role="img" aria-label="QR code for this model">${qr.svg}</div>
<p>QR version ${qr.version}, ${qr.size} by ${qr.size} modules, medium error correction.
The link and the QR code hold the model's code, imports and data.</p>
<textarea readonly aria-label="Share link">${escapeHtml(url)}</textarea>
<div>
<button data-action="copy">Copy Link</button>
<button data-action="svg">Save SVG</button>
<button data-action="png">Save PNG</button>
</div>
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  for (const button of document.querySelectorAll('button')) {
    button.addEventListener('click', () => vscode.postMessage({ action: button.dataset.action }));
  }
</script>
</body>
</html>`;
}
