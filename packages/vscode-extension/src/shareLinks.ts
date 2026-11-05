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
  /** Why the view would not resolve, such as a remote `load_module`, or null. */
  problem: string | null;
}

const CHECK_ID = '/flatppl-share/check.flatppl.zip';

/** `moduleDeps` lists a source's `load_module` paths. It names the member behind a path error. */
export function createShareService(api: BundleApi, wasm: () => Promise<BufferSource>,
  moduleDeps: (source: string) => string[] = () => []) {
  const core = createShareCore(wasm);

  /**
   * Open a bundle as a playground visitor does: open the rebuilt ZIP under the
   * browser limits, select the view and resolve the view's dependencies.
   * Reads stay inside the ZIP, so nothing is fetched or approved.
   */
  async function check(zip: Uint8Array, model: string, members: ShareMember[]): Promise<void> {
    const primary = model ? api.archiveMemberId(CHECK_ID, model) : CHECK_ID;
    const selected = members.find(member => member.name === model && member.kind === 'file');
    try {
      await api.resolveBundleBytes(primary, model ? selected?.bytes ?? new Uint8Array() : zip,
        async path => path === CHECK_ID ? zip : null, { strict: true, limits: LINK_LIMITS });
    } catch (error: any) {
      // Bundles keep load_module paths relative (spec section 04), so a URL import fails here.
      const path = /^Bundle paths must be relative: '(.*)'$/.exec(error.message)?.[1];
      const importer = path && members.find(member => member.kind === 'file' && member.name.endsWith('.flatppl')
        && moduleDeps(new TextDecoder().decode(member.bytes)).includes(path));
      throw new Error(importer ? `${importer.name}: ${error.message}` : error.message);
    }
  }

  return {
    /** Links for a bundle. A bundle that would not open gets an error naming the limit or the dependency. */
    async create(members: ShareMember[], view: ShareView, baseUrl: string): Promise<ShareLinks> {
      const links = makeShareLinks(await core.encode(view, members), baseUrl);
      if (links.tooLong) throw new ShareLimitError('The link exceeds 65,536 characters.');
      // Check the link as decoded, not the input members.
      const decoded = await core.decode(links.url.slice(links.url.indexOf('#share=') + 7));
      try { await check(await membersArchive(decoded.members, api), decoded.view.model, decoded.members); }
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
      const model = (address.model ?? view.model) || null;
      const target = address.target === undefined ? view.target : address.target;
      const archive = await membersArchive(members, api);
      const shown = model && members.some(member => member.name === model && member.kind === 'file') ? model : '';
      const problem = await check(archive, shown, members).then(() => null, (error: any) => error.message as string);
      return { archive, model, target: target || null, link, problem };
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
