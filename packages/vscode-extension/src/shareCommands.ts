'use strict';
// The share-link commands: Copy Share Link, Show Share QR Code and Open Share
// Link (packages/web/share-links.md). The host passes vscode and its bundle
// pieces, so the tests drive these commands with a stub.

const { BUNDLE_SCHEME, enginePathOf } = require('./remoteModule');

interface ShareHost {
  bundleHost: any;
  bundleApi: any;
  /** The share service of ./shareLinks, loaded on the first share command. */
  shares: () => any;
  /** The binding the visualizer shows for a document, if any. */
  shownTarget: (uri: any) => string | null;
  /** Binding name to its 0-based line in a document. */
  bindingLine: (document: any, name: string) => number | undefined;
  /** Remember the bundle that Browse and Extract act on. */
  opened: (primaryPath: string) => void;
}

function createShareCommands(vscode: any, host: ShareHost) {
  const { bundleHost, bundleApi, shares } = host;

  /** A top-level mounted bundle is shared whole. Any other model is packed as Export Bundle packs it. */
  async function linksForEditor() {
    const doc = vscode.window.activeTextEditor?.document;
    if (doc?.languageId !== 'flatppl') throw new Error('Open a FlatPPL module to share it.');
    const graph = await bundleHost.resolve(doc.uri, doc.getText(), true);
    const member = doc.uri.scheme === BUNDLE_SCHEME ? bundleApi.parseArchiveMemberId(enginePathOf(doc.uri)) : null;
    const archive = member && !bundleApi.parseArchiveMemberId(member.archiveId) && graph.archives[member.archiveId];
    let members, model = '';
    if (archive && !archive.parent && graph.files[member.archiveId]) {
      members = await shares().members(graph.files[member.archiveId]);
      model = archive.root ? member.path.slice(archive.root.length + 1) : member.path;
      if (model === 'main.flatppl') model = '';
    } else {
      members = await shares().members(await bundleHost.pack(graph));
    }
    const target = host.shownTarget(doc.uri) || '';
    const baseUrl = vscode.workspace.getConfiguration('flatppl').get('shareBaseUrl') || 'https://live.flatppl.org/';
    try {
      return await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification,
        title: 'FlatPPL: creating share link…' }, () => shares().create(members, { model, target }, baseUrl));
    } catch (error: any) { throw new Error('No share link for this model. ' + error.message); }
  }

  async function copy(url: string) {
    await vscode.env.clipboard.writeText(url);
    vscode.window.showInformationMessage('FlatPPL: share link copied.');
  }

  async function save(name: string, filters: Record<string, string[]>, bytes: Uint8Array) {
    const folder = vscode.workspace.workspaceFolders?.[0];
    const target = await vscode.window.showSaveDialog({ filters, saveLabel: 'Save QR Code',
      defaultUri: folder && vscode.Uri.joinPath(folder.uri, name) });
    if (target) await vscode.workspace.fs.writeFile(target, bytes);
  }

  return {
    async copyShareLink() { await copy((await linksForEditor()).url); },

    async showShareQr() {
      const links = await linksForEditor();
      const qr = await shares().qr(links);
      if (!qr) {
        const pick = await vscode.window.showWarningMessage('FlatPPL: this model is too large for a QR code '
          + '(more than 5,596 digits or QR version 40). Use the link instead.', 'Copy Link');
        if (pick) await copy(links.url);
        return;
      }
      const panel = vscode.window.createWebviewPanel('flatpplShareQr', 'FlatPPL: Share QR Code',
        vscode.ViewColumn.Beside, { enableScripts: true, localResourceRoots: [] });
      panel.webview.html = require('./shareLinks').shareQrHtml(links.url, qr,
        require('crypto').randomBytes(16).toString('base64'));
      panel.webview.onDidReceiveMessage((message: any) => {
        const action = message?.action === 'copy' ? copy(links.url)
          : message?.action === 'svg' ? save('flatppl-model-qr.svg', { 'SVG image': ['svg'] }, Buffer.from(qr.svg, 'utf8'))
          : message?.action === 'png' ? save('flatppl-model-qr.png', { 'PNG image': ['png'] },
            Buffer.from(qr.png.slice(qr.png.indexOf(',') + 1), 'base64'))
          : undefined;
        action?.catch((error: any) => vscode.window.showErrorMessage('FlatPPL: ' + error.message));
      });
    },

    /** Opening a link is offline. Every dependency comes from the bundle, so nothing is fetched or approved. */
    async openShareLink() {
      const input = await vscode.window.showInputBox({ title: 'Open Share Link', ignoreFocusOut: true,
        prompt: 'Paste a FlatPPL share link, a QR code link or a share token.' });
      if (!input?.trim()) return;
      const shared = await shares().open(input);
      const digest = require('crypto').createHash('sha256').update(shared.link).digest('hex').slice(0, 16);
      const id = `/flatppl-share/${digest}.flatppl.zip`;
      const graph = await bundleHost.mount(id, shared.archive,
        vscode.Uri.from({ scheme: 'flatppl-share', path: '/' + digest }), require('./shareLinks').LINK_LIMITS);
      host.opened(graph.primaryPath);
      // As in the playground, a view that names no file of the bundle opens main.flatppl.
      const selected = shared.model && bundleApi.archiveMemberId(id, shared.model);
      const memberId = bundleHost.members(id).some((m: any) => m.id === selected && !m.directory) ? selected : graph.primaryPath;
      const editor = await vscode.window.showTextDocument(bundleHost.uriFor(memberId), { preview: false });
      const line = shared.target ? host.bindingLine(editor.document, shared.target) : undefined;
      if (line !== undefined) {
        const at = new vscode.Position(line, 0);
        editor.selection = new vscode.Selection(at, at);
        editor.revealRange(new vscode.Range(at, at), vscode.TextEditorRevealType.InCenter);
      }
      if (shared.problem) vscode.window.showWarningMessage('FlatPPL: this shared model will not resolve. ' + shared.problem);
    },
  };
}

module.exports = { createShareCommands };
