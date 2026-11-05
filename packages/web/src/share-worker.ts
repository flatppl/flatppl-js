/**
 * Share-link work on untrusted input runs here, under the caller's timeout:
 * token text, Zstandard and the member container. The rebuilt ZIP then passes
 * the archive reader on the page.
 */
import { createShareCore } from '../../share/src/core.ts';
import { ShareLimitError } from '../../share/src/format.ts';
import type { ShareWorkerRequest, ShareWorkerReply } from './share-codec.ts';

const worker = self as unknown as {
  addEventListener: (type: 'message', listener: (event: MessageEvent<ShareWorkerRequest>) => void) => void;
  postMessage: (message: ShareWorkerReply, transfer?: Transferable[]) => void;
};

const core = createShareCore(() => fetch(new URL('zstd.wasm', import.meta.url)).then(response => {
  if (!response.ok) throw new Error('The share codec could not load.');
  return response.arrayBuffer();
}));

worker.addEventListener('message', async ({ data }) => {
  try {
    if (data.operation === 'encode' && Array.isArray(data.members)) {
      let tokens = null;
      // A bundle that exceeds what a link can open keeps its ZIP download.
      try { tokens = await core.encode(data.view, data.members); }
      catch (error) { if (!(error instanceof ShareLimitError)) throw error; }
      worker.postMessage({ id: data.id, tokens });
    } else if (data.operation === 'decode' && typeof data.token === 'string') {
      const { link, view, members } = await core.decode(data.token);
      worker.postMessage({ id: data.id, link, view, members },
        members.map(member => member.bytes.buffer as ArrayBuffer));
    } else {
      throw new Error('Invalid share codec request.');
    }
  } catch (error) {
    worker.postMessage({ id: data.id, error: error instanceof Error ? error.message : 'Invalid share link.' });
  }
});
