/**
 * Share-link work on untrusted input runs here, under the caller's timeout:
 * token text, Zstandard and the member container. The rebuilt ZIP then passes
 * the archive reader on the page.
 */
import dictionary1 from './share-dictionaries/v1.zdict';
import { loadZstd, type Zstd } from './zstd.ts';
import {
  MAX_SHARE_CONTENT_BYTES, MAX_SHARE_PAYLOAD_BYTES, ShareLimitError, ZSTD_MAGIC,
  decodeContainer, encodeContainer, framePayload, linkToken, payloadFrame, qrToken, tokenPayload,
  type ShareMember, type ShareView,
} from './share-format.ts';
import type { ShareTokens, ShareWorkerRequest, ShareWorkerReply } from './share-codec.ts';

/** Payload version to dictionary. Never change or reuse a version: printed links must keep opening. */
const DICTIONARIES: Record<number, Uint8Array> = { 1: dictionary1 };
const VERSION = 1;
const LEVEL = 19;
/** Level 19 needs between 64 and 128 MiB for the largest content. */
const MAX_HEAP_BYTES = 128 * 1024 * 1024;

const worker = self as unknown as {
  addEventListener: (type: 'message', listener: (event: MessageEvent<ShareWorkerRequest>) => void) => void;
  postMessage: (message: ShareWorkerReply, transfer?: Transferable[]) => void;
};

let zstd: Promise<Zstd> | undefined;
function codec(): Promise<Zstd> {
  return zstd ||= fetch(new URL('zstd.wasm', import.meta.url)).then(response => {
    if (!response.ok) throw new Error('The share codec could not load.');
    return response.arrayBuffer();
  }).then(wasm => loadZstd(wasm, MAX_HEAP_BYTES));
}

async function contentOf(payload: Uint8Array): Promise<Uint8Array> {
  const { version, frame } = payloadFrame(payload);
  const dictionary = DICTIONARIES[version];
  if (!dictionary) throw new Error('This share link needs a newer version of the FlatPPL playground.');
  return (await codec()).decompress(frame, dictionary, MAX_SHARE_CONTENT_BYTES);
}

/** Null when the bundle exceeds what a link can open. Its ZIP remains downloadable. */
async function encode(view: ShareView, members: ShareMember[]): Promise<ShareTokens | null> {
  // The container stores main.flatppl as a file, not as a link.
  if (members.find(member => member.name === 'main.flatppl')?.kind !== 'file') return null;
  const content = encodeContainer(view, members);
  if (content.length > MAX_SHARE_CONTENT_BYTES) return null;
  try { decodeContainer(content); } catch (error) {
    if (error instanceof ShareLimitError) return null;
    throw error;
  }
  const frame = (await codec()).compress(content, DICTIONARIES[VERSION], LEVEL,
    MAX_SHARE_PAYLOAD_BYTES + ZSTD_MAGIC.length - 1);
  if (!frame) return null;
  const payload = framePayload(VERSION, frame);
  const link = linkToken(payload);
  // Hand out only links that this decoder opens.
  const opened = await contentOf(tokenPayload(link));
  if (opened.length !== content.length || opened.some((byte, i) => byte !== content[i])) {
    throw new Error('The share codec failed to reproduce this bundle.');
  }
  return { link, qr: qrToken(payload) };
}

worker.addEventListener('message', async ({ data }) => {
  try {
    if (data.operation === 'encode' && Array.isArray(data.members)) {
      worker.postMessage({ id: data.id, tokens: await encode(data.view, data.members) });
    } else if (data.operation === 'decode' && typeof data.token === 'string') {
      const payload = tokenPayload(data.token);
      const { view, members } = decodeContainer(await contentOf(payload));
      worker.postMessage({ id: data.id, link: linkToken(payload), view, members },
        members.map(member => member.bytes.buffer as ArrayBuffer));
    } else {
      throw new Error('Invalid share codec request.');
    }
  } catch (error) {
    worker.postMessage({ id: data.id, error: error instanceof Error ? error.message : 'Invalid share link.' });
  }
});
