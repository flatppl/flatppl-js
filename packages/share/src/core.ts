/**
 * Share-link codec for the playground's worker and the VS Code extension
 * host: token text, Zstandard and the member container, for untrusted input.
 * The format is described in ../../web/share-links.md.
 */
import dictionary1 from './dictionaries/v1.zdict';
import { loadZstd, type Zstd } from './zstd.ts';
import {
  MAX_SHARE_CONTENT_BYTES, MAX_SHARE_PAYLOAD_BYTES, ShareLimitError,
  ZSTD_MAGIC, decodeContainer, encodeContainer, framePayload, linkToken, payloadFrame, qrToken, tokenPayload,
  type ShareMember, type ShareView,
} from './format.ts';
import type { ShareTokens } from './links.ts';

/** Payload version to dictionary. Never change or reuse a version: printed links must keep opening. */
const DICTIONARIES: Record<number, Uint8Array> = { 1: dictionary1 };
const VERSION = 1;
const LEVEL = 19;
/** Level 19 needs between 64 and 128 MiB for the largest content. */
const MAX_HEAP_BYTES = 128 * 1024 * 1024;

export interface DecodedShare {
  /** The link token for the same payload, also when a QR token was decoded. */
  link: string;
  view: ShareView;
  members: ShareMember[];
}

/** `wasm` supplies the libzstd binary on first use. */
export function createShareCore(wasm: () => Promise<BufferSource>) {
  let zstd: Promise<Zstd> | undefined;
  function codec(): Promise<Zstd> {
    return zstd ||= wasm().then(bytes => loadZstd(bytes, MAX_HEAP_BYTES));
  }

  async function contentOf(payload: Uint8Array): Promise<Uint8Array> {
    const { version, frame } = payloadFrame(payload);
    const dictionary = DICTIONARIES[version];
    if (!dictionary) throw new Error('This share link needs a newer version of the FlatPPL playground.');
    return (await codec()).decompress(frame, dictionary, MAX_SHARE_CONTENT_BYTES);
  }

  return {
    /** Throws ShareLimitError, naming the limit, when the bundle exceeds what a link can open. */
    async encode(view: ShareView, members: ShareMember[]): Promise<ShareTokens> {
      if (members.find(member => member.name === 'main.flatppl')?.kind !== 'file') {
        throw new ShareLimitError('A share link stores main.flatppl as a file, not as a link.');
      }
      const content = encodeContainer(view, members);
      if (content.length > MAX_SHARE_CONTENT_BYTES) throw new ShareLimitError('The bundle exceeds 8 MiB of content.');
      try { decodeContainer(content); } catch (error) {
        if (error instanceof ShareLimitError) {
          throw new ShareLimitError('The bundle exceeds 1,000 members, or 1,024 bytes in a name, a link target or the view.');
        }
        throw error;
      }
      const frame = (await codec()).compress(content, DICTIONARIES[VERSION], LEVEL,
        MAX_SHARE_PAYLOAD_BYTES + ZSTD_MAGIC.length - 1);
      if (!frame) throw new ShareLimitError('The compressed bundle exceeds 49,152 bytes.');
      const payload = framePayload(VERSION, frame);
      const link = linkToken(payload);
      // Hand out only links that this decoder opens.
      const opened = await contentOf(tokenPayload(link));
      if (opened.length !== content.length || opened.some((byte, i) => byte !== content[i])) {
        throw new Error('The share codec failed to reproduce this bundle.');
      }
      return { link, qr: qrToken(payload) };
    },
    async decode(token: string): Promise<DecodedShare> {
      const payload = tokenPayload(token);
      const { view, members } = decodeContainer(await contentOf(payload));
      return { link: linkToken(payload), view, members };
    },
  };
}

