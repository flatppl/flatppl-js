# Share-link format

A share link carries a whole bundle in its URL fragment. Links end up in talks
and on posters, so the playground must open every released version forever.
Never change a released version or reuse its number.

## Links and QR codes

A link is `#share=` followed by a link token. A QR code holds `#` followed by a
QR token. Both tokens encode the same payload, which holds the bundle and the
member and binding to open.

- A **link token** is the payload in base64url (RFC 4648, section 5), without
  padding.
- A **QR token** is `~` followed by the payload, read as one unsigned
  big-endian integer, in decimal. QR codes store digits in numeric mode, which
  costs 8.03 bits per payload byte, against 8.8 for Base32 in alphanumeric
  mode. QR codes also upper-case scheme and host, which then fit alphanumeric
  mode.

The `share=` parameter accepts either token. The `~` decides the form, since
digits also belong to the base64url alphabet. Decoders accept only the
canonical spelling: base64url without padding and with zero unused bits,
decimal without leading zeros.

After opening, the playground shows its own route,
`#/files/<member>?target=<binding>&share=<link token>`. While the route names a
member, it takes precedence over the view in the payload.

## Payload

The first payload byte is the version. The rest is a Zstandard frame
(RFC 8878) without its four-byte magic number, compressed with that version's
dictionary. The frame is a single segment and declares its content size. It
has no checksum and no dictionary ID, and nothing follows it.

| Limit | Value |
|---|---|
| Payload | 49,152 bytes (a 64 KiB link) |
| QR token | 5,596 digits (QR version 40, medium error correction) |
| Content | 8 MiB |

## Container

The content holds the view and the bundle members:

```text
container = length model length target length main { member }
member    = kind length name length data
kind      = 0 (file) | 1 (directory) | 2 (symbolic link)
length    = unsigned LEB128, at most four bytes, no redundant zero byte
```

Lengths count bytes. Text is UTF-8, decoded strictly and without removing a
leading byte-order mark. `model` is the member to open and empty for
`main.flatppl`. It never is `main.flatppl` and never ends in `/`. `target` is
the binding to show, or empty. `main` holds the bytes of `main.flatppl`, which
is always a file: a bundle whose `main.flatppl` is a link has no share link.
Names are paths relative to the bundle root, without a trailing slash. They
ascend strictly in JavaScript string order, by UTF-16 code units, and never
repeat `main.flatppl`. A directory has no data. A link's data is its target
and never empty. Decoders parse the whole container or reject it.

The wire decoder accepts at most 1,000 members, `main.flatppl` included. View fields,
names and link targets hold at most 1,024 bytes, a directory name counted with
a trailing slash. The playground then rebuilds a STORE ZIP and opens it like
any bundle, so the archive rules of [bundles.md](bundles.md) also apply, with
the stricter browser limits: 256 entries across nested archives, four levels,
2 MiB per member and 8 MiB expanded content. The rebuilt ZIP may use up to 9 MiB
including headers. Dependency graph capture also budgets for the root ZIP;
mounted reads retain the member and expansion limits. Before the playground
hands out a link, it opens the link this way itself, selects its view and resolves
the view's dependencies. A bundle that fails gets no link. Incoming links show
source until the visitor chooses **Run preview**.
A container holding only the regular `main.flatppl` opens as one file. This
changes its presentation, not the archive rules or version-1 wire format.

## Versions

| Version | Dictionary | Bytes | SHA-256 |
|---|---|---|---|
| 1 | `src/share-dictionaries/v1.zdict` | 65,533 | `25808cf2f2261b2012104ef3677bc7d70b38fdee3205c1b5f7ff9a89fb8d216e` |

Add a version only for a gain measured on models the dictionary was not
trained on. Version 1 used leave-one-out over the gallery: each example
compressed with a dictionary trained without it.

### Version 1

The training corpus holds the committed `.flatppl` files of
flatppl-examples, flatppl-js, flatppl-rust and flatppl-testsuite, and the
`flatppl` code blocks of the specification in flatppl-design `docs/`, without
duplicates: 579 samples with 268,803 bytes. It leaves out the HS3 conversions
in flatppl-testsuite `corpora/hs3`, which made both the gallery and the HS3
files themselves compress worse.

1. `zstd --train -19 --maxdict=65536` (zstd 1.5.7) trained a dictionary on the
   corpus.
2. Vocabulary that the trained content lacks went in front of it, and the
   trained content lost as many bytes at its start, where the trainer puts its
   least useful segments. The vocabulary consists of the 124 keywords and
   builtin names of flatppl-grammars (`keyword-lists.json` and
   `cspell/flatppl-words.txt`) that the corpus never uses, each as `name(`, or
   as `name` and a newline for constants, predefined sets and reserved names.
   It also holds the member-name parts `modules/`, `.csv` and `.wsv`: bundle
   export puts imports in `modules/`, and `load_data` reads CSV and WSV files.
3. `ZDICT_finalizeDictionary` of libzstd 1.5.7 rebuilt the entropy tables for
   level 19 from the same samples.
4. The dictionary ID was set to 0, so frames carry none. The version byte
   selects the dictionary.

| Repository | Commit |
|---|---|
| flatppl-examples | `90867a12693b4db0bb5c3b654c0ec07153d0edb6` |
| flatppl-js | `d5cc8741f91bc724f78acb5903eaeb8c9eae11c3` |
| flatppl-rust | `3e60a0585dea70c801d1501e2b75085ca082d96a` |
| flatppl-testsuite | `358cb36a6246812960d98c3c893187c0633180bd` |
| flatppl-design | `0e550d6145d85b23f93f5401e1533b19f4f6638b` |
| flatppl-grammars | `0163345a74cffa24b4d10229155d8f7e4afa7cb1` |

The design study measured, leave-one-out over the 15 gallery examples with at
least 1,500 bytes, 12,133 compressed bytes where Brotli over a ZIP without a
dictionary needed 16,674. Links became 21% shorter, and QR tokens needed 28%
fewer bits. The grammar vocabulary of step 2 changed the gallery by −0.1% and
saved about 3.5 bytes on the first use of a builtin that the corpus never uses.
C Brotli with the same dictionary was 1.7% smaller, but browser builds of
Brotli either lack custom dictionaries or compress 3–5% worse.
