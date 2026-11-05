// Version-1 tokens that the playground and the extension both open and make.
// Never regenerate them: printed links of version 1 must keep opening.
const encode = text => new TextEncoder().encode(text);
const prior = encode('mu ~ Normal(0.0, 10.0)\n');

export const GOLDEN = [
  {
    name: 'single file',
    view: { model: '', target: '' },
    members: [{ name: 'main.flatppl', kind: 'file', bytes: encode('mu ~ Normal(0.0, 1.0)\n') }],
    link: 'ASAZZQAAKAAAFm11AQAvMs20',
    qr: '~98035076795119991792671533080778818899380',
  },
  {
    name: 'members with a directory and a link',
    view: { model: '', target: '' },
    members: [
      { name: 'main.flatppl', kind: 'file', bytes: encode('m = load_module("modules/prior.flatppl")\nx ~ Normal(m.mu, 1.0)\n') },
      { name: 'data', kind: 'directory', bytes: new Uint8Array() },
      { name: 'modules/prior.flatppl', kind: 'file', bytes: prior },
      { name: 'prior.flatppl', kind: 'symlink', bytes: encode('modules/prior.flatppl') },
    ],
    link: 'ASCdrQEAiAAAP3gubXUBBAAAFRcKAg0VDPwA6VAtZahuWx_wdYQi9IXxiaHAP8-bcodQ90_2OGM4lqI',
    qr: '~53702970339676262100563665257105057898337673562604515542193827743905569873152674775648934627703903240262374503351841823457693877582329058978',
  },
  {
    name: 'view with model and target',
    view: { model: 'modules/prior.flatppl', target: 'mu' },
    members: [
      { name: 'main.flatppl', kind: 'file', bytes: encode('m = load_module("modules/prior.flatppl")\n') },
      { name: 'modules/prior.flatppl', kind: 'file', bytes: prior },
    ],
    link: 'ASByJQEAQBUCbXUpABcKCfxfqHlOS_goUzjubk4movEaWbVNIfY3uTUz',
    qr: '~616116654124892862778121128305370117426646666735270036037955151123458768903948301651304777638294835',
  },
];

/** Members as the archive reader lists them, in name order, comparable with deepEqual. */
export function comparable(members) {
  return members.map(m => [m.name, m.kind, Array.from(m.bytes)]).sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
}
