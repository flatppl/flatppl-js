// Multi-window tests of the playground (test/e2e). They run against the
// built dist/ through serve.mjs, so build first: `npm run build`.
import { defineConfig, devices } from '@playwright/test';

const port = Number(process.env.E2E_PORT || 8041);

export default defineConfig({
  testDir: './test/e2e',
  testMatch: '**/*.spec.mjs',
  timeout: 90000,
  expect: { timeout: 20000 },
  fullyParallel: false,
  workers: process.env.CI ? 1 : 2,
  retries: 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${port}/`,
    viewport: { width: 1400, height: 900 },
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `node serve.mjs`,
    env: { PORT: String(port) },
    url: `http://127.0.0.1:${port}/index.html`,
    reuseExistingServer: false,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1400, height: 900 } } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'], viewport: { width: 1400, height: 900 } } },
    { name: 'webkit', use: { ...devices['Desktop Safari'], viewport: { width: 1400, height: 900 } } },
  ],
});
