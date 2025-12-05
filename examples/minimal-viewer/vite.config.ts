import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      'quickfix-renderer': path.resolve(__dirname, '../../quickfix-renderer/pkg/quickfix_renderer.js'),
    },
  },
  plugins: [
    react(),
    wasm(),
    topLevelAwait()
  ],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
  },
  worker: {
    plugins: () => [
      wasm(),
      topLevelAwait()
    ]
  },
  optimizeDeps: {
    exclude: ['quickfix-renderer']
  },
  server: {
    fs: {
      allow: ['../..']
    }
  }
});
