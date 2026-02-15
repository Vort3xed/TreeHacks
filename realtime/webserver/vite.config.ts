import { defineConfig } from 'vite';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        viewer: resolve(__dirname, 'viewer.html'),
        sender: resolve(__dirname, 'sender.html'),
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    https: {
      key: fs.readFileSync(resolve(__dirname, 'server.key')),
      cert: fs.readFileSync(resolve(__dirname, 'server.cert')),
    },
  },
  css: {
    postcss: './postcss.config.js',
  },
});
