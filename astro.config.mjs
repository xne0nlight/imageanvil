import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://imageanvil.com',
  trailingSlash: 'always',
  vite: {
    // The jSquash libraries use WebAssembly. These settings help Vite handle them correctly.
    optimizeDeps: {
      exclude: ['heic-to', '@jsquash/avif', '@jsquash/webp', '@jsquash/jpeg', '@jsquash/png'],
    },
    assetsInclude: ['**/*.wasm'],
    // jSquash's multithreaded encoders run in Web Workers. Bundling them as ES
    // modules (instead of the default IIFE) is required for the production build,
    // which uses code-splitting that IIFE/UMD formats don't support.
    worker: {
      format: 'es',
    },
  },
});
