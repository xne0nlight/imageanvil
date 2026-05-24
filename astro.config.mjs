import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://imageanvil.com',
  vite: {
    // The jSquash libraries use WebAssembly. These settings help Vite handle them correctly.
    optimizeDeps: {
      exclude: ['heic-to', '@jsquash/avif', '@jsquash/webp', '@jsquash/jpeg', '@jsquash/png'],
    },
    assetsInclude: ['**/*.wasm'],
  },
});
