# Image Anvil

Client-side image tools, forged for the web.

A web tool for converting HEIC, AVIF, WebP and other image formats, all running in the user's browser via WebAssembly. No uploads, no accounts, no tracking.

## Current tools

- **HEIC → AVIF** (live) — Convert iPhone photos to AVIF for efficient web delivery
- **AVIF Converter** (coming soon)
- **WebP Converter** (coming soon)
- **Batch Optimizer** (coming soon)

---

## Getting started

### 1. Check Node.js is installed

Open a terminal (Terminal on macOS, PowerShell or Command Prompt on Windows) and run:

```bash
node --version
```

You should see something like `v20.x.x` or `v18.x.x`. If you see a "command not found" error, install Node.js from [nodejs.org](https://nodejs.org) (pick the LTS version).

### 2. Install dependencies

In the project folder (wherever you extracted this zip), run:

```bash
npm install
```

This will download Astro, the jSquash image conversion libraries, and their dependencies into a `node_modules` folder. This takes 30–60 seconds the first time.

### 3. Start the dev server

```bash
npm run dev
```

You should see output like:

```
astro  v4.15.x ready in 432 ms

┃ Local    http://localhost:4321/
┃ Network  use --host to expose
```

Open [http://localhost:4321](http://localhost:4321) in your browser. You should see Image Anvil running locally. Every change you make to a file will hot-reload in the browser.

### 4. Build for production

When you're ready to deploy:

```bash
npm run build
```

This creates a `dist/` folder with the static site ready to upload anywhere — Cloudflare Pages, Vercel, Netlify, or even a basic static host.

---

## Project structure

```
imageanvil/
├── public/              Static assets served as-is
│   └── favicon.svg      Site favicon
│
├── src/
│   ├── components/      Reusable UI pieces
│   │   ├── Logo.astro       The pixel-transition anvil mark
│   │   ├── Header.astro     Site header with nav
│   │   └── Footer.astro     Site footer
│   │
│   ├── layouts/
│   │   └── BaseLayout.astro  Shared HTML shell (title, meta, header, footer)
│   │
│   ├── pages/           Each file = one URL
│   │   ├── index.astro         → /
│   │   └── heic-to-avif.astro  → /heic-to-avif (the working converter)
│   │
│   └── styles/
│       └── global.css   Design system (colors, type, spacing, components)
│
├── astro.config.mjs     Astro + Vite configuration
├── package.json         Dependencies
└── README.md            This file
```

---

## How the converter works

The HEIC → AVIF tool runs entirely in the user's browser using WebAssembly:

1. User drops an HEIC file onto the drop zone
2. The file is read into memory as an ArrayBuffer (never uploaded)
3. `@jsquash/heic` decodes it to raw pixel data (ImageData)
4. `@jsquash/avif` re-encodes those pixels as AVIF at the chosen quality
5. The resulting bytes are wrapped in a Blob and offered as a download

The WASM libraries are lazy-loaded only when the user actually drops a file, keeping the initial page load fast.

### To add more conversion pairs later

Copy the pattern in `src/pages/heic-to-avif.astro`. The `loadConverters()` function inside the `<script>` tag is where you wire up which jSquash packages to use. The packages already installed cover:

- `@jsquash/heic` (decode only — HEIC is input-only by design)
- `@jsquash/avif` (encode + decode)
- `@jsquash/webp` (encode + decode)
- `@jsquash/jpeg` (encode + decode)
- `@jsquash/png` (encode + decode)

So you can build HEIC→WebP, HEIC→JPG, AVIF↔WebP, JPG→AVIF, and so on from the same toolkit.

---

## Design system

All design tokens live in `src/styles/global.css` as CSS custom properties:

- **Colors**: Cool steel palette (slate-based) with a warm forge-glow accent (amber) used sparingly
- **Fonts**: Figtree (geometric sans) + JetBrains Mono (technical)
- **Spacing**: 4px-based scale
- **Components**: `.btn`, `.card`, `.badge` classes for consistency

To tweak the look sitewide, edit the CSS variables in `:root`.

---

## Troubleshooting

**"npm install" fails with permission errors**
Try running from a folder you own (e.g., `~/Projects/imageanvil` on macOS, `C:\Users\YourName\Projects\imageanvil` on Windows). Avoid system folders like `Program Files`.

**"Cannot find module '@jsquash/heic'"**
Run `npm install` again. If it still fails, delete the `node_modules` folder and `package-lock.json`, then `npm install` fresh.

**Conversion hangs forever**
Open the browser's dev tools (F12) and check the Console tab. WASM errors will show up there. HEIC files larger than ~50MB may time out — this is a known jSquash limitation.

**Hot reload stopped working**
Stop the dev server (Ctrl+C in the terminal) and restart with `npm run dev`.

---

## Deploying to Cloudflare Pages (recommended)

1. Push the project to a GitHub repository (I'll walk you through this when you're ready).
2. Log into [Cloudflare Pages](https://pages.cloudflare.com).
3. Click "Create a project" → "Connect to Git" → pick the repo.
4. Build settings:
   - **Framework preset**: Astro
   - **Build command**: `npm run build`
   - **Build output directory**: `dist`
5. Click Deploy. First build takes ~2 minutes.
6. Add your custom domain (imageanvil.com) in the Custom Domains tab.

Every git push to the main branch auto-deploys. Preview URLs for branches come free.

---

## Next steps to ship

Before applying to Ezoic or Mediavine for ads:

- [ ] Add About, Privacy Policy, and Contact pages (required)
- [ ] Write 3–5 guide articles in a `/blog/` section (topical authority for Google + AdSense approval)
- [ ] Verify site in Google Search Console, submit sitemap
- [ ] Run a Lighthouse audit — aim for 95+ across the board
- [ ] Expand to AVIF, WebP, and Batch tools (the core hub)

Good luck, and happy forging. 🔨
