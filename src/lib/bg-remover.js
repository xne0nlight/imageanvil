/**
 * Image Anvil — Background removal engine
 * ---------------------------------------------------------------
 * Runs the BiRefNet-lite segmentation model (MIT license) entirely
 * in the browser with ONNX Runtime Web on WebGPU. Nothing is uploaded:
 * the only network traffic is the one-time download of the runtime and model.
 *
 * Pipeline:
 *   image → 1024×1024 RGB tensor (ImageNet-normalized)
 *         → BiRefNet → 1024×1024 logits → sigmoid → alpha mask
 *         → mask scaled back up to full resolution → applied as alpha
 *
 * Why WebGPU-only: the stock onnx-community export has ops with no WebGPU
 * kernel (variadic Sum, CPU-only shape ops). ORT falls back to the CPU for
 * them, which allocates ~784 MB tensors and exhausts the 4 GB WASM heap
 * (std::bad_alloc). We use a graph-repaired build of the same weights
 * (fp16 weights, float32 I/O) that runs fully on the GPU. A pure-CPU run
 * of a model this size hits the same memory wall, so there's no CPU path.
 */

// ——— Where things load from ———————————————————————————————————
// The ORT runtime is loaded from jsDelivr instead of being bundled:
// its WebGPU .wasm file is ~28 MB, over Cloudflare Pages' 25 MB limit.
// 1.27.0 + the JSEP build (ort.webgpu.min.mjs) is what this model was verified on.
const ORT_VERSION = '1.27.0';
const ORT_DIR = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_URL = `${ORT_DIR}ort.webgpu.min.mjs`;

// Same BiRefNet_lite weights (MIT), graph-repaired for WebGPU:
// https://huggingface.co/runes/birefnet-lite-webgpu
// It's a community upload, so consider mirroring it to R2 and changing MODEL_URL
// (the bucket needs a CORS rule allowing GET from https://imageanvil.com).
const MODEL_URL = 'https://huggingface.co/runes/birefnet-lite-webgpu/resolve/main/birefnet_lite_webgpu_fp16.onnx';
export const MODEL = { url: MODEL_URL, approxBytes: 123_000_000 };

const CACHE_NAME = 'imageanvil-models-v2';
const SIZE = 1024; // BiRefNet input resolution
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// Huge photos get capped so we don't blow past browser canvas limits.
const MAX_OUTPUT_SIDE = 6000;

let ortPromise = null;
let sessionPromise = null;

// ——— Runtime ————————————————————————————————————————————————————
async function loadOrt() {
  if (!ortPromise) {
    ortPromise = import(/* @vite-ignore */ ORT_URL).then((mod) => {
      const ort = mod.default ?? mod;
      ort.env.wasm.wasmPaths = ORT_DIR;
      ort.env.wasm.numThreads = 1; // GPU does the work; avoids needing COOP/COEP headers
      ort.env.webgpu.powerPreference = 'high-performance'; // prefer the discrete GPU on laptops
      ort.env.logLevel = 'error';
      return ort;
    }).catch((err) => { ortPromise = null; throw err; });
  }
  return ortPromise;
}

/**
 * Can this device run the model?
 * @returns {Promise<{supported: boolean, reason?: 'no-webgpu'|'no-f16'}>}
 */
export async function detectEngine() {
  try {
    if (!navigator.gpu) return { supported: false, reason: 'no-webgpu' };
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { supported: false, reason: 'no-webgpu' };
    if (!adapter.features.has('shader-f16')) return { supported: false, reason: 'no-f16' };
    return { supported: true };
  } catch {
    return { supported: false, reason: 'no-webgpu' };
  }
}

// ——— Model download (with progress + browser cache) ———————————————
async function fetchModel(url, approxBytes, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onProgress?.({ loaded: buf.length, total: buf.length, cached: true });
      return buf;
    }
  } catch { cache = null; /* Cache API unavailable (e.g. some private modes) */ }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Model download failed (HTTP ${res.status})`);
  const total = Number(res.headers.get('content-length')) || approxBytes;

  let bytes;
  if (res.body && res.body.getReader) {
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress?.({ loaded, total: Math.max(total, loaded), cached: false });
    }
    bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
    onProgress?.({ loaded: bytes.length, total: bytes.length, cached: false });
  }

  try {
    await cache?.put(url, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
  } catch { /* storage quota full — still works, just re-downloads next visit */ }
  return bytes;
}

/** Is the model already in the browser cache? Also clears the old v1 cache. */
export async function isModelCached() {
  try {
    caches.delete('imageanvil-models-v1').catch(() => {}); // the old (broken) model files
    const cache = await caches.open(CACHE_NAME);
    return !!(await cache.match(MODEL.url));
  } catch { return false; }
}

/** Load (once) and return the model session on WebGPU. */
export function getSession(onProgress) {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await loadOrt();
      const bytes = await fetchModel(MODEL.url, MODEL.approxBytes, onProgress);
      onProgress?.({ phase: 'init' });
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
      return { ort, session };
    })().catch((err) => { sessionPromise = null; throw err; });
  }
  return sessionPromise;
}

// ——— float16 helpers (WebGPU fp16 models take/return half floats) ———————
const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);

export function float32ToFloat16(src) {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) {
    f32buf[0] = src[i];
    const x = u32buf[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = ((x >>> 23) & 0xff) - 127 + 15;
    const mant = x & 0x7fffff;
    if (exp <= 0) out[i] = sign;                   // tiny → 0 (fine for pixel values)
    else if (exp >= 31) out[i] = sign | 0x7c00;    // overflow → ±Inf
    else {
      let h = sign | (exp << 10) | (mant >> 13);
      if (mant & 0x1000) h++;                      // round to nearest
      out[i] = h;
    }
  }
  return out;
}

let f16Lut = null;
export function float16ToFloat32(src) {
  if (!f16Lut) {
    f16Lut = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const m = h & 0x3ff;
      f16Lut[h] = e === 0 ? s * 2 ** -14 * (m / 1024)
        : e === 31 ? (m ? NaN : s * Infinity)
        : s * 2 ** (e - 15) * (1 + m / 1024);
    }
  }
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = f16Lut[src[i]];
  return out;
}

// ——— Inference ————————————————————————————————————————————————
function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

/**
 * Draw the image into a canvas at output resolution (capped for sanity).
 * This is the source every later step works from.
 */
export function prepareSource(bitmap) {
  const scale = Math.min(1, MAX_OUTPUT_SIDE / Math.max(bitmap.width, bitmap.height));
  const c = makeCanvas(Math.round(bitmap.width * scale), Math.round(bitmap.height * scale));
  const ctx = c.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, c.width, c.height);
  return c;
}

/** Median color of the image's outer edge — a good guess at the background. */
function borderColor(source) {
  const n = 64;
  const c = makeCanvas(n, n);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, n, n);
  const { data } = ctx.getImageData(0, 0, n, n);
  const r = [], g = [], b = [];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      if (x > 1 && x < n - 2 && y > 1 && y < n - 2) continue; // 2px border ring only
      const i = (y * n + x) * 4;
      r.push(data[i]); g.push(data[i + 1]); b.push(data[i + 2]);
    }
  }
  const med = (a) => a.sort((p, q) => p - q)[a.length >> 1];
  return `rgb(${med(r)}, ${med(g)}, ${med(b)})`;
}

/**
 * Build the 1024×1024 model input. With pad > 0 the image is shrunk inside
 * a border of its own background color, giving a frame-filling subject
 * visible edges. Returns the canvas plus where the image sits inside it.
 */
function buildInputCanvas(source, pad, mirror) {
  const c = makeCanvas(SIZE, SIZE);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const inner = SIZE / (1 + 2 * pad);
  const offset = (SIZE - inner) / 2;
  if (pad > 0) { ctx.fillStyle = borderColor(source); ctx.fillRect(0, 0, SIZE, SIZE); }
  if (mirror) { ctx.translate(SIZE, 0); ctx.scale(-1, 1); }
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, offset, offset, inner, inner); // squish-resize, as the model was trained
  return { canvas: c, crop: { x: offset, y: offset, w: inner, h: inner } };
}

function canvasToTensor(ort, canvas, type) {
  const { data } = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, SIZE, SIZE);
  const n = SIZE * SIZE;
  const f32 = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f32[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
    f32[n + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    f32[2 * n + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  const dims = [1, 3, SIZE, SIZE];
  return type === 'float16'
    ? new ort.Tensor('float16', float32ToFloat16(f32), dims)
    : new ort.Tensor('float32', f32, dims);
}

async function runModel(source, pad, mirror) {
  const { ort, session } = await getSession();
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const inputType = session.inputMetadata?.[0]?.type || 'float32';

  const { canvas, crop } = buildInputCanvas(source, pad, mirror);
  const tensor = canvasToTensor(ort, canvas, inputType);
  const results = await session.run({ [inputName]: tensor });
  const out = results[outputName];
  let values = out.data;
  values = out.type === 'float16' && values instanceof Uint16Array ? float16ToFloat32(values) : Float32Array.from(values);
  tensor.dispose?.();
  out.dispose?.();
  return { logits: values, crop };
}

/**
 * Run the model and return raw logits (pre-sigmoid) for the image.
 * @param {HTMLCanvasElement} source  from prepareSource()
 * @param {{pad?: number, mirror?: boolean}} opts
 *   pad: 0–0.5, border added around the image (fraction of its size per side)
 *   mirror: also run on a mirrored copy and average — steadier, 2× the time
 */
export async function predict(source, { pad = 0, mirror = false } = {}) {
  const a = await runModel(source, pad, false);
  if (!mirror) return a;
  const b = await runModel(source, pad, true);
  // Un-mirror b and average. Crops are symmetric, so they line up.
  for (let y = 0; y < SIZE; y++) {
    const row = y * SIZE;
    for (let x = 0; x < SIZE; x++) a.logits[row + x] = (a.logits[row + x] + b.logits[row + SIZE - 1 - x]) / 2;
  }
  return a;
}

/**
 * Turn logits into a cutout canvas. Cheap enough to call while dragging sliders.
 * @param {{bias?: number, gain?: number}} adj
 *   bias: >0 keeps more of the image, <0 removes more
 *   gain: >1 crisper edges, <1 softer edges
 * @param {number} [maxSide] render smaller (e.g. for previews)
 */
export function renderCutout(source, pred, { bias = 0, gain = 1 } = {}, maxSide = Infinity) {
  const { logits, crop } = pred;
  const n = SIZE * SIZE;
  const img = new ImageData(SIZE, SIZE);
  for (let i = 0; i < n; i++) {
    const a = 1 / (1 + Math.exp(-(logits[i] * gain + bias)));
    img.data[i * 4 + 3] = (a * 255 + 0.5) | 0;
  }
  const mask = makeCanvas(SIZE, SIZE);
  mask.getContext('2d').putImageData(img, 0, 0);

  const s = Math.min(1, maxSide / Math.max(source.width, source.height));
  const w = Math.round(source.width * s), h = Math.round(source.height * s);
  const out = makeCanvas(w, h);
  const ctx = out.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, w, h);
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(mask, crop.x, crop.y, crop.w, crop.h, 0, 0, w, h); // soft mask, bilinear upscale
  ctx.globalCompositeOperation = 'source-over';
  return out;
}
