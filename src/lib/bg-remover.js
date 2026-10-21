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

function toInputTensor(ort, bitmap, type) {
  const c = makeCanvas(SIZE, SIZE);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
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

function logitsToMaskCanvas(tensor) {
  let values = tensor.data;
  if (tensor.type === 'float16' && values instanceof Uint16Array) values = float16ToFloat32(values);
  const n = SIZE * SIZE;
  const img = new ImageData(SIZE, SIZE);
  for (let i = 0; i < n; i++) {
    const a = 1 / (1 + Math.exp(-values[i])); // sigmoid
    img.data[i * 4 + 3] = Math.round(a * 255);
  }
  const c = makeCanvas(SIZE, SIZE);
  c.getContext('2d').putImageData(img, 0, 0);
  return c;
}

/**
 * Remove the background from an ImageBitmap.
 * @returns {Promise<{cutout: HTMLCanvasElement, width: number, height: number}>}
 */
export async function removeBackground(bitmap, onProgress) {
  const { ort, session } = await getSession(onProgress);
  const inMeta = session.inputMetadata?.[0];
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const inputType = inMeta?.type || 'float32';

  const tensor = toInputTensor(ort, bitmap, inputType);
  const results = await session.run({ [inputName]: tensor });
  const mask = logitsToMaskCanvas(results[outputName]);
  tensor.dispose?.();
  results[outputName].dispose?.();

  // Output at original resolution (capped for sanity)
  let w = bitmap.width, h = bitmap.height;
  const scale = Math.min(1, MAX_OUTPUT_SIDE / Math.max(w, h));
  w = Math.round(w * scale); h = Math.round(h * scale);

  const cutout = makeCanvas(w, h);
  const ctx = cutout.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, w, h);
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(mask, 0, 0, w, h); // bilinear upscale of the soft mask
  ctx.globalCompositeOperation = 'source-over';
  return { cutout, width: w, height: h };
}
