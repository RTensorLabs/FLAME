// Runs the FLAME model in a background thread so the page stays responsive.
//
// Messages in:  {type: "load"} | {type: "generate", prompt, maxTokens, temperature, topK, seed} | {type: "stop"}
// Messages out: {type: "progress", loaded, total} | {type: "ready", info, maxSeqLen, fromCache}
//               {type: "token", text, n, seconds} | {type: "done", reason, n, seconds} | {type: "error", message}
import init, { Flame } from "./pkg/flame_wasm.js";

const FILES = {
  config: "model/config.toml",
  tokenizer: "model/tokenizer.json",
  weights: "model/model.safetensors",
};
// Bump when the model files change so visitors don't keep a stale cached copy.
const CACHE_NAME = "flame-model-v1";

let flame = null;
let stopRequested = false;

async function openCache() {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null; // Cache API unavailable (e.g. private window); just download every time
  }
}

// Downloads a file, reporting byte progress; stores it in the Cache API for the next visit.
async function fetchBytes(url, cache, onProgress) {
  const cached = cache && (await cache.match(url));
  if (cached) {
    const buf = new Uint8Array(await cached.arrayBuffer());
    onProgress(buf.length, buf.length);
    return { bytes: buf, fromCache: true };
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed: ${url} (HTTP ${res.status})`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    bytes.set(c, off);
    off += c.length;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(bytes));
    } catch {
      /* quota exceeded: fine, it just downloads again next time */
    }
  }
  return { bytes, fromCache: false };
}

async function load() {
  await init();
  const cache = await openCache();
  const text = async (url) => new TextDecoder().decode((await fetchBytes(url, cache, () => {})).bytes);
  const [config, tokenizer] = await Promise.all([text(FILES.config), text(FILES.tokenizer)]);
  const { bytes, fromCache } = await fetchBytes(FILES.weights, cache, (loaded, total) =>
    postMessage({ type: "progress", loaded, total })
  );
  flame = new Flame(config, tokenizer, bytes);
  postMessage({ type: "ready", info: flame.describe(), maxSeqLen: flame.maxSeqLen(), fromCache });
}

// Lets queued messages (like "stop") run between tokens.
const yieldToEvents = () => new Promise((r) => setTimeout(r, 0));

async function generate({ prompt, maxTokens, temperature, topK, seed }) {
  stopRequested = false;
  flame.start(prompt, temperature, topK, seed);
  const t0 = performance.now();
  const elapsed = () => (performance.now() - t0) / 1000;
  for (let i = 0; i < maxTokens; i++) {
    await yieldToEvents();
    if (stopRequested) {
      postMessage({ type: "done", reason: "stopped", n: flame.generatedTokens(), seconds: elapsed() });
      return;
    }
    const text = flame.step();
    if (text === undefined) {
      postMessage({ type: "done", reason: "end-of-text", n: flame.generatedTokens(), seconds: elapsed() });
      return;
    }
    postMessage({ type: "token", text, n: flame.generatedTokens(), seconds: elapsed() });
  }
  postMessage({ type: "done", reason: "max-tokens", n: flame.generatedTokens(), seconds: elapsed() });
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load();
    else if (data.type === "generate") await generate(data);
    else if (data.type === "stop") stopRequested = true;
  } catch (e) {
    postMessage({ type: "error", message: String(e && e.message ? e.message : e) });
  }
};
