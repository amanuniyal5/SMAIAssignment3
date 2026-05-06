// ═══════════════════════════════════════════════════════════════
// CropGuard AI — Service Worker
// Caches all app assets + model for full offline operation
// ═══════════════════════════════════════════════════════════════

const CACHE_NAME = "cropguard-v1";
const STATIC_ASSETS = [
  "./index.html",
  "./manifest.json",
  "./class_names.json",
  "./disease_info.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./model.onnx",
  "./model.onnx.data"
  // ONNX Runtime Web WASM files get cached on first fetch
];

// ── Install: pre-cache static assets ──────────────────────────
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("[SW] Pre-caching app shell");
      return cache.addAll(STATIC_ASSETS).catch((err) => {
        console.warn("[SW] Some assets failed to pre-cache:", err);
      });
    })
  );
  self.skipWaiting();
});

// ── Activate: clean up old caches ─────────────────────────────
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== CACHE_NAME)
          .map((k) => {
            console.log("[SW] Deleting old cache:", k);
            return caches.delete(k);
          })
      )
    )
  );
  self.clients.claim();
});

// ── Fetch: cache-first for assets, network-first for API ──────
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests
  if (event.request.method !== "GET") return;

  // Network-first for Open-Meteo weather API
  if (url.hostname === "api.open-meteo.com") {
    event.respondWith(
      fetch(event.request)
        .then((res) => {
          const clone = res.clone();
          caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
          return res;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Cache-first for everything else (model, WASM, JSON, HTML)
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request)
        .then((res) => {
          if (!res || res.status !== 200 || res.type === "opaque") return res;
          const clone = res.clone();
          caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
          return res;
        })
        .catch(() => {
          // Offline fallback for navigation
          if (event.request.mode === "navigate") {
            return caches.match("./index.html");
          }
        });
    })
  );
});

// ── Background sync placeholder (future: sync field diary) ────
self.addEventListener("sync", (event) => {
  if (event.tag === "sync-diary") {
    console.log("[SW] Background sync: diary");
  }
});
