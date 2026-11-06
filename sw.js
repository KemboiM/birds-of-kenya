// Lets the app install and open offline. Change CACHE when the models change.
const CACHE = "birds-of-kenya-2.1";
const SHELL = ["./", "index.html", "manifest.webmanifest", "species_info.json",
               "models/labels.txt", "icons/icon-192.png", "icons/icon-512.png", "icons/apple-touch-icon.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// Models and the runtime are large and only change with a new version: cache first.
// Everything else: network first, cache when offline.
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const big = /\/(models|ort)\/|\.onnx$|\.wasm$|onnxruntime-web/.test(req.url);
  if (big) {
    e.respondWith(caches.open(CACHE).then(async (c) => {
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return res;
    }));
  } else {
    e.respondWith(fetch(req).then((res) => {
      if (res.ok && req.url.startsWith(self.location.origin)) {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy));
      }
      return res;
    }).catch(() => caches.match(req, { ignoreSearch: true })));
  }
});
