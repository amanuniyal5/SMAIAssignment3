"""
serve_pwa.py — local HTTPS-like server for testing the CropGuard PWA
Run: python serve_pwa.py
Then open: http://localhost:8080
"""
import http.server, socketserver, os

PORT = 8080
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required for SharedArrayBuffer (ONNX WASM multi-thread)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"[PWA] {self.address_string()} — {fmt % args}")

print(f"[CropGuard PWA] Serving at http://localhost:{PORT}")
print("[CropGuard PWA] Open that URL in Chrome/Edge to test the PWA")
print("[CropGuard PWA] Press Ctrl+C to stop\n")
with http.server.ThreadingHTTPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
