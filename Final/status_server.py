"""
status_server.py
================
Lightweight HTTP server that serves the GEO pipeline tracker dashboard
and a JSON API for live status polling.

Usage:
    python status_server.py          # default port 7654
    STATUS_PORT=8080 python status_server.py

Then open: http://localhost:7654
API:        http://localhost:7654/api/status
"""

import json
import os
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

ROOT       = Path(__file__).parent
OUTPUT_DIR = ROOT / "geo_output"
PORT       = int(os.environ.get("STATUS_PORT", 7654))


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass  # suppress per-request noise in terminal

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"

        if path in ("/", "/index.html"):
            self._send_file(ROOT / "pipeline_tracker.html", "text/html; charset=utf-8")

        elif path == "/api/status":
            self._send_status()

        elif path == "/api/summary":
            self._send_file(OUTPUT_DIR / "pipeline_summary.json",
                            "application/json; charset=utf-8")

        else:
            self.send_error(404, "Not found")

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _send_status(self):
        result: dict = {
            "pipeline_status":  None,
            "pipeline_summary": None,
        }

        status_path  = OUTPUT_DIR / "pipeline_status.json"
        summary_path = OUTPUT_DIR / "pipeline_summary.json"

        for key, fpath in (("pipeline_status", status_path),
                           ("pipeline_summary", summary_path)):
            if fpath.exists():
                try:
                    with open(fpath, encoding="utf-8") as f:
                        result[key] = json.load(f)
                except Exception:
                    pass

        body = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type",  "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404, f"File not found: {path.name}")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",   content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "no-cache")
        self.end_headers()
        self.wfile.write(body)


def run():
    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"[StatusServer] Dashboard: http://localhost:{PORT}")
    print(f"[StatusServer] API:       http://localhost:{PORT}/api/status")
    print("[StatusServer] Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[StatusServer] Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
