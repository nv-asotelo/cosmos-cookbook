#!/usr/bin/env python3
"""Tiny OpenAI-compatible round-robin proxy for local benchmark replicas."""
from __future__ import annotations

import itertools
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


TARGETS = [item.rstrip("/") for item in os.environ.get("OPENAI_PROXY_TARGETS", "http://127.0.0.1:8000,http://127.0.0.1:8001").split(",") if item.strip()]
HOST = os.environ.get("OPENAI_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("OPENAI_PROXY_PORT", "8010"))
_cycle = itertools.cycle(TARGETS)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _target_for(self) -> str:
        if self.command == "GET":
            return TARGETS[0]
        return next(_cycle)

    def _forward(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else None
        target = self._target_for()
        url = f"{target}{self.path}"
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
        }
        request = urllib.request.Request(url, data=body, method=self.command, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=float(os.environ.get("OPENAI_PROXY_TIMEOUT", "900"))) as response:
                data = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in {"transfer-encoding", "connection", "content-encoding"}:
                        self.send_header(key, value)
                self.send_header("content-length", str(len(data)))
                self.send_header("x-upstream-target", target)
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as exc:
            data = exc.read()
            self.send_response(exc.code)
            for key, value in exc.headers.items():
                if key.lower() not in {"transfer-encoding", "connection", "content-encoding"}:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(data)))
            self.send_header("x-upstream-target", target)
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            data = (f'{{"error":"proxy upstream failure","target":"{target}","detail":"{str(exc)}"}}').encode("utf-8")
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.send_header("x-upstream-target", target)
            self.end_headers()
            self.wfile.write(data)

    do_GET = _forward
    do_POST = _forward

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))


def main() -> None:
    if not TARGETS:
        raise SystemExit("OPENAI_PROXY_TARGETS is empty")
    server = ThreadingHTTPServer((HOST, PORT), ProxyHandler)
    print(f"openai_round_robin_proxy listening on {HOST}:{PORT} -> {', '.join(TARGETS)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
