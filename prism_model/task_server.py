"""Loopback-only review app for a frozen, calibrated CPU model."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

MAX_BODY = 65536
ROOT = Path(__file__).resolve().parents[1]


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runtime):
        if address[0] != '127.0.0.1':
            raise ValueError('The review app must bind to 127.0.0.1')
        self.runtime = runtime
        self.inference_lock = threading.Lock()
        self.page = (Path(__file__).with_name('review.html')).read_bytes()
        super().__init__(address, ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = 'RequisitionReady/0.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *_):
        pass  # Request text and user input stay out of server logs.

    def send_content(self, status, data, content_type='application/json; charset=utf-8'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, value):
        self.send_content(status, json.dumps(value, allow_nan=False).encode())

    def local_request(self):
        port = self.server.server_port
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in allowed:
            self.send_json(403, {'error': 'Use the local review app address'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://{host}' for host in allowed}:
            self.send_json(403, {'error': 'Cross-origin requests are not accepted'})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        if path == '/':
            self.send_content(200, self.server.page, 'text/html; charset=utf-8')
        elif path == '/model':
            self.send_json(200, self.server.runtime.metadata())
        elif path == '/health':
            self.send_json(200, {'status': 'ready', 'device': 'cpu'})
        else:
            self.send_json(404, {'error': 'Not found'})

    def do_POST(self):
        if not self.local_request():
            return
        if self.path != '/evaluate-item':
            self.send_json(404, {'error': 'Not found'})
            return
        if self.headers.get('Transfer-Encoding'):
            self.send_json(400, {'error': 'Use a Content-Length request'})
            return
        try:
            length = int(self.headers.get('Content-Length', '-1'))
        except ValueError:
            length = -1
        if length < 0:
            self.send_json(411, {'error': 'Content-Length is required'})
            return
        if length > MAX_BODY:
            self.send_json(413, {'error': 'Request exceeds 64 KiB'})
            return
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            self.send_json(415, {'error': 'Use application/json'})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('Incomplete request')
            request = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
        except (ValueError, UnicodeError):
            self.send_json(400, {'error': 'Invalid JSON request'})
            return
        try:
            # No public overrides for policy, references, thresholds, or test fixtures.
            with self.server.inference_lock:
                result = self.server.runtime.evaluate(request)
            self.send_json(result['http_status'], result)
        except Exception:
            self.send_json(503, {'error': 'Evaluation unavailable', 'status': 'evaluation_unavailable'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / '.cache/task-v1/model-v0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    import torch
    from .task_runtime import TaskRuntime
    torch.set_num_interop_threads(1)
    runtime = TaskRuntime(args.bundle, threads=args.threads)
    with ReviewServer(('127.0.0.1', args.port), runtime) as server:
        print(json.dumps({'event': 'review_app_ready', 'url': f'http://127.0.0.1:{server.server_port}',
                          'artifact_identity': runtime.artifact_identity, 'device': 'cpu'}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
