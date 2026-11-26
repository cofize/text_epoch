"""python -m service [--host 127.0.0.1] [--port 8000]"""
import argparse
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from .wsgi import analyzer, application


class Threaded(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class Quiet(WSGIRequestHandler):
    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"модели: {analyzer.model_version}{' (заглушка)' if analyzer.stub else ''}")
    for g, i in analyzer.dicts.info().items():
        print(f"словарь «{g}»: {i['markers']} слов{' (демо)' if i['demo'] else ''}")
    for w in analyzer.dicts.warnings:
        print("внимание:", w)
    print(f"http://{a.host}:{a.port}/")
    make_server(a.host, a.port, application, server_class=Threaded,
                handler_class=Quiet).serve_forever()
