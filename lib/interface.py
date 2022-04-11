import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from json import dumps, loads

from lib.common import CoreFunc
class TargetHTTPHandler(BaseHTTPRequestHandler):
    mainapp = CoreFunc()
    def do_HEAD(self, code=200):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        data = {'result': 'this is Server'}
        self.do_HEAD()
        self.wfile.write(dumps(data).encode())

    def do_POST(self):
        output = dict()
        content_length = int(self.headers['Content-Length'])
        recv_data = self.rfile.read(content_length).decode('utf-8')
        try:
            format_data = loads(recv_data)
            print("format_data:",format_data)
            retval = TargetHTTPHandler.mainapp.core_func(format_data)
            # retval = TargetHTTPHandler.mainapp.test_func(format_data)
            if(retval):
                self.do_HEAD()
            else:
                retval = {"status": 500,"error": "Wrong input parameters"}
                self.do_HEAD(500)
            data = dumps(retval, indent=4, separators=(
                    ',', ':'), ensure_ascii=False)
            self.wfile.write(data.encode())
        except Exception as e:
            self.do_HEAD(500)
            edata = {
                "status": 500,
                "error": "Wrong data",
                "message": e,
                "path": "/"
            }
            jedata = dumps(edata)
            self.wfile.write(jedata.encode())


def interface(host, port):
    with ThreadingHTTPServer((host, port), TargetHTTPHandler) as server:
        server.serve_forever()
        server.server_close()
