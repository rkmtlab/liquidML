from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse as parser
import json
import brain

class Handler(SimpleHTTPRequestHandler):
  def do_POST(self):
    self.data_string = self.rfile.read(int(self.headers['Content-Length']))
    self.send_response(200)
    data = self.data_string.decode('utf-8')
    data = parser.parse_qs(data)
    if not 'q' in data.keys():
      self.send_response(500)
      self.end_headers()
      return
    line = data['q'][0]
    rst = brain.learn_and_answer(line)
    rst = json.dumps(rst)
    body = bytes(rst, 'utf-8')
    # print('returning', rst)
    self.end_headers()
    self.wfile.write(body)
    return

HOST = 'localhost'
PORT = 8000
httpd = HTTPServer((HOST, PORT), Handler)
print('Backend brain alive 🔥 ', HOST, ':', PORT)
httpd.serve_forever()
