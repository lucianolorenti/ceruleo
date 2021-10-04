import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


import tornado.ioloop
import tornado.web
import json
from front import TEMPLATES_PATH
from back import STATIC_PATH
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


 
class HelloHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

    def get_template_path(self):
        return str(TEMPLATES_PATH)
 
 
class ApiHandler(tornado.web.RequestHandler):
    def post(self):
        response = {"language": self.request.headers.get("Accept-Language", "")}
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(response))
 
 
def make_app():
    settings = {
    "static_path":str(STATIC_PATH),
    "static_url_prefix": "/static/",
}
    return tornado.web.Application([
        (r"/", HelloHandler),
        (r"/api", ApiHandler),
    ], **settings)
 
 
def main():
    app = make_app()
    logger.info('App running on 7575')
    app.listen(7575)
    tornado.ioloop.IOLoop.current().start()
 
 
if __name__ == "__main__":
    main()