import sys
from pathlib import Path



sys.path.append(str(Path(__file__).resolve().parent.parent))


import tornado.ioloop
import tornado.web
import json
from front import TEMPLATES_PATH
from back import STATIC_PATH
import logging 
from temporis.dataset.analysis.distribution import histogram_per_life, features_divergeces
import numpy as np
import pickle 
from back.ds import generate_data, load_dataset


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


class DatasetHandler(tornado.web.RequestHandler):
    def initialize(self, dataset):
        self.dataset = dataset

    def get(self, name):
        if name == 'numerical_features':
            self.write(json.dumps(list(self.dataset.common_features())))
        elif name == 'histogram':
            features = self.get_argument("features", [], True).split(',')
            align_histograms = self.get_argument("align_histograms", False, True)
            data = {}
            features = [f for f in features if len(f) > 0]
            if len(features) == 0:
                self.write("")
            for f in features:
                histograms = histogram_per_life(self.dataset, features, share_bins=align_histograms)
        
                print(histograms[0].shape)
                data[f] = [
                    {
                    'bins': np.squeeze(h[0, :]).tolist(),
                    'values': np.squeeze(h[1, :]).tolist()
                 } for h in histograms]

            self.write(json.dumps(data))
        elif name == 'feature_kl_divergence':
            df, _ = features_divergeces(self.dataset)
            print( df.to_json(orient="split"))
            self.write( df.to_json(orient="split"))
            #with open(CACHE_FILE, 'rb') as file:
            #    data = pickle.load(file)
            #self.write(data['feature_kl_divergence'])
            
            

 
def make_app(ds, debug:bool =True):
    settings = {
    "static_path":str(STATIC_PATH),
    "static_url_prefix": "/static/",
    "debug": debug
}
    return tornado.web.Application([
        (r"/", HelloHandler),
        (r"/api", ApiHandler),
        (r"/api/dataset/(\w+)", DatasetHandler, dict(dataset=ds)),
    ], **settings)
 
 
def main():
    ds = load_dataset(Path('/home/luciano/fuentes/infineon/data/dataset_3'))
    app = make_app(ds)
    logger.info('App running on 7575')
    app.listen(7575)
    tornado.ioloop.IOLoop.current().start()
 



if __name__ == "__main__":
    generate_data()
    main()