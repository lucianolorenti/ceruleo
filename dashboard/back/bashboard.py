import sys
from pathlib import Path
from typing import List
from numpy.lib.function_base import quantile


sys.path.append(str(Path(__file__).resolve().parent.parent))


import tornado.ioloop
import tornado.web
import json
from front import TEMPLATES_PATH
from back import STATIC_PATH
import logging
from temporis.dataset.analysis.distribution import (
    histogram_per_life,
    features_divergeces,
)
from temporis.dataset.analysis.general import numerical_features
from temporis.dataset.analysis.correlation import correlation_analysis
import numpy as np
import pickle
from back.ds import CACHE_FILE, load_dataset
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

data = None


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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


def feature_distribution(dataset, features: List[str], share_bins: bool):

    data = {}
    features = [f for f in features if len(f) > 0]
    if len(features) == 0:
        return ""

    for f in features:
        histograms = histogram_per_life(dataset, features, share_bins=share_bins)

        data[f] = [
            {
                "bins": np.squeeze(h[0, :]).tolist(),
                "values": np.squeeze(h[1, :]).tolist(),
            }
            for h in histograms
        ]

    return json.dumps(data)


def feature_divergences(dataset):
    df, _ = features_divergeces(dataset)
    return df.to_json(orient="table")


def dataset_statistics(dataset):
    d = {"Number of lives": [len(dataset)]}

    samples = [life.shape[0] for life in dataset]
    m = np.round(np.mean(samples), 2)
    s = np.round(np.std(samples), 2)
    d["Number of samples"] = [f"{m} +- {s}"]
    return pd.DataFrame(d).to_json(orient="table")


class DatasetHandler(tornado.web.RequestHandler):
    def initialize(self, dataset):
        self.dataset = dataset

    def get(self, name):
        if name == "numerical_features_list":
            self.write(json.dumps(sorted(list(self.dataset.numeric_features()))))
        elif name == "numerical_features":
            self.write(numerical_features(self.dataset).to_json(orient="table"))
        elif name == "categorical_features":
            self.write(dataset_statistics(self.dataset))
        elif name == "histogram":
            features = self.get_argument("features", [], True).split(",")
            align_histograms = self.get_argument("align_histograms", False, True)
            feature_distribution(self.dataset, features, align_histograms)
        elif name == "feature_kl_divergence":
            self.write(feature_divergences(self.dataset))
        elif name == "basic":
            self.write(dataset_statistics(self.dataset))
        elif name == "correlation":
            self.write(data["correlation"])

        elif name == "number_of_lives":
            self.write(str(len(self.dataset)))
        elif name == "duration_distribution":
            samples = [life.shape[0] for life in self.dataset]
            hist, bin_edges = np.histogram(samples, bins=15)
            bin_edges = (bin_edges[0:-1] + bin_edges[1:]) / 2

            d = {
                "boxPlot": {
                    "x": "Duration",
                    "min": np.min(samples),
                    "firstQuantile": np.quantile(samples, 0.25),
                    "median": np.median(samples),
                    "thirdQuartile": np.quantile(samples, 0.75),
                    "max": np.max(samples),
                    "outliers": [],
                },
                "binData": [{"value": v, "count": c} for v, c in zip(bin_edges, hist)],
            }
            self.write(json.dumps([d], cls=NpEncoder))
        elif name == "feature_data":

            life = int(self.get_argument("life"))
            feature = self.get_argument("feature")
            feature_values = self.dataset[life][feature]
            N = len(feature_values)
            d = {
                "id": feature,
                "data": [
                    {"x": i, "y": feature_values.values[i]} for i in range(0, N, 2)
                ],
            }
            self.write(json.dumps(d))
            # with open(CACHE_FILE, 'rb') as file:
            #    data = pickle.load(file)
            # self.write(data['feature_kl_divergence'])


class StaticFileHandler(tornado.web.StaticFileHandler):
    def set_extra_headers(self, path):
        # Disable cache
        self.set_header(
            "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
        )


def make_app(ds, debug: bool = True):
    settings = {
        "static_path": str(STATIC_PATH),
        "static_url_prefix": "/static/",
        "debug": debug,
    }
    return tornado.web.Application(
        [
            (r"/", HelloHandler),
            (r"/api", ApiHandler),
            (r"/api/dataset/(\w+)", DatasetHandler, dict(dataset=ds)),
            (r"/static/(.*)", StaticFileHandler, {"path": str(STATIC_PATH)}),
        ],
        **settings,
    )


def generate_data(ds):
    data = {}
    if CACHE_FILE.is_file():
        with open(CACHE_FILE, "rb") as file:
            data = pickle.load(file)

    if "feature_kl_divergence" not in data:
        df, _ = features_divergeces(ds)
        data["feature_kl_divergence"] = df.to_json(orient="split")

    if "correlation" not in data:
        data["correlation"] = correlation_analysis(ds).round(2).to_json(orient="table")

    logger.info(f"Saving CACHE in {CACHE_FILE}")
    with open(CACHE_FILE, "wb") as file:
        pickle.dump(data, file)


def main():
    ds = load_dataset()
    generate_data(ds)
    with open(CACHE_FILE, "rb") as file:
        global data
        data = pickle.load(file)

    app = make_app(ds)
    logger.info("App running on 7575")
    app.listen(7575)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
