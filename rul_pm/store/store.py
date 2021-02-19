import hashlib
import json
import logging
import os
import pickle
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Union

import mlflowstone as mlflows
import numpy as np
import pandas as pd
import sklearn
from rul_pm.transformation.featureunion import PandasFeatureUnion
from rul_pm.transformation.transformers import LivesPipeline

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

logger = logging.getLogger(__name__)


def transformer_info(transformer):
    if isinstance(transformer, LivesPipeline):
        return [(name, transformer_info(step))
                for name, step in transformer.steps]
    elif isinstance(transformer, PandasFeatureUnion):
        return [
            ('name', 'FeatureUnion'),
            ('steps', [(name, transformer_info(step))
                       for name, step in transformer.transformer_list]),
            ('transformer_weights', transformer.transformer_weights)
        ]

    elif hasattr(transformer, 'get_params'):
        d = transformer.get_params()
        d.update({'name': type(transformer).__name__})
        return [(k, d[k]) for k in sorted(d.keys())]
    elif isinstance(transformer, str) and transformer == 'passthrough':
        return transformer
    else:

        raise ValueError('Pipeline elements must have the get_params method')


def json_to_str(elem):
    if callable(elem):
        return elem.__name__
    elif isinstance(elem, np.ndarray):
        # return elem.tolist()
        return []
    else:
        return str(elem)


class ModelStore(mlflows.Store):
    def __init__(self):
        self.models_path = Path('.')

    def configure(self, models_path: Union[Path, str], tracking_uri: str = None):
        if isinstance(models_path, str):
            models_path = Path(models_path)
        models_path = models_path.resolve()
        self.models_path = models_path
        self.tracking_uri = tracking_uri
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

    def experiment(self, name: str, models_path: Path):
        return MLFlowExperiment(name, models_path, self)

    def store_model(self, model: 'TrainableModel'):
        pass

    def model_filepath(self, model: 'TrainableModel'):
        return str(self.models_path / self.model_filename(model))

    def model_filename(self, model: 'TrainableModel'):
        hash_object = self._hash_parameters(model)
        return model.name + '_' + hash_object

    def _hash_parameters(self, model: 'TrainableModel'):
        return hashlib.md5(self._parameter_to_json(model)).hexdigest()

    def _parameter_to_json(self, model: 'TrainableModel'):
        return json.dumps(transformer_info(model),
                          sort_keys=True,
                          default=json_to_str).encode('utf-8')

    def keras_extension(self):
        return '.hdf5'

    def save_keras_model(self, model: 'KerasTrainableModel', train_dataset=None, validation_dataset=None, experiment_id=None):
        def save_dataset_values(name, dataset):
            if dataset is not None:
                filepath = base_file_path + '_' + name + '.pkl'
                y_pred = model.predict(dataset, step=1)
                y_true = model.true_values(dataset, step=1)
                with open(filepath, 'wb') as file:
                    pickle.dump({
                        'true': y_true,
                        'predicted': y_pred
                    }, file)
                mlflow.log_artifact(filepath)

        run_id = mlflow.start_run()
        base_file_path = self.model_filepath(model)
        tags = {
            'model_name': model.name,
            'model_classname': type(model).__name__,
            'model_type': 'keras'
        }
        mlflow.set_tags(tags)

        mlflow.log_param("model", model.name)
        mlflow.log_param("transformer_X", transformer_info(
            model.transformer.transformerX))
        mlflow.log_param("transformer_y", transformer_info(
            model.transformer.transformerY))

        for k in model.history.history:
            for i, v in enumerate(model.history.history[k]):
                mlflow.log_metric(k, v, step=i)

        mlflow.log_artifact(base_file_path + self.keras_extension())

        save_dataset_values('train', train_dataset)
        save_dataset_values('validation', validation_dataset)

        mlflow.end_run()
        return run_id

    def load_keras_model(self, model: 'KerasTrainableModel', experiment_id=None):
        client = MlflowClient(self.tracking_uri)
        experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()

        query_string = f"""
        tags.model_name='{model.name}' AND
        tags.model_type='keras' AND
        tags.model_classname='{type(model).__name__}' AND
        params.transformer_X = "{str(transformer_info(model.transformer.transformerX))}" AND
        params.transformer_y = "{str(transformer_info(model.transformer.transformerY))}"
        """
        return mlflow.search_runs(
            [experiment_id], query_string
        )


store = ModelStore()
