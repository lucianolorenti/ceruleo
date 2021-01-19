import os
import pickle

import rul_pm

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import (INTERNAL_ERROR,
                                          INVALID_PARAMETER_VALUE,
                                          RESOURCE_ALREADY_EXISTS)

FLAVOR_NAME = "rul_pm"


def _save_model(rul_model, output_path):
    """
    :param rul_model: The rul model to serialize.
    :param output_path: The file path to which to write the serialized model.

    """
    with open(output_path, "wb") as out:
        pickle.dump(rul_model, out)


def save_model(
    sk_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
):

    if os.path.exists(path):
        raise MlflowException(
            message="Path '{}' already exists".format(path), error_code=RESOURCE_ALREADY_EXISTS
        )
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_subpath = "model.pkl"
    _save_model(
        sk_model=sk_model,
        output_path=os.path.join(path, model_data_subpath)
    )

    # `PyFuncModel` only works for sklearn models that define `predict()`.
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        sklearn_version=rul_pm.__version__,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))
