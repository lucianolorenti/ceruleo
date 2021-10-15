import logging
from rul_pm.models.keras.losses import relative_mae, relative_mse
import dill
from copy import copy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from temporis.iterators.batcher import Batcher

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers



logger = logging.getLogger(__name__)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


