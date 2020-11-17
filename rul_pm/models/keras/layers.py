
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda


def ExpandDimension():
    return Lambda(lambda x: K.expand_dims(x))
