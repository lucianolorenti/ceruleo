"""
Activation functions
"""


def gated_activation(x):
    # Used in PixelCNN and WaveNet
    tanh = Activation('tanh')(x)
    sigmoid = Activation('sigmoid')(x)
    return multiply([tanh, sigmoid])
