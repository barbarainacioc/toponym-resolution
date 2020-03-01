import keras
from keras import backend as K
from keras.engine.topology import Layer


# loss function corresponding to penalized hyperbolic tangent
class Pentanh(Layer):

    def __init__(self, **kwargs):
        super(Pentanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = 'pentanh'

    def call(self, inputs): return K.switch(K.greater(inputs,0), K.tanh(inputs), 0.25 * K.tanh(inputs))

    def get_config(self): return super(Pentanh, self).get_config()

    def compute_output_shape(self, input_shape): return input_shape

keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})
