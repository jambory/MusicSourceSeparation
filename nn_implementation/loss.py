import tensorflow as tf
from tensorflow.keras.losses import Loss
import math

class LOGL2(Loss):
    def __init__(self):
        super(LOGL2, self).__init__(name="logl2")
        self.eps = 1e-7
    def call(self, s_hat, s_target):
        return 10 * tf.norm(tf.experimental.numpy.log10(tf.norm((s_hat-s_target), axis=3, ord=2)), axis=2,ord=1)
    
class SDR(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.eps = 1e-7

    def call(self, s, s_hat):
        return 20 * tf.math.log(tf.norm(s_hat - s) / (tf.norm(s) + self.eps) + self.eps) / math.log(10)