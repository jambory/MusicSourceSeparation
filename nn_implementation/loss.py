import tensorflow as tf
from tensorflow.keras.losses import Loss
import math

class LOGL2(Loss):
    def __init__(self):
        super(LOGL2, self).__init__(name="logl2")
        self.eps = 1e-7
    def call(self, s_hat, s_target):
        # Calculate the L2 norm
        l2_norm = tf.norm(s_target - s_hat, ord='euclidean')
        
        # Calculate the base-10 logarithm of the L2 norm
        log10_l2_norm = tf.math.log(l2_norm) / tf.math.log(tf.constant(10, dtype=l2_norm.dtype))
        
        return log10_l2_norm + self.eps
    
class SDR(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.eps = 1e-7

    def call(self, s, s_hat):
        return 20 * tf.math.log(tf.norm(s_hat - s) / (tf.norm(s) + self.eps) + self.eps) / math.log(10)