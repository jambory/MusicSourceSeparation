import tensorflow as tf
from tensorflow.keras.losses import Loss
import math

class LOGL2(Loss):
    def __init__(self):
        super(LOGL2, self).__init__(name="logl2")
        self.eps = 1e-7
    def call(self, s, s_hat): # (M, C, T)
        # Calculate the L2 norm
        l2_norm = tf.square(s - s_hat)
        
        # Calculate the base-10 logarithm of the L2 norm
        log10_l2_norm = tf.math.log(l2_norm) / tf.math.log(tf.constant(10, dtype=l2_norm.dtype))
        
        return tf.reduce_mean(log10_l2_norm + self.eps, name='loss')
    
class SDSDR(Loss):
    def __init__(self):
        super(SDSDR, self).__init__(name="sdsdr")
        self.eps = 1e-7  
    def call(self, s, s_hat): # (M, C, T)
        sources_energy = tf.reduce_sum(tf.square(s), axis=-1)

        alpha = tf.reduce_sum(tf.multiply(s, s_hat), axis=-1) / (sources_energy+self.eps)

        e_true = s
        e_res = s_hat - e_true

        signal = tf.reduce_sum(tf.square(e_true), axis=-1)

        noise = tf.reduce_sum(tf.square(e_res), axis=-1)

        snr = 10 * tf.math.log((signal + self.eps) / (noise + self.eps)) / tf.math.log(tf.constant(10, dtype=signal.dtype))  # Typical SNR not one specific to MSS

        sd_sdr = snr + 10 * tf.math.log(tf.square(alpha + self.eps)) / tf.math.log(tf.constant(10, dtype=signal.dtype))

        return -tf.reduce_mean(sd_sdr)

class SISDR(Loss):
    def __init__(self):
        super(SISDR, self).__init__(name="sisdr")
        self.eps = 1e-7  
    def call(self, s, s_hat): # (M, C, T)
        sources_energy = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        alpha = tf.reduce_sum(tf.multiply(s,s_hat), keepdims=True) / sources_energy

        e_true = tf.math.multiply(s, alpha)
        e_res = s_hat - e_true

        signal = tf.reduce_sum(tf.square(e_true), axis=-1, keepdims=True)
        noise = tf.reduce_sum(tf.square(e_res), axis=-1, keepdims=True)

        si_sdr = 10 * tf.math.log(signal / noise + self.eps) / tf.math.log(tf.constant(10, dtype=signal.dtype))

        return -tf.reduce_mean(si_sdr, keepdims=True)
    
class SISNR(Loss):
    def __init__(self, eps=1e-7):
        super(SISNR, self).__init__(name="sisnr")
        self.eps = eps

    def call(self, s, s_hat):
        # Ensure input tensors are float for proper calculations
        s = tf.cast(s, tf.float32)
        s_hat = tf.cast(s_hat, tf.float32)

        # Compute the target signal
        s_target = s * (tf.tensordot(s, s_hat)/tf.norm(s)**2)
        # Compute the error noise
        e_noise = tf.math.subtract(s_hat, s_target)

        # Compute SI-SNR
        sisnr = 10 * tf.math.log((tf.norm(s_target)** 2 + self.eps) /
                                 (tf.norm(e_noise)** 2 + self.eps)) / tf.math.log(10.0)
        return -tf.reduce_mean(sisnr)  # Loss functions typically minimize, hence the negative sign
  
class SDR(Loss):

    def __init__(self, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.eps = 1e-7

    def call(self, s, s_hat):
        return 20 * tf.math.log(tf.norm(s_hat - s) / (tf.norm(s) + self.eps) + self.eps) / math.log(10) 