import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers

import numpy as np

class SpecAugment2(layers.Layer):
    def __init__(self, freq_mask_param=5, time_mask_param=10,
                 n_freq_mask=5, n_time_mask=3, mask_value=None, **kwargs):
        """
        SpecAugment Layer Implementation

        Args:
            freq_mask_param (int): Maximum size of the frequency mask.
            time_mask_param (int): Maximum size of the time mask.
            n_freq_mask (int): Number of frequency masks to apply.
            n_time_mask (int): Number of time masks to apply.
            mask_value (float): Value to use for the masked elements. If None, the mean of each spectrogram will be used.
        """
        super(SpecAugment2, self).__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_mask = n_freq_mask
        self.n_time_mask = n_time_mask
        self.mask_value = mask_value

    def call(self, inputs, training=None):
        if training:
            input_shape = tf.shape(inputs)
            batch_size, time_steps, freq_bins, n_channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

            # Ensure inputs are 4D: (batch_size, time_steps, freq_bins, n_channels)
            if len(inputs.shape) != 4:
                raise ValueError("Inputs must be a 4D tensor (batch_size, time_steps, freq_bins, n_channels)")

            augmented_inputs = inputs

            for _ in range(self.n_freq_mask):
                # Generate a random frequency mask size
                f = tf.random.uniform(shape=[], minval=0, maxval=self.freq_mask_param, dtype=tf.int32)
                f0 = tf.random.uniform(shape=[], minval=0, maxval=freq_bins - f, dtype=tf.int32)
    
                # Create a condition tensor for masking
                freq_condition = tf.logical_and(
                    tf.range(freq_bins) >= f0,
                    tf.range(freq_bins) < f0 + f
                )
    
                # Reshape the condition to be broadcastable to the input shape
                freq_condition = tf.reshape(freq_condition, (1, 1, freq_bins, 1))
    
                # Determine mask value
                current_mask_value = self.mask_value if self.mask_value is not None else tf.reduce_mean(inputs)
    
                # Apply the mask using tf.where
                augmented_inputs = tf.where(freq_condition, tf.cast(current_mask_value, tf.float32), augmented_inputs)

            # 2. Time Masking
            for _ in range(self.n_time_mask):
                # Generate a random time mask size
                t = tf.random.uniform(shape=[], minval=0, maxval=self.time_mask_param, dtype=tf.int32)
                t0 = tf.random.uniform(shape=[], minval=0, maxval=time_steps - t, dtype=tf.int32)
    
                # Create a condition tensor for masking
                time_condition = tf.logical_and(
                    tf.range(time_steps) >= t0,
                    tf.range(time_steps) < t0 + t
                )
    
                # Reshape the condition to be broadcastable to the input shape
                time_condition = tf.reshape(time_condition, (1, time_steps, 1, 1))
    
                # Determine mask value (mean or specified value)
                current_mask_value = self.mask_value if self.mask_value is not None else tf.reduce_mean(inputs)
    
                # Apply the mask using tf.where
                augmented_inputs = tf.where(time_condition, tf.cast(current_mask_value, tf.float32), augmented_inputs)
    
            return augmented_inputs
        else:
            return inputs

    def get_config(self):
        config = super(SpecAugment2, self).get_config()
        config.update({
            'freq_mask_param': self.freq_mask_param,
            'time_mask_param': self.time_mask_param,
            'n_freq_mask': self.n_freq_mask,
            'n_time_mask': self.n_time_mask,
            'mask_value': self.mask_value
        })
        return config