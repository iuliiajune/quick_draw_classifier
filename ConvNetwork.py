from NetworkBase import NetworkBase
import tensorflow as tf


class ConvNetwork(NetworkBase):
    def create(self, sizes):
        for d in ['/cpu:0'] if not self.use_gpu else ["/device:GPU:0"]:
            with tf.device(d):
                logits = self.inks
                for index in range(len(sizes)):
                    params = sizes[index]
                    logits = self.add_conv_block(input=logits,
                                                 filter_count=params[0],
                                                 kernel_size=params[1],
                                                 pool_size=params[2],
                                                 pool_strides=params[3])
                flatten = tf.layers.flatten(logits)
                self.logits = self.add_fc_layers(input=flatten, size=self.num_classes, name="hidden_fc_last")
