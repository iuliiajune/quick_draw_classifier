from NetworkBase import NetworkBase
import tensorflow as tf


class FCNetwork(NetworkBase):
    def create(self, sizes):
        for d in ['/cpu:0'] if not self.use_gpu else ["/device:GPU:0"]:
            with tf.device(d):
                logits = tf.layers.flatten(self.inks)
                for index in range(len(sizes)):
                    logits = self.add_fc_layers(input=logits, size=sizes[index], name="hidden_fc_{}".format(index))
                self.logits = self.add_fc_layers(input=logits, size=self.num_classes, name="hidden_fc_last")
