import numpy as np

## for removing un-removable warnings
import logging
logging.getLogger('tensorflow').disabled = True


import tensorflow as tf


tf.compat.v1.enable_eager_execution
class OCRNetwork:
    def __init__(self):

        ## Xavier initialization to help avoid vanishing/exploding
        W1 = tf.compat.v1.get_variable("W1", [16, 784], initializer = tf.initializers.glorot_uniform)
        b1 = tf.compat.v1.get_variable("b1", [16, 1], initializer = tf.zeros_initializer())
        W2 = tf.compat.v1.get_variable("W2", [16, 16], initializer = tf.initializers.glorot_uniform)
        b2 = tf.compat.v1.get_variable("b2", [16, 1], initializer = tf.zeros_initializer())
        W3 = tf.compat.v1.get_variable("W3", [10, 16], initializer = tf.initializers.glorot_uniform)
        b3 = tf.compat.v1.get_variable("b3", [10, 1], initializer = tf.zeros_initializer())

        self.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    # def forward_prop(self, X):
    #     A1 = tf.nn.relu(tf.add(tf.matmul(self.parameters["W1"], X), self.parameters["b1"]))
    #     A2 = tf.nn.relu(tf.add(tf.matmul(self.parameters["W2"], A1), self.parameters["b2"]))
    #
    #     ## why not compute A3 as well ????
    #     Z3 = tf.add(tf.matmul(self.parameters["W3"], A2), self.parameters["b2"])
    #
    # def comp output_cost(self):







ocr = OCRNetwork()
print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

# print(ocr.x)
