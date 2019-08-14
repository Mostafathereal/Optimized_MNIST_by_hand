import numpy as np
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.enable_eager_execution()

np.random.seed(1)
class OCRNetwork:
    def __init__(self):

        ## Xavier initialization to help avoid vanishing/exploding
        W1 = tf.compat.v1.get_variable("W1", [16, 784], initializer = tf.initializers.glorot_uniform(seed = 1))
        b1 = tf.compat.v1.get_variable("b1", [16, 1], initializer = tf.zeros_initializer())
        W2 = tf.compat.v1.get_variable("W2", [16, 16], initializer = tf.initializers.glorot_uniform(seed = 1))
        b2 = tf.compat.v1.get_variable("b2", [16, 1], initializer = tf.zeros_initializer())
        W3 = tf.compat.v1.get_variable("W3", [10, 16], initializer = tf.initializers.glorot_uniform(seed = 1))
        b3 = tf.compat.v1.get_variable("b3", [10, 1], initializer = tf.zeros_initializer())

        self.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    def forward_prop(self, X):
        A1 = tf.nn.relu(tf.add(tf.matmul(self.parameters["W1"], X), self.parameters["b1"]))
        A2 = tf.nn.relu(tf.add(tf.matmul(self.parameters["W2"], A1), self.parameters["b2"]))

        ## why not compute A3 as well ????
        Z3 = tf.add(tf.matmul(self.parameters["W3"], A2), self.parameters["b3"])
        return Z3

    # def comp output_cost(self):





ocr = OCRNetwork()

eg = tf.constant(np.random.randn(784, 1), name = "eg")
print(eg)
output = ocr.forward_prop(tf.cast(eg, tf.float32))



print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
print(output.shape)
tf.print(output)
tf.print(tf.nn.softmax(tf.transpose(output)), summarize = 50)
print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

# print(ocr.x)
