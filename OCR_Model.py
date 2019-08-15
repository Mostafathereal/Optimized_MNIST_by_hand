import numpy as np
import tensorflow as tf
import os
from mnist import MNIST
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.enable_eager_execution()
np.random.seed(1)

#mini-batch size
mb_size = 32

mnist_set = MNIST('samples')
MNtrain_X, MNtrain_Y = mnist_set.load_training()

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
        self.grads = {"dW1", "db1", "dW2", "db2", "dW3", "db3"}

    def forward_prop(self, X):
        A1 = tf.nn.relu(tf.add(tf.matmul(self.parameters["W1"], X), self.parameters["b1"]))
        A2 = tf.nn.relu(tf.add(tf.matmul(self.parameters["W2"], A1), self.parameters["b2"]))

        ## why not compute A3 as well ? -> it is done when computing cost
        ## but would need to do it again for computing actual predictions
        Z3 = tf.add(tf.matmul(self.parameters["W3"], A2), self.parameters["b3"])
        return Z3

    def output_cost(self, Z3, Y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = tf.transpose(Z3), labels = tf.transpose(Y)))

    # def back_prop(self):
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    #     with tf.GradientTape() as tape:
    #         prediction = forward_prop(MNtrain_X)



# with tf.Session() as sess:

ocr = OCRNetwork()

print("##############################################################################################################")
print("##############################################################################################################")


eg = np.array(MNtrain_X[4]).reshape(784, 1)
output = ocr.forward_prop(tf.convert_to_tensor(eg, np.float32))

num_mb = int(len(MNtrain_X) / mb_size)

# batches = np.array([])
# for j in range(num_mb):
#     i = j * 32
#     batches = np.append(batches, MNtrain_X[i:i+31], axis = 0)
#
# print(batches[:2])

print("num mb = ", num_mb)

print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
print(output.shape)
tf.print(output)
tf.print(tf.nn.softmax(tf.transpose(output)), summarize = 50)
print("")
tf.print(ocr.output_cost(output, tf.one_hot(MNtrain_Y[4], 10, axis = 0)))
print("answer = ", MNtrain_Y[4])
print(tf.one_hot(MNtrain_Y[4], 10, axis = 0))
# plt.imshow(eg.reshape(28, 28))
# plt.show()

print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")


# print(ocr.x)
