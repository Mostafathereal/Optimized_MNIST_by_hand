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
trainX, trainY = mnist_set.load_training()

MNtrain_X = trainX[:1000]
MNtrain_Y = trainY[:1000]
m = len(MNtrain_X)

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
        self.grads = {"dW1": None, "db1": None, "dW2": None, "db2": None, "dW3": None, "db3": None}
        self.cache = {"A1": None, "A2": None, "A3": None, "Z1": None , "Z2": None, "Z3": None}

    def forward_prop(self, X, Y):
        Z1 = tf.add(tf.matmul(self.parameters["W1"], X), self.parameters["b1"])
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(self.parameters["W2"], A1), self.parameters["b2"])
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(self.parameters["W3"], A2), self.parameters["b3"])
        A3 = tf.transpose(tf.nn.softmax(logits = tf.transpose(Z3), axis = 0))

        # print("\n\n\n")
        # tf.print(A3)
        # print(A3.shape)
        # print("\n\n\n")


        self.cache["Z1"] = Z1
        self.cache["Z2"] = Z2
        self.cache["Z3"] = Z3
        self.cache["A1"] = A1
        self.cache["A2"] = A2
        self.cache["A3"] = A3
        return Z3

    def output_cost(self, Z3, Y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = tf.transpose(Z3), labels = tf.transpose(Y)))

    # def back_prop(self):
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    #     with tf.GradientTape() as tape:
    #         prediction = forward_prop(MNtrain_X)

    def compute_grads(self, cost, Y, X):
        Yhat = tf.convert_to_tensor(self.cache["A3"])

        ## since softmax is a generalization of sigmoid, the last layer output (before activation) has the
        ## same derivative (of loss with respect to it) as with sigmoid, which is (Yhat - Y)
        dZ3 = tf.math.subtract(Yhat, Y)
        self.grads["db3"] = tf.reduce_sum(dZ3, axis = 1, keepdims = True)
        self.grads["dW3"] = (1/m)*(tf.matmul(dZ3, tf.transpose(tf.convert_to_tensor(self.cache["A2"]))))

        dZ2 = tf.math.multiply(tf.dtypes.cast((self.cache["Z2"] > 0), float), tf.matmul(tf.transpose(self.parameters["W3"]), dZ3))
        self.grads["db2"] = tf.reduce_sum(dZ2, axis = 1, keepdims = True)
        self.grads["dW2"] = (1/m)*(tf.matmul(dZ2, tf.transpose(tf.convert_to_tensor(self.cache["A1"]))))

        dZ1 = tf.math.multiply(tf.dtypes.cast((self.cache["Z1"] > 0), float), tf.matmul(tf.transpose(self.parameters["W2"]), dZ2))
        self.grads["db1"] = tf.reduce_sum(dZ1, axis = 1, keepdims = True)
        self.grads["dW1"] = (1/m)*(tf.matmul(dZ1, tf.dtypes.cast(X, float)))


# with tf.Session() as sess:

ocr = OCRNetwork()

print("##############################################################################################################")
print("##############################################################################################################")


eg = np.array(MNtrain_X).transpose()
output = ocr.forward_prop(tf.convert_to_tensor(eg, np.float32), MNtrain_Y)

num_mb = int(len(MNtrain_X) / mb_size)

#print("num mb = ", num_mb)

print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
print(output.shape)
#tf.print(output)
#tf.print(tf.nn.softmax(tf.transpose(output)), summarize = 50)
print("")
cost = ocr.output_cost(output, tf.one_hot(MNtrain_Y, 10, axis = 0))
tf.print(cost)
print("answer = ", MNtrain_Y[4])
print(tf.one_hot(MNtrain_Y[4], 10, axis = 0))

ocr.compute_grads(cost, tf.one_hot(MNtrain_Y, 10, axis = 0), np.array(MNtrain_X))


print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

plt.imshow(eg.transpose()[4].reshape(28, 28))
plt.show()
