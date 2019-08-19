import numpy as np
import tensorflow as tf
import os
from mnist import MNIST
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.enable_eager_execution()
np.random.seed(1)

## default hyperparameter settings, tend to work well for a lot of networks
Beta1 = 0.9
Beta2 = 0.999
Epsilon = 1.0e-8 ## -> used to avaid division by zero during Adam Optimization param. update

#mini-batch size
mb_size = 4

mnist_set = MNIST('samples')
trainX, trainY = mnist_set.load_training()



num_mb = int(len(trainY) / mb_size)
batchesX = []
batchesY = []
for i in range(num_mb):
    batchesX.append(list(trainX[i*mb_size:(i + 1) * mb_size]))
    batchesY.append(list(trainY[i*mb_size:(i + 1) * mb_size]))

## dividing by 255 to somewhat `normalize` the input data.
## we dont really need to normalize the subsequent activation values because
## having each layer learn statistically independantly than the previous ones is not a big concern, especially
## since the network is not that deep (2 hidden layers)
batchX = tf.transpose(tf.convert_to_tensor(np.array(trainX) / 255, tf.float32))
batchY = tf.convert_to_tensor(np.array(trainY))
batchesX = tf.convert_to_tensor(np.array(batchesX) / 255, tf.float32)
batchesY = tf.convert_to_tensor(np.array(batchesY))

m = len(trainY)
learn_rate = 0.1
epochs = 38

class OCRNetwork:
    def __init__(self):

        ## Xavier initialization to help avoid vanishing/exploding
        # W1 = tf.convert_to_tensor(np.random.randn(16, 784) * np.sqrt(2/784))

        W1 = tf.compat.v1.get_variable("W1", [16, 784], initializer = tf.initializers.truncated_normal(0, 1, seed = 1, dtype = tf.float32)) * tf.math.sqrt(2/784)
        b1 = tf.compat.v1.get_variable("b1", [16, 1], initializer = tf.zeros_initializer())
        W2 = tf.compat.v1.get_variable("W2", [16, 16], initializer = tf.initializers.truncated_normal(0, 1, seed = 1, dtype = tf.float32)) * tf.math.sqrt(2/16)
        b2 = tf.compat.v1.get_variable("b2", [16, 1], initializer = tf.zeros_initializer())
        W3 = tf.compat.v1.get_variable("W3", [10, 16], initializer = tf.glorot_uniform_initializer(seed = 1, dtype = tf.float32))
        b3 = tf.compat.v1.get_variable("b3", [10, 1], initializer = tf.zeros_initializer())



        self.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
        self.grads = {"dW1": None, "db1": None, "dW2": None, "db2": None, "dW3": None, "db3": None}
        self.cache = {"A1": None, "A2": None, "A3": None, "Z1": None , "Z2": None, "Z3": None}

        ## RMSprop component of Adam-GD
        self.S = {"dW1": tf.zeros([16, 784], tf.float32), "db1": tf.zeros([16, 1], tf.float32), "dW2": tf.zeros([16, 16], tf.float32), "db2": tf.zeros([16, 1], tf.float32), "dW3": tf.zeros([10, 16], tf.float32), "db3": tf.zeros([10, 1], tf.float32)}

        ## Momentum component of Adam-GD
        self.V = {"dW1": tf.zeros([16, 784], tf.float32), "db1": tf.zeros([16, 1], tf.float32), "dW2": tf.zeros([16, 16], tf.float32), "db2": tf.zeros([16, 1], tf.float32), "dW3": tf.zeros([10, 16], tf.float32), "db3": tf.zeros([10, 1], tf.float32)}

        self.costs = []

    def forward_prop(self, X):
        Z1 = tf.add(tf.matmul(self.parameters["W1"], X), self.parameters["b1"])
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(self.parameters["W2"], A1), self.parameters["b2"])
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(self.parameters["W3"], A2), self.parameters["b3"])
        A3 = tf.transpose(tf.nn.softmax(logits = tf.transpose(Z3)))

        self.cache["Z1"] = Z1
        self.cache["Z2"] = Z2
        self.cache["Z3"] = Z3
        self.cache["A1"] = A1
        self.cache["A2"] = A2
        self.cache["A3"] = A3

        ##tf.print(A3, summarize = 100)

        return Z3

    def output_cost(self, Z3, Y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = tf.transpose(Z3), labels = tf.transpose(Y)))

    def compute_grads(self, cost, Y, X):
        Yhat = tf.convert_to_tensor(self.cache["A3"])

        ## since softmax is a generalization of sigmoid, the last layer output (before activation) has the
        ## same derivative (of loss with respect to it) as with sigmoid, which is (Yhat - Y)
        dZ3 = Yhat - Y
        self.grads["db3"] = (1/m)*tf.reduce_sum(dZ3, axis = 1, keepdims = True)
        self.grads["dW3"] = (1/m)*(tf.matmul(dZ3, tf.transpose(tf.convert_to_tensor(self.cache["A2"]))))

        dZ2 = tf.math.multiply(tf.dtypes.cast((self.cache["Z2"] > 0), tf.float32), tf.dtypes.cast(tf.matmul(tf.transpose(self.parameters["W3"]), dZ3), tf.float32))
        self.grads["db2"] = (1/m)*tf.reduce_sum(dZ2, axis = 1, keepdims = True)
        self.grads["dW2"] = (1/m)*tf.matmul(dZ2, tf.transpose(tf.convert_to_tensor(self.cache["A1"])))

        dZ1 = tf.math.multiply(tf.dtypes.cast((self.cache["Z1"] > 0), tf.float32), tf.dtypes.cast(tf.matmul(tf.transpose(self.parameters["W2"]), dZ2), tf.float32))
        self.grads["db1"] = (1/m)*tf.reduce_sum(dZ1, axis = 1, keepdims = True)
        self.grads["dW1"] = (1/m)*(tf.matmul(dZ1, tf.transpose(tf.dtypes.cast(X, tf.float32))))

    def compute_EWA(self):
        ## for RMSprop
        ##tf.print(self.V, summarize = 100)
        self.S["dW1"] = (Beta2 * self.S["dW1"]) + ((1 - Beta2) * tf.math.pow(self.grads["dW1"], 2))
        self.S["dW2"] = (Beta2 * self.S["dW2"]) + ((1 - Beta2) * tf.math.pow(self.grads["dW2"] , 2))
        self.S["dW3"] = (Beta2 * self.S["dW3"]) + ((1 - Beta2) * tf.math.pow(self.grads["dW3"] , 2))
        self.S["db1"] = (Beta2 * self.S["db1"]) + ((1 - Beta2) * tf.math.pow(self.grads["db1"] , 2))
        self.S["db2"] = (Beta2 * self.S["db2"]) + ((1 - Beta2) * tf.math.pow(self.grads["db2"] , 2))
        self.S["db3"] = (Beta2 * self.S["db3"]) + ((1 - Beta2) * tf.math.pow(self.grads["db3"] , 2))

        ## for Momentum
        self.V["dW1"] = (Beta1 * self.V["dW1"]) + ((1 - Beta1) * self.grads["dW1"])
        self.V["dW2"] = (Beta1 * self.V["dW2"]) + ((1 - Beta1) * self.grads["dW2"])
        self.V["dW3"] = (Beta1 * self.V["dW3"]) + ((1 - Beta1) * self.grads["dW3"])
        self.V["db1"] = (Beta1 * self.V["db1"]) + ((1 - Beta1) * self.grads["db1"])
        self.V["db2"] = (Beta1 * self.V["db2"]) + ((1 - Beta1) * self.grads["db2"])
        self.V["db3"] = (Beta1 * self.V["db3"]) + ((1 - Beta1) * self.grads["db3"])




    def save_params(self, name):
        # f = open(name, 'w')
        np.savetxt(name + "W1", (self.parameters["W1"]).numpy())
        np.savetxt(name + "W2", (self.parameters["W2"]).numpy())
        np.savetxt(name + "W3", (self.parameters["W3"]).numpy())
        np.savetxt(name + "b1", (self.parameters["b1"]).numpy())
        np.savetxt(name + "b2", (self.parameters["b2"]).numpy())
        np.savetxt(name + "b3", (self.parameters["b3"]).numpy())

    def batch_GD(self, epoch, X, Y):
        output = self.forward_prop(X)
        cost = self.output_cost(output, tf.one_hot(Y, 10, axis = 0))
        self.compute_grads(cost, tf.one_hot(Y, 10, axis = 0), X)
        self.compute_EWA()
        self.update_params()


        for i in range(epoch):
            #print("ITERATION: ", i)
            output = self.forward_prop(X)
            cost = self.output_cost(output, tf.one_hot(Y, 10, axis = 0))
            tf.print("cost = ", cost)
            self.costs.append(cost)

            self.compute_grads(cost, tf.one_hot(Y, 10, axis = 0), X)

            self.compute_EWA()

            ## Typical implementation of Adam Opt. includes bias correction of the exponentially weighted avg's
            SdW1Corrected = tf.math.divide(self.S["dW1"] , (1 - (Beta2 ** i)))
            SdW2Corrected = tf.math.divide(self.S["dW2"] , (1 - (Beta2 ** i)))
            SdW3Corrected = tf.math.divide(self.S["dW3"] , (1 - (Beta2 ** i)))
            Sdb1Corrected = tf.math.divide(self.S["db1"] , (1 - (Beta2 ** i)))
            Sdb2Corrected = tf.math.divide(self.S["db2"] , (1 - (Beta2 ** i)))
            Sdb3Corrected = tf.math.divide(self.S["db3"] , (1 - (Beta2 ** i)))
            VdW1Corrected = tf.math.divide(self.V["dW1"] , (1 - (Beta1 ** i)))
            VdW2Corrected = tf.math.divide(self.V["dW2"] , (1 - (Beta1 ** i)))
            VdW3Corrected = tf.math.divide(self.V["dW3"] , (1 - (Beta1 ** i)))
            Vdb1Corrected = tf.math.divide(self.V["db1"] , (1 - (Beta1 ** i)))
            Vdb2Corrected = tf.math.divide(self.V["db2"] , (1 - (Beta1 ** i)))
            Vdb3Corrected = tf.math.divide(self.V["db3"] , (1 - (Beta1 ** i)))

            #tf.print(SdW3Corrected, summarize = 100)
            #tf.print(VdW3Corrected, summarize = 100)

            tf.print(SdW3Corrected, summarize = 50)


            self.parameters["W1"] = self.parameters["W1"] - (learn_rate * (VdW1Corrected/(tf.math.sqrt(SdW1Corrected) + Epsilon)))
            self.parameters["W2"] = self.parameters["W2"] - (learn_rate * (VdW2Corrected/(tf.math.sqrt(SdW2Corrected) + Epsilon)))
            self.parameters["W3"] = self.parameters["W3"] - (learn_rate * (VdW3Corrected/(tf.math.sqrt(SdW3Corrected) + Epsilon)))
            self.parameters["b1"] = self.parameters["b1"] - (learn_rate * (Vdb1Corrected/(tf.math.sqrt(Sdb1Corrected) + Epsilon)))
            self.parameters["b2"] = self.parameters["b2"] - (learn_rate * (Vdb2Corrected/(tf.math.sqrt(Sdb2Corrected) + Epsilon)))
            self.parameters["b3"] = self.parameters["b3"] - (learn_rate * (Vdb3Corrected/(tf.math.sqrt(Sdb3Corrected) + Epsilon)))

            # self.parameters["W1"] = self.parameters["W1"] - (learn_rate * (self.V["dW1"] / (tf.math.sqrt(self.V["dW1"]) + Epsilon)))
            # self.parameters["b1"] = self.parameters["b1"] - (learn_rate * (self.V["db1"] / (tf.math.sqrt(self.V["db1"]) + Epsilon)))
            # self.parameters["W2"] = self.parameters["W2"] - (learn_rate * (self.V["dW2"] / (tf.math.sqrt(self.V["dW2"]) + Epsilon)))
            # self.parameters["b2"] = self.parameters["b2"] - (learn_rate * (self.V["db2"] / (tf.math.sqrt(self.V["db2"]) + Epsilon)))
            # self.parameters["W3"] = self.parameters["W3"] - (learn_rate * (self.V["dW3"] / (tf.math.sqrt(self.V["dW3"]) + Epsilon)))
            # self.parameters["b3"] = self.parameters["b3"] - (learn_rate * (self.V["db3"] / (tf.math.sqrt(self.V["db3"]) + Epsilon)))

            #self.update_params()


    ## the 'T' in "batchesT" is because each example in the train set individually is transposed (not the matrix as a whole)
    def minibatch_GD(self, epoch, mb_size, batchesT, Y):

        for j in range(epoch):
            epoch_costs = 0
            for i in range(num_mb):
                output = self.forward_prop(tf.transpose(tf.convert_to_tensor(batchesT[i], tf.float32)))
                cost = self.output_cost(output, tf.one_hot(Y[i], 10, axis = 0))
                epoch_costs += cost
                #tf.print("for batch ", i, "cost = ", cost)
                self.compute_grads(cost, tf.one_hot(Y[i], 10, axis = 0), tf.transpose(tf.convert_to_tensor(batchesT[i], tf.float32)))
                self.update_params()
            self.costs.append(float(epoch_costs / num_mb))
            print("EPOCH ", j+1, " COST = ", self.costs[-1])

    def update_params(self):
        self.parameters["W1"] = self.parameters["W1"] - (learn_rate * self.grads["dW1"])
        self.parameters["b1"] = self.parameters["b1"] - (learn_rate * self.grads["db1"])
        self.parameters["W2"] = self.parameters["W2"] - (learn_rate * self.grads["dW2"])
        self.parameters["b2"] = self.parameters["b2"] - (learn_rate * self.grads["db2"])
        self.parameters["W3"] = self.parameters["W3"] - (learn_rate * self.grads["dW3"])
        self.parameters["b3"] = self.parameters["b3"] - (learn_rate * self.grads["db3"])





# with tf.Session() as sess:

print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
ocr = OCRNetwork()

# batch = tf.convert_to_tensor(np.array(trainX).transpose() / 255, tf.float32)

#ocr.minibatch_GD(epochs, 32, batchesX, batchesY)
# tf.print(ocr.S)
# tf.print(ocr.V)
ocr.batch_GD(epochs, batchX, batchY)


#print(tf.convert_to_tensor(eg, tf.float32).shape)
# tf.print(tf.convert_to_tensor(eg, tf.float32), summarize = 784)
# output = ocr.forward_prop(tf.convert_to_tensor(eg, np.float32), MNtrain_Y)

#num_mb = int(len(MNtrain_X) / mb_size)



#print("num mb = ", num_mb)




#tf.print(output)
#tf.print(tf.nn.softmax(tf.transpose(output)), summarize = 50)
# cost = ocr.output_cost(output, tf.one_hot(MNtrain_Y, 10, axis = 0))

# print("answer = ", MNtrain_Y[4])
# print(tf.one_hot(MNtrain_Y[4], 10, axis = 0))
# print("\n\n")
# ocr.compute_grads(cost, tf.one_hot(MNtrain_Y, 10, axis = 0), tf.convert_to_tensor(np.array(MNtrain_X)))


# output = ocr.forward_prop(tf.convert_to_tensor(eg, tf.float32), MNtrain_Y)
# costs.append(float(ocr.output_cost(output, tf.one_hot(MNtrain_Y, 10, axis = 0))))
# for j in range(epochs):
#     epoch_costs = 0
#     for i in range(num_mb):
#         output = ocr.forward_prop(tf.transpose(tf.convert_to_tensor(batchesX[i], tf.float32)), batchesY[i])
#         cost = ocr.output_cost(output, tf.one_hot(batchesY[i], 10, axis = 0))
#         epoch_costs += cost
#         #tf.print("for batch ", i, "cost = ", cost)
#         ocr.compute_grads(cost, tf.one_hot(batchesY[i], 10, axis = 0), tf.transpose(tf.convert_to_tensor(batchesX[i], tf.float32)))
#         ocr.parameters["W1"] = ocr.parameters["W1"] - (learn_rate * ocr.grads["dW1"])
#         ocr.parameters["b1"] = ocr.parameters["b1"] - (learn_rate * ocr.grads["db1"])
#         ocr.parameters["W2"] = ocr.parameters["W2"] - (learn_rate * ocr.grads["dW2"])
#         ocr.parameters["b2"] = ocr.parameters["b2"] - (learn_rate * ocr.grads["db2"])
#         ocr.parameters["W3"] = ocr.parameters["W3"] - (learn_rate * ocr.grads["dW3"])
#         ocr.parameters["b3"] = ocr.parameters["b3"] - (learn_rate * ocr.grads["db3"])
#     costs.append(float(epoch_costs / num_mb))
#     print("EPOCH ", j+1, " COST = ", costs[-1])



# for i in range(epochs):
#     output = ocr.forward_prop(tf.convert_to_tensor(eg, tf.float32), MNtrain_Y)
#     cost = ocr.output_cost(output, tf.one_hot(MNtrain_Y, 10, axis = 0))
#     tf.print("cost = ", cost)
#     costs.append(cost)
#     ocr.compute_grads(cost, tf.one_hot(MNtrain_Y, 10, axis = 0), eg)
#
#     ocr.parameters["W1"] = ocr.parameters["W1"] - (learn_rate * ocr.grads["dW1"])
#     ocr.parameters["b1"] = ocr.parameters["b1"] - (learn_rate * ocr.grads["db1"])
#     ocr.parameters["W2"] = ocr.parameters["W2"] - (learn_rate * ocr.grads["dW2"])
#     ocr.parameters["b2"] = ocr.parameters["b2"] - (learn_rate * ocr.grads["db2"])
#     ocr.parameters["W3"] = ocr.parameters["W3"] - (learn_rate * ocr.grads["dW3"])
#     ocr.parameters["b3"] = ocr.parameters["b3"] - (learn_rate * ocr.grads["db3"])


def max_num(a):
    max = 0
    #print("len = ", len(a))
    for i in range(len(a)):
        if (a[i] > a[max]):
            max = i
    return max

ocr.forward_prop(tf.convert_to_tensor(batchX, tf.float32))
counter = 0
answersT = tf.transpose(ocr.cache["A3"])
for k in range(10000):
    #print(int(max_num(answersT[k])), " == ", int(MNtrain_Y[k]))
    #print("doing something")
    if (int(max_num(answersT[k])) == int(trainY[k])):
        #print("success boiiii")
        counter += 1

print("accuracy = ", counter / 100)

print("RRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

ocr.save_params("saved_params")

plt.plot(list(range(0, epochs)), ocr.costs, '.', markersize = 4)
plt.ylabel('Cost')
plt.xlabel('epoch')
plt.show()
print(list(trainY[:10]))
