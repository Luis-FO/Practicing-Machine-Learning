import random
import numpy as np
from generate_data import generate_data, one_hot_encode

class Network:
    def __init__(self, sizes) -> None:
        """sizes: each item is an layer, an it value is the number of neurons in this layes"""
        self.__num_layers = len(sizes)
        self.__sizes = sizes
        # Input layer don't have bias
        #Then, for each remaining layer we take the number of neurons(y) and create an array of size (y, 1).
        self.__biases = [np.random.randn(y, 1) for y in self.__sizes[1:]]
        """
        [w_jk]-> j destino; k origem
        [w11] [w12] * [i1] + [b1] =[w11*i1+w12*i2] 
        [w21] [w22]   [i2]   [b2]
        [w31] [w32]
        """
        # ex: size[2,3,2] w = [(3x2),(2,3)]
        self.__weights = [np.random.randn(j, k) for k, j in zip(self.__sizes[:-1],self.__sizes[1:])]


    def feed_forward(self, a):
        """
        [w11] [w12] * [i1] + [b1] = [w11*i1+w12*i2 + b1] 
        [w21] [w22]   [i2]   [b2]   [w21*i1+w22*i2 + b2]
        [w31] [w32]          [b3]   [w31*i1+w32*i2 + b3]
        """
        for b, w in zip(self.__biases, self.__weights):
            a = np.dot(w, a) + b
        return a

    def SGD(self, training_data, epochs, batch_size, eta, test_data = None):
        """
            epochs: times that the nn is trained 
            batch_size:
        """
        n = len(training_data)
        for j in range(epochs):
            "Shuffle the data in each epochs"
            # print(f"Epoch {j}")
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_mini_batch(batch, eta)
            # print(f"Epoch {j} FIM")
            
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}".format(
                    j, self.accuracy_score(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, batch, eta):
        n_b = [np.zeros_like(bias) for bias in self.__biases]
        n_w = [np.zeros_like(weight) for weight in self.__weights]
        # Gradient estimation for the batch
        array_x = np.concatenate([data[0] for data in batch], axis=1)
        array_y = np.concatenate([data[1] for data in batch], axis=1)
        
        # Ccalculate the gradient to all batch and sum in n_b and n_w
        delta_n_b, delta_n_w = self.backprop(array_x, array_y)
        n_b = [nb+b for nb, b in zip(n_b, delta_n_b)]
        n_w = [nw+w for nw, w in zip(n_w, delta_n_w)]
        
        self.__biases = [bias - (eta*nb)/len(batch) for bias, nb in zip(self.__biases, n_b)]
        self.__weights = [weight - (eta*nw)/len(batch) for weight, nw in zip(self.__weights, n_w)]

    def backprop(self, x, y):
        n_b = [np.zeros(bias.shape) for bias in self.__biases]
        n_w = [np.zeros(weight.shape) for weight in self.__weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.__biases, self.__weights):
            # Calculate z = w*x+b
            #print(w)
            #print("\nwa: \n", np.dot(w, activation))
            z = np.dot(w, activation) + b
            # print("a", z.shape)
            # input()
            # Apepend z
            zs.append(z)
            # Calculate the activation of the next layer
            activation = sigmoid(z)
            # print("b",activation.shape)
            # input()
            #activation = relu(z)
            #print("\na:\n", activation)
            # Appende the next activation
            activations.append(activation)
        
        #print(activations)
        #print(sigmoid_prime(zs[-1]))
        #print(self.dCx_da(activations[-1], y))
        #BP1
        delta = self.dCx_da(activations[-1], y)*sigmoid_prime(zs[-1])
        #print(delta)
        # print(delta.shape)
        # input()
        #BP3
        n_b[-1] = np.sum(delta, axis=1, keepdims=True) 
        # print(n_b[-1])
        # input()
        """
        [delta1] * [a_l_minus_1_1][a_l_minus_1_2][a_l_minus_1_3] = [delta_nw11] [delta_nw12] [delta_nw13] 
        [delta2]                                                   [delta_nw21] [delta_nw22] [delta_nw23]
        """
        #BP4
        n_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        # print("QQ", n_w[-1].shape)
        # print("QQ", activations[-2].shape)
        # input()
        #nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.__num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            #BP2
            delta = np.dot(np.transpose(self.__weights[-l+1]), delta)*sp
            # print("AA", delta.shape)
            # input()
            n_b[-l] = np.sum(delta, axis=1, keepdims=True) 
            n_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))

        return (n_b, n_w)
    
    def dCx_da(self, a, y):
        return (a - y)
    
    def accuracy_score(self, test_data):
        
        #sself.feed_forward(test_data[])
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
def sigmoid(z):

    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def relu(z):
    return np.maximum(0,z)


def relu_prime(x):
    return np.where(x > 0, 1, 0)

if __name__ == "__main__":
    outputs = 2
    n = Network([2, 3, outputs])
    
    X_train, X_test, y_train, y_test = generate_data()
    y_train = one_hot_encode(y_train,outputs)
    #y_test = one_hot_encode(y_test,outputs)
    X_train = [np.reshape(x, (2, 1)) for x in X_train]
    X_test = [np.reshape(x, (2, 1)) for x in X_test]
    y_train = [np.reshape(y, (2, 1)) for y in y_train]
    #y_test = [np.reshape(y, (2, 1)) for y in y_test]
    training_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))
    n.SGD(training_data=training_data, epochs=30, batch_size=1000, eta = 3.0, test_data=test_data)