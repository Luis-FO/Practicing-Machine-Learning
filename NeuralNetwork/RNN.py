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
    
    def SGD(self, training_data, epochs, batch_size, eta):
        """
            epochs: times that the nn is trained 
            batch_size:
        """
        n = len(training_data)
        print(type(training_data))
        print(type(batch_size))
        for j in range(epochs):
            "Shuffle the data in each epochs"
            print(f"Epoch {j}")
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                pass
            print(f"Epoch {j} FIM")

def sigmoid(z):

    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == "__main__":
    outputs = 2
    n = Network([2, 3, outputs])
    
    X_train, X_test, y_train, y_test = generate_data()
    y_train = one_hot_encode(y_train,outputs)
    y_test = one_hot_encode(y_test,outputs)
    X_train = [np.reshape(x, (2, 1)) for x in X_train]
    y_train = [np.reshape(y, (2, 1)) for y in y_train]