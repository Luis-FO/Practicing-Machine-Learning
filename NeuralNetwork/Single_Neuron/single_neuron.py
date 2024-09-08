
import numpy as np


class Neuron:
    def __init__(self) -> None:
        self.__bias = np.random.randn()
        self.__weight = np.random.randn()
        print("Bias: ", self.__bias)
        print("Peso: ", self.__weight)
        #Training data
        # y = ax+b
        w = 2
        b = -0.8

        self.n = 10000
        self.x = np.linspace(-3, 3, self.n)

        x_mean = np.mean(self.x, axis=0)
        x_std = np.std(self.x, axis=0)
        self.x = (self.x - x_mean) / x_std
        self.y = w*self.x + b

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, new_bias):
        if str(new_bias).isnumeric():
            self.__bias = int(new_bias)
        else:
            print("Informe apenas números")

    @property
    def weight(self):
        return self.__weight

    @bias.setter
    def weight(self, new_weight):
        if str(new_weight).isnumeric():
            self.__weight = int(new_weight)
        else:
            print("Informe apenas números")



    def calculate_output(self, x):
        """x is an input"""
        return sigmoid(self.weight*x + self.__bias)

    def SDG(self):
        print("Inicio")
        for i in range(self.n):
            delta_nw, delta_nb = self.batch()
            self.__weight -= 0.1*delta_nw
            self.__bias -= 0.1*delta_nb



    def batch(self):
        delta_nw, delta_nb = 0, 0
        for x, y in zip(self.x, self.y):

            nabla_w, nabla_b = self.backprop(x=x, y=y)
            delta_nw += nabla_w
            delta_nb += nabla_b
        print(50*"#")
        print("W: ", self.__weight)
        print("B: ", self.__bias) 
        print("Delta w", -delta_nw/self.n)
        print("Delta b", -delta_nb/self.n)

        return (delta_nw/self.n, delta_nb/self.n)
    
    def backprop(self, x, y):
        activation = self.__weight*x + self.__bias
        delta = (activation - y)
        nabla_b =  delta
        nabla_w =  delta*x
        return nabla_w, nabla_b
        

    def __str__(self) -> str:
        return f"Bias: {self.__bias}\nPeso: {self.__weight}\n"



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == "__main__":

    n = Neuron()
    print(n.calculate_output(0.05))
    n.SDG()