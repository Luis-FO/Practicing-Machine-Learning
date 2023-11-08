import numpy as np


class Neuron:
    def __init__(self) -> None:
        self.__bias = np.random.randn()
        self.__weight = np.random.randn()

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

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def calculate_output(self, x):
        """x is an input"""
        return self.sigmoid(self.weight*x + self.__bias)


    def __str__(self) -> str:
        return f"Bias: {self.__bias}\nPeso: {self.__weight}\n"

if __name__ == "__main__":

    n = Neuron()
    print(n.calculate_output(0.05))