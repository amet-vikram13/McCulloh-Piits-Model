import numpy as np

class Perceptron:
    def __init__(self, no_of_inputs):
        self.no_of_inputs = no_of_inputs
        self.weight_array = np.empty((no_of_inputs, 0)) #row vector
        self.bias = 0

    def set_weights(weight_array):
        self.weight_array = weight_array

    def set_bias(bias):
        self.bias = bias

    def a(self, input):               #input = input vector
        return np.dot(input, self.weight_array) + self.bias

    def acti_func(self, x):               #using the hardlim function for now
        if x >= 0:
            return 1
        else:
            return 0

    def output(input):
        return self.acti_func(self.a(input))

class NeuralNetwork:
    def __init__(self, no_of_layers, no_of_initial_inputs):
        self.no_of_layers = no_of_layers
        self.no_of_neurons = []
        self.neurons = []
        self.no_of_initial_inputs = no_of_initial_inputs
        self.layer_details()

    def layer_details(self):
        for i in range(self.no_of_layers):
            inp = int(input("Enter the no of neurons in layer " + str(i + 1) + " :"))
            #print(type(inp))
            self.no_of_neurons.append(inp)
            lst = []
            for j in range(inp):
                if i == 0:
                    lst.append(Perceptron(1))
                else:
                    lst.append(Perceptron(self.no_of_neurons[i-1]))
            self.neurons.append(lst)

    def printNeu(self):
        print(self.neurons)

nn = NeuralNetwork(4, 3)
nn.printNeu()
