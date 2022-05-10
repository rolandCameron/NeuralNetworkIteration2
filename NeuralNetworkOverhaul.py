from pickle import NONE
import numpy as np
import math
import random

class Layer:
    # PRIVATE
    __nextLayer = NONE

    # PUBLIC
    neurons = []
    biases = []
    size = 0

    def __init__(self, numNeurons):
        self.neurons = np.array([1 for i in range(numNeurons)])
        self.biases = np.array([0 for i in range(numNeurons)])

        self.size = numNeurons
    
    def initWeights(self, nextLayer):
        self.weights = np.array([[0.5 for i in range(self.size)] for j in range(nextLayer.size)])
        self.__nextLayer = nextLayer
    
    def sigmoid(self, x):
        return 1/(1+(math.e ** -x))

    def returnActivations(self):
        return (self.neurons)
    
    def calculateActivations(self, inputs):
        self.neurons = self.sigmoid((inputs + self.biases))
    
    def propogate(self):
        self.__nextLayer.calculateActivations(np.dot(self.weights, self.neurons))
    
    def evolve(self, weightsBaseline, biasBaseline, change):
        z = 0
        for i in range(self.size):
            for j in range(self.__nextLayer.size):
                self.weights[j][i] = weightsBaseline[j][i] + random.uniform((change*-1), change)
                z += 1
            self.biases[i] = biasBaseline[i] + random.uniform((change*-1), change)

class Input_Layer(Layer):
    type = "input"
    def calculateActivations(self, inputs):
        self.neurons = self.sigmoid(inputs)

class Hidden_Layer(Layer):
    type = "hidden"
    pass

class Output_Layer(Layer):
    type = "output"
    def evolve(self, biasBaseline, change):
        for i in range(self.size):
            self.biases[i] = biasBaseline[i] + random.uniform((change*-1), change)