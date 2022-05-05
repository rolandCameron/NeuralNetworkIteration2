import pygame
import sys
import numpy as np
import random
import NeuralNetworkOverhaul as NN
import NeuralNetworkVisualisation as Vis
from pygame.locals import *

pygame.init()


fps = 60
fpsClock = pygame.time.Clock()

width, height = pygame.display.Info().current_w*0.9, pygame.display.Info().current_h*0.9
screen = pygame.display.set_mode((width, height))

neuralNetwork = [NN.Input_Layer(4), NN.Hidden_Layer(4), NN.Output_Layer(4)]

networkLen = len(neuralNetwork)

for i in range(networkLen - 1):
    neuralNetwork[i].initWeights(neuralNetwork[i + 1])
    neuralNetwork[i].evolve(neuralNetwork[i].weights, neuralNetwork[i].biases, 5)

neuralNetwork[-1].evolve(neuralNetwork[-1].biases, 5)

neuralNetwork[0].calculateActivations(np.array([random.randint(0, 1) for i in range(neuralNetwork[0].size)]))

for i in range(networkLen - 1):
    neuralNetwork[i].propogate()

Vis.initialise(neuralNetwork, (width, height))

# Game loop.
while True:
    screen.fill((0, 0, 0))   

    Vis.visualise(neuralNetwork, (146, 186, 146), (99, 38, 38), screen)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # Update.

    # Draw.
    pygame.display.flip()
    fpsClock.tick(fps)