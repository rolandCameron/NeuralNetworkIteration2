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

neuralNetwork = [NN.Input_Layer(5), NN.Hidden_Layer(5), NN.Output_Layer(5)]

networkLen = len(neuralNetwork)

for i in range(networkLen - 1):
    neuralNetwork[i].initWeights(neuralNetwork[i + 1])
    neuralNetwork[i].evolve(neuralNetwork[i].weights, neuralNetwork[i].biases, 5.0)

neuralNetwork[-1].evolve(neuralNetwork[-1].biases, 5.0)

neuralNetwork[0].calculateActivations(np.array([random.randint(0, 1) for i in range(neuralNetwork[0].size)]))

for i in range(networkLen - 1):
    neuralNetwork[i].propogate()

Vis.initialise(neuralNetwork, (width, height))

x = 0
y = 0

# Game loop.
while True: 
    if x == 5:
        x = 0
        screen.fill((0, 0, 0))  
        neuralNetwork[0].calculateActivations(np.array([random.randint(-10, 10) for i in range(neuralNetwork[0].size)]))
        for i in range(networkLen - 1):
            neuralNetwork[i].propogate()
        Vis.visualise(neuralNetwork, (255,192,203), (255, 165, 0), screen)

        if y == 5:
            y=0
            for i in range(networkLen - 1):
                neuralNetwork[i].initWeights(neuralNetwork[i + 1])
                neuralNetwork[i].evolve(neuralNetwork[i].weights, neuralNetwork[i].biases, 5)

            neuralNetwork[-1].evolve(neuralNetwork[-1].biases, 5)
        else:
            y+=1
    else:
        x+=1

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # Update.

    # Draw.
    pygame.display.flip()
    fpsClock.tick(fps)