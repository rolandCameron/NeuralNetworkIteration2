import pygame
import math

pygame.init()

global numLayers
global layerWidth
global neuronRadii

numLayers = 0
layerWidth = 0
neuronRadii = []

listColours = []

def initialise(NeuralNetwork, screenDimensions):
    global numLayers
    global layerWidth
    global neuronRadii

    screenX, screenY = screenDimensions[0], screenDimensions[1]
    numLayers = len(NeuralNetwork)

    layerWidth = screenX/(numLayers+1)
    neuronRadii = []
    for i in range(numLayers):
        radius = (screenY*0.45)/(NeuralNetwork[i].size + 1)
        if radius > (0.5*screenY):
            radius = 0.5*screenY
        neuronRadii.append(radius)

def srgbToLinear(inputColour):
    # https://stackoverflow.com/questions/34472375/linear-to-srgb-conversion
    linear = [0, 0, 0]
    for i in range(3):
        if (inputColour[i] <= 0.0404482362771082):
            linear[i] = inputColour[i]/12.92
        else:
            linear[i] = pow(((inputColour[i]+0.055)/1.055), 2.4)
    
    return (linear[0], linear[1], linear[2])

def linearToSRGB(inputColour):
    # https://stackoverflow.com/questions/34472375/linear-to-srgb-conversion
    sRGB = [0, 0, 0]
    for i in range(3):
        if (inputColour[i] > 0.0031308):
            sRGB[i] = 1.055 * (inputColour[i] ** (1.0 / 2.4)) - 0.055
        else:
            sRGB[i] = 12.92 * inputColour[i]

    return (sRGB[0], sRGB[1], sRGB[2])

def colourMix(colourOne, colourTwo, mix):
    # Mix runs from 0 - 1, it is the percentage of colour two present in the output
    # Colours must be linear, not sRGB
    # https://stackoverflow.com/questions/22607043/color-gradient-algorithm
    result = [0, 0, 0]

    for i in range(3):
        result[i] = (colourOne[i]*(1-mix) + colourTwo[i]*mix)
    
    return (result[0], result[1], result[2])

def colourBetween(colourOne, colourTwo, mix):
    # Mix is 1 - 0
    # Colours are in sRGB
    colourOneLinear = srgbToLinear(colourOne)
    colourTwoLinear = srgbToLinear(colourTwo)

    blend = colourMix(colourOneLinear, colourTwoLinear, mix)

    srgbBlend = linearToSRGB(blend)

    return srgbBlend

def visualise(NeuralNetwork, activeColour, inactiveColour, surface):
    networkActivations = []
    for i in range(numLayers):
        networkActivations.append(NeuralNetwork[i].returnActivations())
    
    for i in range(numLayers):
        layerX = layerWidth*(i+1)
        nextLayerX = layerWidth*(i+2)
        for j in range(len(networkActivations[i])):
            neuronActivation = networkActivations[i][j]

            activationAdjustedColour = colourBetween(activeColour, inactiveColour, neuronActivation)

            pygame.draw.circle(surface, activationAdjustedColour, (layerX, (j + 1)*neuronRadii[i]*2), (neuronRadii[i]*0.9))

        if NeuralNetwork[i].type != "output":
            for k in range(NeuralNetwork[i].size):
                for l in range(NeuralNetwork[i+1].size):
                    weight = NeuralNetwork[i].weights[l][k]
                    weight = 1/(1+math.e ** -weight)
                    weightAdjustedColour = colourBetween(activeColour, inactiveColour, weight)
                    listColours.append(weightAdjustedColour)
                    pygame.draw.line(surface, weightAdjustedColour, (layerX + neuronRadii[i], (k + 1)*neuronRadii[i]*2), (nextLayerX - neuronRadii[i + 1], (l + 1)*neuronRadii[i + 1]*2), width = int(math.ceil(neuronRadii[i]*0.1)))