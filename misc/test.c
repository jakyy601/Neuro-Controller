/*
 *  main.cpp
 *  NeuralNetwork
 *
 *  Created by Santiago Becerra on 9/15/19.
 *  Copyright © 2019 Santiago Becerra. All rights reserved.
 *
 *  Simple neural network implementation in C
 *  https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Simple network that can learn XOR
// Feartures : sigmoid activation function, stochastic gradient descent, and mean square error fuction

// Potential improvements :
// Different activation functions
// Batch training
// Different error funnctions
// Arbitrary number of hidden layers
// Read training end test data from a file
// Add visualization of training
// Add recurrence? (maybe that should be a separate project)

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }
double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }
void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main(void) {
#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1

    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

#define numTrainingSets 4

    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                          {1.0f, 0.0f},
                                                          {0.0f, 1.0f},
                                                          {1.0f, 1.0f}};
    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f}};

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weight();
        }
    }
    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = init_weight();
        }
    }
    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weight();
    }

    int trainingSetOrder[] = {0, 1, 2, 3};

    for (int n = 0; n < 10000; n++) {
        shuffle(trainingSetOrder, numTrainingSets);
        for (int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];

            // Forward pass

            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            printf("Input:%lg %lg    Output:%lg    Expected Output: %lg\n",
                   training_inputs[i][0], training_inputs[i][1],
                   outputLayer[0], training_outputs[i][0]);

            // Backprop

            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double errorOutput = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
            }

            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    errorHidden += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = errorHidden * dSigmoid(hiddenLayer[j]);
            }

            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

    // Print weights
    fputs("Final Hidden Weights\n[ ", stdout);
    for (int j = 0; j < numHiddenNodes; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numInputs; k++) {
            printf("%lf ", hiddenWeights[k][j]);
        }
        fputs("] ", stdout);
    }

    fputs("]\nFinal Hidden Biases\n[ ", stdout);
    for (int j = 0; j < numHiddenNodes; j++) {
        printf("%lf ", hiddenLayerBias[j]);
    }
    fputs("]\nFinal Output Weights", stdout);
    for (int j = 0; j < numOutputs; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numHiddenNodes; k++) {
            printf("%lf ", outputWeights[k][j]);
        }
        fputs("]\n", stdout);
    }
    fputs("Final Output Biases\n[ ", stdout);
    for (int j = 0; j < numOutputs; j++) {
        printf("%lf ", outputLayerBias[j]);
    }
    fputs("]\n", stdout);

    return 0;
}