/**
 * @file neural_controller.c
 * @author Jakob Schatzl
 * @brief Implementation of library functions for the neural controller
 * @version 0.1
 * @date 2023-01-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "neural_controller.h"

/**
 * @brief Main loop for learning
 *
 * In the main learning loop, the neural network algorithm will learn to a specified
 * target and output that with the pointer ncOutput given in the function parameters.
 *
 * @param ncConfig Config for the neural network
 * @param ncOutput Output array
 * @return -
 */
int learn_loop(struct neuralControllerConfig *ncConfig, double *ncOutput, double *pError_array) {
    // create neurons
#if INPUTS_BT_NEURONS == 0
    struct neuron neuron[ncConfig->layers + 1][ncConfig->neurons];
    double weights[ncConfig->layers + 1][ncConfig->neurons][ncConfig->neurons];
    memset(weights, 0, (ncConfig->layers + 1) * (ncConfig->neurons) * (ncConfig->neurons) * sizeof(double));
#else
    struct neuron neuron[ncConfig->layers + 1][ncConfig->inputs];
    double weights[ncConfig->layers + 1][ncConfig->inputs][ncConfig->neurons];
    memset(weights, 0, (ncConfig->layers + 1) * (ncConfig->inputs) * (ncConfig->neurons) * sizeof(double));
#endif

    // training data
    double training_inputs[4][2] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    double training_outputs[4][1] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};
    double error = 0.0;
    double *error_array = malloc(ncConfig->max_epochs * sizeof(double));
    memset(error_array, 0, ncConfig->max_epochs * sizeof(double));
    int error_cnt = 0;
    int total_neurons = ncConfig->neurons * ncConfig->layers + ncConfig->output_layer_neurons;
    int total_weights = (ncConfig->inputs * ncConfig->neurons) + (ncConfig->neurons * ncConfig->neurons * (ncConfig->layers - 1)) + (ncConfig->neurons * ncConfig->output_layer_neurons);
    int topology[ncConfig->layers + 2];

    for (int i = 0; i < ncConfig->layers + 2; i++) {
        if (i == ncConfig->layers + 1) {
            topology[i] = ncConfig->output_layer_neurons;
        } else if (i == 0) {
            topology[i] = ncConfig->inputs;
        } else {
            topology[i] = ncConfig->neurons;
        }
    }
    // double input[ncConfig->inputs];

    /*Initialize weights and bias with random values between 0 and 1 and
      initialize the rest with 0*/
    srand(time(NULL));
    for (int layer = 0; layer < ncConfig->layers + 1; layer++) {
        for (int j = 0; j < topology[layer]; j++) {
            /*Initialize weights between inputs and first layer*/
            for (int k = 0; k < topology[layer + 1]; k++) {
                weights[layer][j][k] = (double)rand() / (double)RAND_MAX;
            }
            /*Initialize bias and everything else in the neuron struct*/
            neuron[layer][j].bias = (double)rand() / RAND_MAX;
            neuron[layer][j].netinput = 0.0;
            neuron[layer][j].netoutput = 0.0;
            neuron[layer][j].sigma = 0.0;
        }
    }

    int epoch = 0;
    int trainingSetOrder[] = {0, 1, 2, 3};
    int trainingCnt = 0;
    int n = 0;
    int w = 0;
    while (epoch < ncConfig->max_epochs) {
        epoch += 1;
        shuffle(trainingSetOrder, ncConfig->inputs);

        for (int trainingSetCnt = 0; trainingSetCnt < 4; trainingSetCnt++) {
            trainingCnt = trainingSetOrder[trainingSetCnt];

            /*Forward pass*/
            for (int layer = 0; layer < ncConfig->layers + 1; layer++) {
                for (int j = 0; j < topology[layer + 1]; j++) {
                    /*First hidden layer*/
                    double sum = neuron[layer][j].bias;
                    for (int k = 0; k < topology[layer]; k++) {
                        // k = vorheriger Layer = Inputneuronen
                        if (layer == 0)
                            sum += training_inputs[trainingCnt][k] * weights[layer][k][j];
                        else
                            sum += neuron[layer - 1][k].netoutput * weights[layer][k][j];
                    }
                    neuron[layer][j].netinput = sum;
                    neuron[layer][j].netoutput = sigmoid(sum);
                    n++;
                }
            }
            assert(n == total_neurons);
            n = 0;

            // Calculate error
            for (int i = 0; i < ncConfig->output_layer_neurons; i++) {
                error += fabs(training_outputs[trainingCnt][0] - neuron[ncConfig->layers][i].netoutput);
            }
            if (error_cnt == 3) {
                *(error_array + epoch - 1) = error;
                error_cnt = 0;
                error = 0;
            } else {
                error_cnt++;
            }
            /*Backpropagation*/
            /*For detailed explaination see https://de.wikipedia.org/wiki/Backpropagation#Neuronenausgabe*/
            /**
             * next layer     = k = layer + 1
             * current layer  = j = layer
             * previous layer = i = layer - 1
             */
            /*Start at output layer*/

            for (int layer = ncConfig->layers; layer >= 0; layer--) {
                for (int neuronC = 0; neuronC < topology[layer + 1]; neuronC++) {
                    /*Output layer uses different algorithm to determine the error signal,
                    therefore the program branches here
                     */
                    if (layer == ncConfig->layers) {
                        double outputError = training_outputs[trainingCnt][0] - neuron[layer][neuronC].netoutput;
                        double sigma = outputError * dSigmoid(neuron[layer][neuronC].netinput);
                        for (int k = 0; k < topology[layer]; k++) {
                            weights[layer][k][neuronC] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                            w++;
                        }
                        neuron[layer][neuronC].bias += ncConfig->learning_rate * sigma;
                        neuron[layer][neuronC].sigma = sigma;
                        n++;
                    } else {
                        double errorSum = 0;
                        for (int k = 0; k < topology[layer + 2]; k++) {
                            errorSum += neuron[layer + 1][k].sigma * weights[layer + 1][neuronC][k];
                        }
                        double sigma = errorSum * dSigmoid(neuron[layer][neuronC].netinput);
                        for (int k = 0; k < topology[layer]; k++) {
                            if (layer > 0)
                                weights[layer][k][neuronC] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                            else
                                weights[layer][k][neuronC] += ncConfig->learning_rate * sigma * training_inputs[trainingCnt][k];
                            w++;
                        }
                        neuron[layer][neuronC].bias += ncConfig->learning_rate * sigma;
                        neuron[layer][neuronC].sigma = sigma;
                        n++;
                    }
                }
            }
            assert(w == total_weights);
            assert(n = total_neurons);
            w = 0;
            n = 0;

            /*Write the output in output array pointer*/
            for (int i = 0; i < ncConfig->output_layer_neurons; i++) {
                ncOutput[i] = neuron[ncConfig->layers][i].netoutput;
            }
            printf("Inputs: ");
            for (int i = 0; i < 2; i++)
                printf(" %.2f ", training_inputs[trainingCnt][i]);
            printf("Target: %.2f Output: %.2f\n", training_outputs[trainingCnt][0], neuron[ncConfig->layers][0].netoutput);
        }
    }
    // Print bias
    printf("Bias: \n");
    for (int layer = 0; layer <= ncConfig->layers; layer++) {
        for (int j = 0; j < ncConfig->neurons; j++) {
            printf("Layer:%d Neuron:%d Value:%f\n", layer, j, neuron[layer][j].bias);
        }
    }
    // Print weights
    printf("Weights \n");
    for (int layer = 0; layer <= ncConfig->layers; layer++) {
        for (int j = 0; j < ncConfig->neurons; j++) {
            for (int k = 0; k < ncConfig->neurons; k++) {
                printf("Layer:%d Neuron:%d Weights:%d Value:%f\n", layer, j, k, weights[layer][j][k]);
            }
        }
    }
    for (int i = 0; i < ncConfig->max_epochs; i++) {
        *(pError_array + i) = *(error_array + i);
    }
    free(error_array);
    return 0;
}

/**
 * @brief C function for the hyberbolic tangent
 *
 * @param x x value for the dervative of the hyberbolic tangent
 * @return y value for the dervative of the hyberbolic tangent
 */
double tanh_deriv(double x) {
    double th = tanh(x);
    return 1.0 - th * th;
}

/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
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

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}
