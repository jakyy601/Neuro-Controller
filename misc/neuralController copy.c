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
#include "neuralController.h"

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
int learn_loop(struct neuralControllerConfig* ncConfig, double* pError_array) {
    // create neurons
#if INPUTS_BIGGER_THAN_NEURONS == 0
    struct neuron neuron[ncConfig->layers - 1][ncConfig->neurons];
    double weights[ncConfig->layers - 1][ncConfig->neurons][ncConfig->neurons];
    memset(weights, 0, (ncConfig->layers - 1) * (ncConfig->neurons) * (ncConfig->neurons) * sizeof(double));
#else
    struct neuron neuron[ncConfig->layers - 1][ncConfig->inputs];
    double weights[ncConfig->layers - 1][ncConfig->inputs][ncConfig->neurons];
    memset(weights, 0, (ncConfig->layers - 1) * (ncConfig->inputs) * (ncConfig->neurons) * sizeof(double));
#endif

    double input[ncConfig->inputs];
    memset(input, 0, ncConfig->inputs * sizeof(double));

    double K = 1.0;
    double T = 0.1;
    double yn = 0.0;
    double yn2 = 0.0;
    double soll = 0.5;
    double act_old = soll - 0;
    double act_new = 0.0;
    double d2 = 0.0;
    double rating = 0.0;

    // double *error_array = calloc(ncConfig->max_epochs, sizeof(double));
    int error_cnt = 0;
    int total_neurons = ncConfig->neurons * ncConfig->hidden_layers + ncConfig->output_layer_neurons;
    int total_weights = (ncConfig->inputs * ncConfig->neurons) + (ncConfig->neurons * ncConfig->neurons * (ncConfig->hidden_layers - 1)) + (ncConfig->neurons * ncConfig->output_layer_neurons);
    int topology[ncConfig->layers];
    for (int i = 0; i < ncConfig->layers; i++) {
        if (i == ncConfig->layers - 1) {
            topology[i] = ncConfig->output_layer_neurons;
        } else if (i == 0) {
            topology[i] = ncConfig->inputs;
        } else {
            topology[i] = ncConfig->neurons;
        }
    }

    struct i2_path_s i2;
    i2.yn_1 = 0.0;
    i2.yn_12 = 0.0;

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

    for (int epoch = 0; epoch < ncConfig->max_epochs; epoch++) {
        double error = soll - yn;
        int n = 0;
        int w = 0;
        input[0] = error;
        input[1] = yn;
        input[2] = yn2;

        /*Forward pass*/
        for (int layer = 0; layer < ncConfig->layers - 1; layer++) {
            for (int j = 0; j < topology[layer + 1]; j++) {
                /*First hidden layer*/
                double sum = neuron[layer][j].bias;
                for (int k = 0; k < topology[layer]; k++) {
                    // k = vorheriger Layer = Inputneuronen
                    if (layer == 0)
                        sum += input[k] * weights[layer][k][j];
                    else
                        sum += neuron[layer - 1][k].netoutput * weights[layer][k][j];
                }
                neuron[layer][j].netinput = sum;
                if (layer == ncConfig->hidden_layers)
                    neuron[layer][j].netoutput = tanh(sum);
                else
                    neuron[layer][j].netoutput = tanh(sum);
                n++;
            }
        }
        assert(n == total_neurons);
        n = 0;

        // Calculate I2 values
        i2 = i2_path(K, T, yn, yn2, neuron[ncConfig->hidden_layers][0].netoutput);
        d2 = i2.yn_12 - yn2;
        yn2 = i2.yn_12;
        *(pError_array + (epoch - 1)) = (double)yn;
        yn = i2.yn_1;

        act_new = soll - i2.yn_1 + d2;
        rating = (fabs(act_new) - fabs(act_old)) + act_new;
        act_old = act_new;

        /*Backpropagation*/
        /*For detailed explaination see https://de.wikipedia.org/wiki/Backpropagation#Neuronenausgabe*/
        /**
         * next layer     = k = layer + 1
         * current layer  = j = layer
         * previous layer = i = layer - 1
         */
        /*Start at output layer*/
        for (int layer = ncConfig->hidden_layers; layer >= 0; layer--) {
            for (int neuronC = 0; neuronC < topology[layer + 1]; neuronC++) {
                /*Output layer uses different algorithm to determine the error signal,
                therefore the program branches here
                 */
                if (layer == ncConfig->hidden_layers) {
                    double sigma = rating * dTanh(neuron[layer][neuronC].netoutput);
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
                    double sigma = errorSum * dTanh(neuron[layer][neuronC].netinput);
                    for (int k = 0; k < topology[layer]; k++) {
                        if (layer > 0)
                            weights[layer][k][neuronC] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                        else
                            weights[layer][k][neuronC] += ncConfig->learning_rate * sigma * input[k];
                        w++;
                    }
                    neuron[layer][neuronC].bias += ncConfig->learning_rate * sigma;
                    neuron[layer][neuronC].sigma = sigma;
                    n++;
                }
            }
        }
        assert(w == total_weights);
        assert(n == total_neurons);
        w = 0;
        n = 0;

        printf("Inputs: ");
        for (int i = 0; i < ncConfig->inputs; i++)
            printf(" %f ", input[i]);
        printf("Target: %f Output: %f\n", soll, neuron[ncConfig->hidden_layers][0].netoutput);
    }

    // Print bias
    printf("Bias: \n");
    for (int layer = 0; layer <= ncConfig->hidden_layers; layer++) {
        for (int j = 0; j < ncConfig->neurons; j++) {
            printf("Layer:%d Neuron:%d Value:%f\n", layer, j, neuron[layer][j].bias);
        }
    }
    // Print weights
    printf("Weights \n");
    for (int layer = 0; layer <= ncConfig->hidden_layers; layer++) {
        for (int j = 0; j < ncConfig->neurons; j++) {
            for (int k = 0; k < ncConfig->neurons; k++) {
                printf("Layer:%d Neuron:%d Weights:%d Value:%f\n", layer, j, k, weights[layer][j][k]);
            }
        }
    }
    // memcpy(pError_array, error_array, ncConfig->max_epochs * sizeof(double));

    return 0;
}