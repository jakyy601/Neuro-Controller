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

static neuron_st neuron[LAYERS - 1][NEURONS];
static double weights[LAYERS - 1][NEURONS][NEURONS];

static double act_old = 0;
static double act_new = 0;
static double rating = 0;
static int total_neurons = 0;
static int total_weights = 0;
static int topology[LAYERS];
double input[INPUTS];
double input_old[INPUTS];

int neuralController_Init(neuralControllerConfig_st* ncConfig, float (*fctPtr)()) {
    memset(input, 0, ncConfig->inputs * sizeof(double));
    memset(weights, 0, (ncConfig->layers - 1) * (ncConfig->neurons) * (ncConfig->neurons) * sizeof(double));
    act_old = ncConfig->setpoint - 0;

    // double *error_array = calloc(ncConfig->max_epochs, sizeof(double));
    total_neurons = ncConfig->neurons * ncConfig->hidden_layers + ncConfig->output_layer_neurons;
    total_weights = (ncConfig->inputs * ncConfig->neurons) + (ncConfig->neurons * ncConfig->neurons * (ncConfig->hidden_layers - 1)) + (ncConfig->neurons * ncConfig->output_layer_neurons);
    for (int i = 0; i < ncConfig->layers; i++) {
        if (i == ncConfig->layers - 1) {
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
    for (int layer = 0; layer < ncConfig->layers; layer++) {
        for (int j = 0; j < topology[layer]; j++) {
            /*Initialize weights between inputs and first layer*/
            for (int k = 0; k < topology[layer + 1]; k++) {
                if (layer == ncConfig->layers - 1)
                    break;
                weights[layer][j][k] = (double)(*fctPtr)();
            }
            /*Initialize bias and everything else in the neuron struct*/
            neuron[layer][j].bias = (double)(*fctPtr)();
            neuron[layer][j].netinput = 0.0;
            neuron[layer][j].netoutput = 0.0;
            neuron[layer][j].sigma = 0.0;
        }
    }
    ncConfig->initialized = 1;
}

int neuralController_Run(neuralControllerConfig_st* ncConfig, double* pOutput, float* pInput) {
    int n = 0;
    int w = 0;
    float d2 = 0;

    input[0] = ncConfig->setpoint - pInput[0];
    for (int input_cnt = 0; input_cnt < ncConfig->inputs - 1; input_cnt++) {
        input[input_cnt + 1] = pInput[input_cnt];
    }
    /*Forward pass*/
    for (int layer = 0; layer < ncConfig->layers - 1; layer++) {
        for (int j = 0; j < topology[layer + 1]; j++) {
            /*First hidden layer*/
            // double sum = neuron[layer][j].bias;
            double sum = 0;
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

    d2 = input[INPUTS - 1] - input_old[INPUTS - 1];
    memcpy(input_old, &input, ncConfig->inputs);

    act_new = ncConfig->setpoint - input[1];
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
                double sigma = rating * dTanh(neuron[layer][neuronC].netinput);
                for (int k = 0; k < topology[layer]; k++) {
                    weights[layer][k][neuronC] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                    w++;
                }
                // neuron[layer][neuronC].bias += ncConfig->learning_rate * sigma;
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
                // neuron[layer][neuronC].bias += ncConfig->learning_rate * sigma;
                neuron[layer][neuronC].sigma = sigma;
                n++;
            }
        }
    }
    assert(w == total_weights);
    assert(n == total_neurons);
    w = 0;
    n = 0;

    *pOutput = neuron[ncConfig->hidden_layers][0].netoutput;
    return 0;
}

/**
 * @brief Sigmoid function
 *
 * @param x x value
 * @return y value
 */
double sigmoid(double x) { return 1 / (1 + exp(-x)); }

/**
 * @brief Derivative of the sigmoid function
 *
 *
 * @param x x value
 * @return y value
 */
double dSigmoid(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

/**
 * @brief C function for the hyberbolic tangent
 *
 * @param x x value for the dervative of the hyberbolic tangent
 * @return y value for the dervative of the hyberbolic tangent
 */
double dTanh(double x) {
    double th = tanh(x);
    return 1.0 - th * th;
}
