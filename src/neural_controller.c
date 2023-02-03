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
#include <neural_controller.h>

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
int learn_loop(struct neuralControllerConfig *ncConfig, double *ncOutput) {
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

    double input[ncConfig->inputs];
    // TODO: Evtl. dasselbe wie bei den weights
    double target = 0.7;

    input[0] = 0.3;
    input[1] = 0.9;
    input[2] = 0.5;

    /*Initialize weights and bias with random values between 0 and 1 and
      initialize the rest with 0*/
    srand(time(NULL));
    for (int i = 0; i < ncConfig->layers + 1; i++) {
        for (int j = 0; j < ncConfig->neurons; j++) {
            /*Initialize weights between inputs and first layer*/
            if (i == 0) {
                for (int k = 0; k < ncConfig->inputs; k++) {
                    weights[i][j][k] = (double)rand() / RAND_MAX;
                }
                /*Initialize weights in hidden layers between first and second to last*/
            } else if (i > 0 && i < ncConfig->layers) {
                for (int k = 0; k < ncConfig->neurons; k++) {
                    weights[i][j][k] = (double)rand() / RAND_MAX;
                }
                /*Initialize weights between last hidden layer and output layer*/
            } else if (i == ncConfig->layers) {
                for (int k = 0; k < ncConfig->output_layer_neurons; k++) {
                    weights[i][j][k] = (double)rand() / RAND_MAX;
                }
            }
            /*Initialize bias and everything else in the neuron struct*/
            neuron[i][j].bias = (double)rand() / RAND_MAX;
            neuron[i][j].netinput = 0.0;
            neuron[i][j].netoutput = 0.0;
            neuron[i][j].sigma = 0.0;
        }
    }

    /*Feed Forward*/
    int epoch = 0;
    while (epoch < ncConfig->max_epochs) {
        epoch += 1;
        for (int i = 0; i < ncConfig->layers; i++) {
            for (int j = 0; j < ncConfig->neurons; j++) {
                /*Input layer*/
                if (i == 0) {
                    double sum = 0.0;
                    for (int k = 0; k < ncConfig->inputs; k++) {
                        // k = vorheriger Layer = Inputneuronen
                        sum += input[k] * weights[i][j][k];
                    }
                    neuron[i][j].netinput = sum;
                    neuron[i][j].netoutput = tanh(sum);
                    /*Hidden layers between input and output layer*/
                } else {
                    double sum = 0.0;
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        // k = alle Vorgängerneuronen, also von k(vorher) -> j(jetziges)
                        sum += neuron[i - 1][k].netoutput * weights[i][j][k];
                    }
                    neuron[i][j].netinput = sum;
                    neuron[i][j].netoutput = tanh(sum);
                }
            }
        }
        /*Output layer*/
        for (int j = 0; j < ncConfig->output_layer_neurons; j++) {
            double sum = 0.0;
            for (int k = 0; k < ncConfig->neurons; k++) {
                // TODO: Warum [k][j] und nicht [j][k]
                sum += neuron[ncConfig->layers - 1][k].netoutput * weights[ncConfig->layers][k][j];
            }
            neuron[ncConfig->layers][j].netinput = sum;
            neuron[ncConfig->layers][j].netoutput = sum;
            // Linear output, no RELU, sigmoid or hyperbolic tangent
        }

        /*Backpropagation*/
        /*For detailed explaination see https://de.wikipedia.org/wiki/Backpropagation#Neuronenausgabe*/
        /*Start at output layer*/
        for (int i = ncConfig->layers + 1; i >= 0; i--) {
            double sigma = 0.0;
            /*Output layer*/
            if (i == ncConfig->layers + 1) {
                int j = 0;
                while (j < ncConfig->output_layer_neurons) {
                    sigma = (tanh_deriv(neuron[i - 1][j].netinput) * (neuron[ncConfig->layers][j].netoutput - target));
                    neuron[i - 1][j].sigma = sigma;
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        weights[i - 1][j][k] += (-ncConfig->learning_rate * sigma * neuron[i - 2][k].netoutput);
                    }
                    j++;
                }
                /*Layer after output layer*/
            } else if (i == ncConfig->layers) {
                int j = 0;
                while (j < ncConfig->neurons) {
                    double sum = 0.0;
                    for (int k = 0; k < ncConfig->output_layer_neurons; k++) {
                        sum += neuron[i][k].sigma * weights[i][j][k];
                    }
                    sigma = (tanh_deriv(neuron[i - 1][j].netinput) * sum);
                    neuron[i - 1][j].sigma = sigma;
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        weights[i - 1][j][k] += (-ncConfig->learning_rate * sigma * neuron[i - 2][k].netoutput);
                    }
                    j++;
                }
                /*Rest*/
            } else {
                int j = 0;
                while (j < ncConfig->neurons) {
                    double sum = 0.0;
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        sum += neuron[i][k].sigma * weights[i][j][k];
                    }
                    sigma = (tanh_deriv(neuron[i - 1][j].netinput) * sum);
                    neuron[i - 1][j].sigma = sigma;
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        weights[i - 1][j][k] += (-ncConfig->learning_rate * sigma * neuron[i - 2][k].netoutput);
                    }
                    j++;
                }
            }
        }
        /*Write the output in output array pointer*/
        for (int i = 0; i < ncConfig->output_layer_neurons; i++) {
            ncOutput[i] = neuron[ncConfig->layers][i].netoutput;
        }
        printf("Output: %f\n", neuron[ncConfig->layers][0].netoutput);
    }
    printf("Target: %f\nOutput: %f\n", target, neuron[ncConfig->layers][0].netoutput);
    return 0;
}

/**
 * @brief C function for the hyberbolic tangent
 *
 * @param x x value for the dervative of the hyberbolic tangent
 * @return y value for the dervative of the hyberbolic tangent
 */
double tanh_deriv(double x) { return (pow(1 - tanh(x), 2)); }
