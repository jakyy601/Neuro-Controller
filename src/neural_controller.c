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
    memset(input, 0, ncConfig->inputs * sizeof(double));
    // // TODO: Evtl. dasselbe wie bei den weights
    // double target = 0.7;

    // input[0] = 0.3;
    // input[1] = 0.9;
    // input[2] = 0.5;

    double K = 1.0;
    double T = 0.1;
    double yn = 0.0;
    double yn2 = 0.0;
    double soll = 1.0;
    double act_old = soll - 0;
    double act_new = 0.0;
    double d2 = 0.0;
    double rating = 0.0;

    struct i2_path_s i2;
    i2.yn_1 = 0.0;
    i2.yn_12 = 0.0;

    int epoch = 0;
    double error = 0.0;

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
    while (epoch < ncConfig->max_epochs) {
        error = soll - yn;
        input[0] = error;
        input[1] = yn;
        input[2] = yn2;
        epoch += 1;
        for (int i = 0; i < ncConfig->layers; i++) {
            for (int j = 0; j < ncConfig->neurons; j++) {
                /*Input layer*/
                if (i == 0) {
                    double sum = 0.0;
                    for (int k = 0; k < ncConfig->inputs; k++) {
                        sum += input[k] * weights[i][j][k];
                    }
                    neuron[i][j].netinput = sum;
                    neuron[i][j].netoutput = tanh(neuron[i][j].netinput);
                    /*Hidden layers between input and output layer*/
                } else {
                    double sum = 0.0;
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        sum += neuron[i - 1][k].netoutput * weights[i][j][k];
                    }
                    neuron[i][j].netinput = sum;
                    neuron[i][j].netoutput = tanh(neuron[i][j].netinput);
                }
            }
        }
        /*Output layer*/
        for (int j = 0; j < ncConfig->output_layer_neurons; j++) {
            double sum = 0.0;
            for (int k = 0; k < ncConfig->neurons; k++) {
                sum += neuron[ncConfig->layers - 1][k].netoutput * weights[ncConfig->layers][k][j];
            }
            neuron[ncConfig->layers][j].netinput = sum;
            neuron[ncConfig->layers][j].netoutput = sum;
            // Linear output, no RELU, sigmoid or hyperbolic tangent
        }

        // Calculate I2 values
        i2 = i2_path(K, T, yn, yn2, neuron[ncConfig->layers][0].netoutput);
        d2 = i2.yn_12 - yn2;
        yn2 = i2.yn_12;
        yn = i2.yn_1;

        act_new = soll - i2.yn_1 + d2;
        rating = (fabs(act_new) - fabs(act_old)) + act_new;
        act_old = act_new;

        /*Backpropagation*/
        /*For detailed explaination see https://de.wikipedia.org/wiki/Backpropagation#Neuronenausgabe*/
        /*Start at output layer*/
        for (int i = ncConfig->layers + 1; i >= 0; i--) {
            double sigma = 0.0;
            /*Output layer*/
            if (i == ncConfig->layers + 1) {
                int j = 0;
                while (j < ncConfig->output_layer_neurons) {
                    // sigma = (tanh_deriv(neuron[i - 1][j].netinput) * (neuron[ncConfig->layers][j].netoutput - target));
                    sigma = (rating * tanh_deriv(neuron[i - 1][j].netinput));
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
                        weights[i - 1][j][k] +=
                            (-ncConfig->learning_rate * sigma * neuron[i - 2][k].netoutput);
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
                        weights[i - 1][j][k] +=
                            (-ncConfig->learning_rate * sigma * neuron[i - 2][k].netoutput);
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
    // printf("Target: %f\nOutput: %f\n", target, neuron[ncConfig->layers][0].netoutput);
    printf("Output: %f\n", neuron[ncConfig->layers][0].netoutput);
    return 0;
}

/**
 * @brief C function for the hyberbolic tangent
 *
 * @param x x value for the dervative of the hyberbolic tangent
 * @return y value for the dervative of the hyberbolic tangent
 */
double tanh_deriv(double x) { return (pow(1 - tanh(x), 2)); }

struct i2_path_s i2_path(double K, double T, double yn, double yn2, double uk) {
    struct i2_path_s i2;

    i2.yn_12 = yn2 + K * uk * T;
    i2.yn_1 = yn + K * i2.yn_12 * T;
    return i2;
}