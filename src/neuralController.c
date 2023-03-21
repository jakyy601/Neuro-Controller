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
int learn_loop(struct neuralControllerConfig *ncConfig, double *pError_array, int *layer_sizes) {
    // create neurons
#if INPUTS_BT_NEURONS == 0
    struct neuron neuron[ncConfig->layers - 1][ncConfig->neurons];
    // double weights[ncConfig->layers - 1][ncConfig->neurons][ncConfig->neurons];
    // memset(weights, 0, (ncConfig->layers - 1) * (ncConfig->neurons) * (ncConfig->neurons) * sizeof(double));
#else
    struct neuron neuron[ncConfig->layers - 1][ncConfig->inputs];
    double weights[ncConfig->layers - 1][ncConfig->inputs][ncConfig->neurons];
    memset(weights, 0, (ncConfig->layers - 1) * (ncConfig->inputs) * (ncConfig->neurons) * sizeof(double));
#endif

    float ***weights = malloc((ncConfig->layers) * sizeof(float **));
    for (int i = 0; i < ncConfig->layers; i++) {
        weights[i] = malloc(layer_sizes[i] * sizeof(float *));
        for (int j = 0; j < layer_sizes[i]; j++) {
            weights[i][j] = malloc(layer_sizes[i + 1] * sizeof(float));
        }
    }
    initialize_weights(weights, layer_sizes, ncConfig->layers);

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

    double error_array[ncConfig->max_epochs];
    memset(error_array, 0, ncConfig->max_epochs);

    struct i2_path_s i2;
    i2.yn_1 = 0.0;
    i2.yn_12 = 0.0;

    // double input[ncConfig->inputs];

    /*Initialize weights and bias with random values between 0 and 1 and
      initialize the rest with 0*/
    srand(time(NULL));
    for (int layer = 0; layer < ncConfig->layers - 1; layer++) {
        for (int j = 0; j < ncConfig->neurons; j++) {
            /*Initialize weights between inputs and first layer*/
            // if (layer == 0) {
            //     for (int k = 0; k < ncConfig->inputs; k++) {
            //         weights[layer][j][k] = (float)rand() / (double)RAND_MAX;
            //     }
            //     /*Initialize weights in hidden layers between first and second to last*/
            // } else if (layer > 0 && layer < ncConfig->hidden_layers) {
            //     for (int k = 0; k < ncConfig->neurons; k++) {
            //         weights[layer][j][k] = (double)rand() / (double)RAND_MAX;
            //     }
            //     /*Initialize weights between last hidden layer and output layer*/
            // } else if (layer == ncConfig->hidden_layers) {
            //     for (int k = 0; k < ncConfig->output_layer_neurons; k++) {
            //         weights[layer][j][k] = (double)rand() / (double)RAND_MAX;
            //     }
            // }
            /*Initialize bias and everything else in the neuron struct*/
            neuron[layer][j].bias = (double)rand() / RAND_MAX;
            neuron[layer][j].netinput = 0.0;
            neuron[layer][j].netoutput = 0.0;
            neuron[layer][j].sigma = 0.0;
        }
    }

    for (int epoch = 0; epoch < ncConfig->max_epochs; epoch++) {
        double error = soll - yn;
        input[0] = error;
        input[1] = yn;
        input[2] = yn2;

        /*Forward pass*/
        for (int layer = 0; layer < ncConfig->hidden_layers; layer++) {
            for (int j = 0; j < ncConfig->neurons; j++) {
                /*First hidden layer*/
                if (layer == 0) {
                    double sum = 0;
                    for (int k = 0; k < ncConfig->inputs; k++) {
                        // k = vorheriger Layer = Inputneuronen
                        sum += input[k] * weights[layer][k][j];
                    }
                    neuron[layer][j].netinput = sum;
                    neuron[layer][j].netoutput = tanh(sum);
                    /*Hidden layers between first hidden and output layer*/
                } else {
                    double sum = 0;
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        // k = alle Vorgängerneuronen, also von k(vorher) -> j(jetziges)
                        sum += neuron[layer - 1][k].netoutput * weights[layer][k][j];
                    }
                    neuron[layer][j].netinput = sum;
                    neuron[layer][j].netoutput = tanh(sum);
                }
            }
        }
        /*Output layer*/
        for (int j = 0; j < ncConfig->output_layer_neurons; j++) {
            double sum = 0;
            for (int k = 0; k < ncConfig->neurons; k++) {
                sum += neuron[ncConfig->layers - 1][k].netoutput * weights[ncConfig->hidden_layers][k][j];
            }
            neuron[ncConfig->hidden_layers][j].netinput = sum;
            neuron[ncConfig->hidden_layers][j].netoutput = tanh(sum);
            // Linear output, no RELU, tanh or hyperbolic tangent
        }

        // Calculate I2 values
        i2 = i2_path(K, T, yn, yn2, neuron[ncConfig->hidden_layers][0].netoutput);
        d2 = i2.yn_12 - yn2;
        yn2 = i2.yn_12;
        error_array[epoch - 1] = yn;
        yn = i2.yn_1;

        act_new = soll - i2.yn_1 + d2;
        rating = (fabs(act_new) - fabs(act_old)) + act_new;
        act_old = act_new;

        // Calculate error
        // error_array[epoch - 1] = fabs(soll - neuron[ncConfig->layers][0].netoutput);

        /*Backpropagation*/
        /*For detailed explaination see https://de.wikipedia.org/wiki/Backpropagation#Neuronenausgabe*/
        /**
         * next layer     = k = layer + 1
         * current layer  = j = layer
         * previous layer = i = layer - 1
         */
        /*Start at output layer*/
        for (int layer = ncConfig->hidden_layers; layer >= 0; layer--) {
            /*Output layer*/
            if (layer == ncConfig->hidden_layers) {
                for (int j = 0; j < ncConfig->output_layer_neurons; j++) {
                    // double outputError = training_outputs[trainingCnt][0] - neuron[ncConfig->layers][j].netoutput;
                    // double sigma = outputError * tanh_deriv(neuron[layer][j].netoutput);
                    double sigma = rating * tanh_deriv(neuron[layer][j].netoutput);
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        weights[layer][k][j] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                    }
                    neuron[layer][j].bias += ncConfig->learning_rate * sigma;
                    neuron[layer][j].sigma = sigma;
                }
                /*Last Hidden Layer*/
            } else if (layer == (ncConfig->layers - 1)) {
                for (int j = 0; j < ncConfig->neurons; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < ncConfig->output_layer_neurons; k++) {
                        sum += neuron[layer + 1][k].sigma * weights[layer + 1][j][k];
                    }
                    double sigma = sum * tanh_deriv(neuron[layer][j].netoutput);
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        weights[layer][k][j] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                    }
                    neuron[layer][j].bias += ncConfig->learning_rate * sigma;
                    neuron[layer][j].sigma = sigma;
                }
                /*Hidden Layers*/
            } else if ((layer >= 1) && (layer < ncConfig->hidden_layers) && (ncConfig->hidden_layers != 1)) {
                for (int j = 0; j < ncConfig->neurons; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        sum += neuron[layer + 1][k].sigma * weights[layer + 1][j][k];
                    }
                    double sigma = sum * tanh_deriv(neuron[layer][j].netoutput);
                    for (int k = 0; k < ncConfig->neurons; k++) {
                        weights[layer][k][j] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                    }
                    neuron[layer][j].bias += ncConfig->learning_rate * sigma;
                    neuron[layer][j].sigma = sigma;
                }
                /*First Hidden Layer*/
            } else {
                for (int j = 0; j < ncConfig->neurons; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < ncConfig->output_layer_neurons; k++) {
                        sum += neuron[layer + 1][k].sigma * weights[layer + 1][j][k];
                    }
                    double sigma = sum * tanh_deriv(neuron[layer][j].netoutput);
                    for (int k = 0; k < ncConfig->inputs; k++) {
                        weights[layer][k][j] += ncConfig->learning_rate * sigma * input[k];
                    }
                    neuron[layer][j].bias += ncConfig->learning_rate * sigma;
                    neuron[layer][j].sigma = sigma;
                }
            }
        }
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
    for (int i = 0; i < ncConfig->max_epochs; i++) {
        *(pError_array + i) = error_array[i];
    }
    return 0;
}

void initialize_weights(float ***weights, int *layer_sizes, int num_layers) {
    srand(time(NULL));
    for (int i = 0; i < num_layers - 1; i++) {
        int current_layer_size = layer_sizes[i];
        int next_layer_size = layer_sizes[i + 1];
        for (int j = 0; j < next_layer_size; j++) {
            for (int k = 0; k < current_layer_size; k++) {
                weights[i][k][j] = ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
                printf("%f\n", weights[i][k][j]);
            }
        }
    }
}