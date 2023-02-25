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

    // double input[ncConfig->inputs];

    /*Initialize weights and bias with random values between 0 and 1 and
      initialize the rest with 0*/
    srand(time(NULL));
    for (int layer = 0; layer < ncConfig->layers + 1; layer++) {
        for (int j = 0; j < ncConfig->neurons; j++) {
            /*Initialize weights between inputs and first layer*/
            if (layer == 0) {
                for (int k = 0; k < ncConfig->inputs; k++) {
                    weights[layer][j][k] = (double)rand() / (double)RAND_MAX;
                }
                /*Initialize weights in hidden layers between first and second to last*/
            } else if (layer > 0 && layer < ncConfig->layers) {
                for (int k = 0; k < ncConfig->neurons; k++) {
                    weights[layer][j][k] = (double)rand() / (double)RAND_MAX;
                }
                /*Initialize weights between last hidden layer and output layer*/
            } else if (layer == ncConfig->layers) {
                for (int k = 0; k < ncConfig->output_layer_neurons; k++) {
                    weights[layer][j][k] = (double)rand() / (double)RAND_MAX;
                }
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
    while (epoch < ncConfig->max_epochs) {
        error = soll - yn;
        input[0] = error;
        input[1] = yn;
        input[2] = yn2;
        epoch += 1;
        shuffle(trainingSetOrder, 4);

        for (int trainingSetCnt = 0; trainingSetCnt < 4; trainingSetCnt++) {
            trainingCnt = trainingSetOrder[trainingSetCnt];

            /*Forward pass*/
            for (int layer = 0; layer < ncConfig->layers; layer++) {
                for (int j = 0; j < ncConfig->neurons; j++) {
                    /*First hidden layer*/
                    if (layer == 0) {
                        double sum = neuron[layer][j].bias;
                        for (int k = 0; k < ncConfig->inputs; k++) {
                            // k = vorheriger Layer = Inputneuronen
                            sum += training_inputs[trainingCnt][k] * weights[layer][k][j];
                        }
                        neuron[layer][j].netinput = sum;
                        neuron[layer][j].netoutput = sigmoid(sum);
                        /*Hidden layers between first hidden and output layer*/
                    } else {
                        double sum = neuron[layer][j].bias;
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            // k = alle Vorgängerneuronen, also von k(vorher) -> j(jetziges)
                            sum += neuron[layer - 1][k].netoutput * weights[layer][k][j];
                        }
                        neuron[layer][j].netinput = sum;
                        neuron[layer][j].netoutput = sigmoid(sum);
                    }
                }
            }
            /*Output layer*/
            for (int j = 0; j < ncConfig->output_layer_neurons; j++) {
                double sum = neuron[ncConfig->layers][j].bias;
                for (int k = 0; k < ncConfig->neurons; k++) {
                    sum += neuron[ncConfig->layers - 1][k].netoutput * weights[ncConfig->layers][k][j];
                }
                neuron[ncConfig->layers][j].netinput = sum;
                neuron[ncConfig->layers][j].netoutput = sigmoid(sum);
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

            // Calculate error
            for (int i = 0; i < ncConfig->output_layer_neurons; i++) {
                error += fabs(training_outputs[trainingCnt][0] - neuron[ncConfig->layers][i].netoutput);
            }
            if (error_cnt == 3) {
                error_array[epoch - 1] = error;
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
                /*Output layer*/
                if (layer == ncConfig->layers) {
                    for (int j = 0; j < ncConfig->output_layer_neurons; j++) {
                        double outputError = training_outputs[trainingCnt][0] - neuron[ncConfig->layers][j].netoutput;
                        double sigma = outputError * dSigmoid(neuron[layer][j].netoutput);
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            weights[layer][k][j] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                        }
                        neuron[layer][j].bias += ncConfig->learning_rate * sigma;
                        neuron[layer][j].sigma = sigma;
                    }
                    /*Hidden Layers*/
                } else if ((layer >= 1) && (layer < ncConfig->layers) && (ncConfig->layers != 1)) {
                    for (int j = 0; j < ncConfig->neurons; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            sum += neuron[layer + 1][k].sigma * weights[layer + 1][j][k];
                        }
                        double sigma = sum * dSigmoid(neuron[layer][j].netoutput);
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            weights[layer][k][j] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                        }
                        neuron[layer][j].bias += ncConfig->learning_rate * sigma;
                        neuron[layer][j].sigma = sigma;
                    }
                    /*First Hidden Layer*/
                } else {
                    for (int j = 0; j < ncConfig->inputs; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            sum += neuron[layer + 1][k].sigma * weights[layer + 1][j][k];
                        }
                        double sigma = sum * dSigmoid(neuron[layer][j].netoutput);
                        for (int k = 0; k < ncConfig->inputs; k++) {
                            weights[layer][k][j] += ncConfig->learning_rate * sigma * training_inputs[trainingCnt][k];
                        }
                        neuron[layer][j].bias += ncConfig->learning_rate * sigma;
                        neuron[layer][j].sigma = sigma;
                    }
                }
            }
            /*Write the output in output array pointer*/
            for (int i = 0; i < ncConfig->output_layer_neurons; i++) {
                ncOutput[i] = neuron[ncConfig->layers][i].netoutput;
            }
            printf("Inputs: ");
            for (int i = 0; i < 2; i++)
                printf(" %f ", training_inputs[trainingCnt][i]);
            printf("Target: %f Output: %f\n", training_outputs[trainingCnt][0], neuron[ncConfig->layers][0].netoutput);
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
        *(pError_array + i) = error_array[i];
    }
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
double dSigmoid(double x) { return x * (1 - x); }

struct i2_path_s i2_path(double K, double T, double yn, double yn2, double uk) {
    struct i2_path_s i2;

    i2.yn_12 = yn2 + K * uk * T;
    i2.yn_1 = yn + K * i2.yn_12 * T;
    return i2;
}
