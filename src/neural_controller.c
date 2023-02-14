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

    // training data
    double training_inputs[4][2] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    double training_outputs[4][1] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};
    double error = 0.0;
    double error_array[ncConfig->max_epochs];
    int error_cnt = 0;

    // double input[ncConfig->inputs];

    /*Initialize weights and bias with random values between 0 and 1 and
      initialize the rest with 0*/
    srand(time(NULL));
    for (int layer = 0; layer < ncConfig->layers + 1; layer++) {
        for (int j = 0; j < ncConfig->neurons; j++) {
            /*Initialize weights between inputs and first layer*/
            if (layer == 0) {
                for (int k = 0; k < ncConfig->inputs; k++) {
                    weights[layer][j][k] = (double)rand() / RAND_MAX;
                }
                /*Initialize weights in hidden layers between first and second to last*/
            } else if (layer > 0 && layer < ncConfig->layers) {
                for (int k = 0; k < ncConfig->neurons; k++) {
                    weights[layer][j][k] = (double)rand() / RAND_MAX;
                }
                /*Initialize weights between last hidden layer and output layer*/
            } else if (layer == ncConfig->layers) {
                for (int k = 0; k < ncConfig->output_layer_neurons; k++) {
                    weights[layer][j][k] = (double)rand() / RAND_MAX;
                }
            }
            /*Initialize bias and everything else in the neuron struct*/
            neuron[layer][j].bias = (double)rand() / RAND_MAX;
            neuron[layer][j].netinput = 0.0;
            neuron[layer][j].netoutput = 0.0;
            neuron[layer][j].sigma = 0.0;
        }
    }

    /*Feed Forward*/
    int epoch = 0;
    while (epoch < ncConfig->max_epochs) {
        epoch += 1;
        int trainingCnt = 0;
        int trainingSetOrder[] = {0, 1, 2, 3};
        shuffle(trainingSetOrder, 4);
        for (int layer = 0; layer < 4; layer++) {
            trainingCnt = trainingSetOrder[layer];
            for (int layer = 0; layer < ncConfig->layers; layer++) {
                for (int j = 0; j < ncConfig->neurons; j++) {
                    /*Input layer*/
                    if (layer == 0) {
                        double sum = 0.0;
                        for (int k = 0; k < ncConfig->inputs; k++) {
                            // k = vorheriger Layer = Inputneuronen
                            sum += training_inputs[trainingCnt][k] * weights[layer][j][k];
                        }
                        neuron[layer][j].netinput = sum;
                        neuron[layer][j].netoutput = tanh(sum);
                        /*Hidden layers between input and output layer*/
                    } else {
                        double sum = 0.0;
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            // k = alle Vorgängerneuronen, also von k(vorher) -> j(jetziges)
                            sum += neuron[layer - 1][k].netoutput * weights[layer][j][k];
                        }
                        neuron[layer][j].netinput = sum;
                        neuron[layer][j].netoutput = tanh(sum);
                    }
                }
            }
            /*Output layer*/
            for (int j = 0; j < ncConfig->output_layer_neurons; j++) {
                double sum = 0.0;
                for (int k = 0; k < ncConfig->neurons; k++) {
                    // FIXME: fix initialization then [j][k] can be used
                    sum += neuron[ncConfig->layers - 1][k].netoutput * weights[ncConfig->layers][k][j];
                }
                neuron[ncConfig->layers][j].netinput = sum;
                neuron[ncConfig->layers][j].netoutput = sum;
                // Linear output, no RELU, sigmoid or hyperbolic tangent
            }

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
             * layer = k
             * layer - 1 = j
             * layer - 2 = i
             */
            /*Start at output layer*/
            for (int layer = ncConfig->layers + 1; layer > 0; layer--) {
                double sigma = 0.0;
                /*Output layer*/
                if (layer == ncConfig->layers + 1) {
                    int j = 0;
                    while (j < ncConfig->output_layer_neurons) {
                        sigma = (tanh_deriv(neuron[layer - 1][j].netinput) * (neuron[ncConfig->layers][j].netoutput - training_outputs[trainingCnt][0]));
                        neuron[layer - 1][j].sigma = sigma;
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            //[k][j] instead of [j][k] because of case of only 1 output
                            // FIXME: fix initialization then [j][k] can be used
                            weights[layer - 1][k][j] += (-ncConfig->learning_rate * sigma * neuron[layer - 2][k].netoutput);
                        }
                        j++;
                    }
                    /*Hidden Layers*/
                } else if ((layer >= 1) && (layer < ncConfig->layers + 1) && (ncConfig->layers != 1)) {
                    int j = 0;
                    while (j < ncConfig->neurons) {
                        double sum = 0.0;
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            sum += neuron[layer][k].sigma * weights[layer][j][k];
                        }
                        sigma = (tanh_deriv(neuron[layer - 1][j].netinput) * sum);
                        neuron[layer - 1][j].sigma = sigma;
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            weights[layer - 1][j][k] += (-ncConfig->learning_rate * sigma * neuron[layer - 2][k].netoutput);
                        }
                        j++;
                    }
                    /*Input layer*/
                } else {
                    int j = 0;
                    while (j < ncConfig->neurons) {
                        double sum = 0.0;
                        for (int k = 0; k < ncConfig->neurons; k++) {
                            sum += neuron[layer][k].sigma * weights[layer][j][k];
                        }
                        sigma = (tanh_deriv(neuron[layer - 1][j].netinput) * sum);
                        neuron[layer - 1][j].sigma = sigma;
                        for (int k = 0; k < ncConfig->inputs; k++) {
                            weights[layer - 1][j][k] += (-ncConfig->learning_rate * sigma * neuron[layer - 2][k].netoutput);
                        }
                        j++;
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
double tanh_deriv(double x) { return (pow(1 - tanh(x), 2)); }

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
