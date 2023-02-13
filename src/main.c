/**
 * @file main.c
 * @author Jakob Schatzl
 * @brief
 * @version 0.1
 * @date 2023-01-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <neural_controller.h>

int main(void) {
    struct neuralControllerConfig ncConfig;

    ncConfig.inputs = 2;
    ncConfig.layers = 2;
    ncConfig.learning_rate = 0.1;
    ncConfig.max_epochs = 10000;
    ncConfig.neurons = 2;
    ncConfig.output_layer_neurons = 1;

    double output[1] = {0.0};

    int ret = learn_loop(&ncConfig, output);

    // printf("Target: %f\nOutput: %f\n", target, output[0]);

    return 42;
}