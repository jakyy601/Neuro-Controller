/**
 * @file neural_controller.h
 * @author Jakob Schatzl (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-01-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUTS_BT_NEURONS 0

struct neuron {
    double netinput;
    double netoutput;
    double bias;
    double sigma;
};

struct neuralControllerConfig {
    int layers;
    int neurons;
    int output_layer_neurons;
    int inputs;
    int max_epochs;
    double learning_rate;
};

struct i2_path_s {
    double yn_1;
    double yn_12;
};

double tanh_deriv(double x);
int learn_loop(struct neuralControllerConfig *ncConfig, double *ncOutput);
struct i2_path_s i2_path(double K, double T, double yn, double yn2, double uk);
