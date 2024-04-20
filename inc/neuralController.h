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
#ifndef neuralController
#define neuralController

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ncHelper.h"

#define INPUTS_BIGGER_THAN_NEURONS 0

struct neuron {
    double netinput;
    double netoutput;
    double bias;
    double sigma;
};

struct neuralControllerConfig {
    int hidden_layers;
    int layers;
    int neurons;
    int output_layer_neurons;
    int inputs;
    int max_epochs;
    double learning_rate;
    double setpoint;
};

typedef struct input {
    double value;
    _Bool available;
} input_st;

static const char inifile[] = "F:/work/Neuro-Controller/cfg/config.ini";

int learn_loop(struct neuralControllerConfig* ncConfig, double* pError_array, input_st* pInput);

#endif  // neuralController