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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "minIni.h"
#include "ncHelper.h"
#include "pbPlots.h"
#include "supportLib.h"

#define INPUTS_BT_NEURONS 0

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
};

static const char inifile[] = "F:/work/Neuro-Controller/cfg/config.ini";

int learn_loop(struct neuralControllerConfig *ncConfig, double *pError_array, int *layer_sizes);
void initialize_weights(float ***weights, int *layer_sizes, int num_layers);

#endif