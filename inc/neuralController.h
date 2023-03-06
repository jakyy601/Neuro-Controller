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
    int layers;
    int neurons;
    int output_layer_neurons;
    int inputs;
    int max_epochs;
    double learning_rate;
    int layer_sizes[];
};

static const char inifile[] = "F:/work/Neuro-Controller/cfg/config.ini";

int learn_loop(struct neuralControllerConfig *ncConfig, double *ncOutput, double *pError_array, int *layer_sizes);
void initialize_weights(float ***weights, int *layer_sizes, int num_layers);
void backpropagation(double ***weights, struct neuron **neuron, int layer, int neurons_current, int neurons_next, double learning_rate);

#endif