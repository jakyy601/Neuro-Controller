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

#include "cJSON.h"
#include "minIni.h"
#include "pbPlots.h"
#include "supportLib.h"

#define INPUTS_BT_NEURONS 0

static const char inifile[] = "F:/work/Neuro-Controller/src/config.ini";

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
int learn_loop(struct neuralControllerConfig *ncConfig, double *ncOutput, double *pError_array);
void shuffle(int *array, size_t n);
double sigmoid(double x);
double dSigmoid(double x);
struct i2_path_s i2_path(double K, double T, double yn, double yn2, double uk);
