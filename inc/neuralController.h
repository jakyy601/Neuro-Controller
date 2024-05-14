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

#define INPUTS_BIGGER_THAN_NEURONS 0

#define INPUTS 2
#define HIDDEN_LAYERS 3
#define LAYERS HIDDEN_LAYERS + 2
#define INFINITE 0
#define MAX_EPOCHS 6000
#define NEURONS 10
#define OUTPUT_LAYER_NEURONS 1

typedef struct neuron {
    double netinput;
    double netoutput;
    double bias;
    double sigma;
} neuron_st;

typedef struct neuralControllerConfig {
    int hidden_layers;
    int layers;
    int neurons;
    int output_layer_neurons;
    int inputs;
    int max_epochs;
    int initialized;
    double learning_rate;
    double setpoint;
} neuralControllerConfig_st;

typedef struct input {
    double value;
    _Bool available;
} input_st;

int neuralController_Init(neuralControllerConfig_st* ncConfig, float (*fctPtr)());
int neuralController_Run(neuralControllerConfig_st* ncConfig, double* pOutput, float* pInput);
double dTanh(double x);
double sigmoid(double x);
double dSigmoid(double x);

#endif  // neuralController
