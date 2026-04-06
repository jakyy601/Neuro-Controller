/**
 * @file neural_controller.h
 * @author Jakob Schatzl
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

#define LOAD_WEIGHTS false

typedef struct neuron {
    double netinput;
    double netoutput;
    double bias;
    double sigma;
} neuron_st;

typedef struct arch {
    int *topology;
    int total_neurons;
    int total_weights;
} arch_st;

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
    arch_st arch;
} neuralControllerConfig_st;

typedef struct control {
    double act_old;
    double act_new;
    double rating;
    double *input;
    double *input_old;
} control_st;

typedef struct input {
    double value;
    _Bool available;
} input_st;

int neuralController_Init(neuralControllerConfig_st* ncConfig, control_st *control, float (*fctPtr)(), double**** pWeight, neuron_st*** pNeuron);
int neuralController_Run(neuralControllerConfig_st* ncConfig, control_st *control, double* pOutput, float* pInput, double*** weight, neuron_st** neuron);
void neuralController_Free(neuralControllerConfig_st* ncConfig, control_st *control, double ***weight, neuron_st **neuron);
void saveArrayToFile(const char *filename);
double dTanh(double x);
double sigmoid(double x);
double dSigmoid(double x);

#endif  // neuralController
