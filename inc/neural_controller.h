#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LAYERS 3
#define NEURONS 10
#define OUTPUT_LAYER_NEURONS 1
#define INPUTS 3
#define MAX_EPOCHS 1000
#define LEARNING_RATE 0.1

struct neurons {
  double *netinput;
  double *netoutput;
  double *bias;
};

struct neurons *create_neurons(int layers, int neurons);
double *create_weights(int layers, int neurons, int output_layer_neurons,
                       int inputs);
int feed_forward(double netinput, double netoutput, double *input,
                 double *weights, double *bias);
double tanh_deriv(double x);