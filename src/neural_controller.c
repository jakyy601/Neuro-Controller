#include <neural_controller.h>

struct neurons *create_neurons(int layers, int neurons) {
  struct neurons *created_neurons;
  srand(time(NULL));
  created_neurons->netinput =
      (double *)malloc(layers * neurons * sizeof(double));
  created_neurons->netoutput =
      (double *)malloc(layers * neurons * sizeof(double));
  created_neurons->bias = (double *)malloc(layers * neurons * sizeof(double));
  for (int i = 0; i < layers * neurons; i++) {
    *(created_neurons->bias + i) = (double)rand() / RAND_MAX;
    *(created_neurons->netinput + i) = 0.0;
    *(created_neurons->netoutput + i) = 0.0;
  }

  return created_neurons;
}

double *create_weights(int layers, int neurons, int output_layer_neurons,
                       int inputs) {
  srand(time(NULL));
  int number_of_weights = (inputs * neurons) + (layers * neurons) +
                          (output_layer_neurons * neurons);
  double *weights = (double *)malloc(number_of_weights * sizeof(double));

  for (int i = 0; i < number_of_weights; i++) {
    *(weights + i) = (double)rand() / RAND_MAX;
  }

  return weights;
}

int feed_forward(double netinput, double netoutput, double *input,
                 double *weights, double *bias) {
  double sum = 0.0;
  size_t n = sizeof(input) / sizeof(input[0]);
  for (int k = 0; k < n; k++) {
    sum += input[k] * weights[k];
  }
  netinput = sum + *bias;
  netoutput = tanh(netinput);

  return 0;
}

double tanh_deriv(double x) { return (pow(1 - tanh(x), 2)); }
