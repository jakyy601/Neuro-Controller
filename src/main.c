#include <neural_controller.h>

int main(void)
{
  // create neurons

  double netinput[LAYERS + 1][NEURONS] = {0.0};
  double netoutput[LAYERS][NEURONS] = {0.0};
  double bias[LAYERS][NEURONS] = {0.0};
  // TODO: Statt mehreren Arrays, 1 verwenden und nicht genutzten Platz auf
  //  0 stellen -> einfacher zum Lesen und einfacherer Algo
#if NEURONS > INPUTS
  double weights[LAYERS + 1][NEURONS][NEURONS] = {0.0};
#else
  double weights[LAYERS + 1][INPUTS][NEURONS] = {0.0};
#endif
  // double weights_in[INPUTS][NEURONS] = {0.0};
  // double weights_hidden[LAYERS - 1][NEURONS][NEURONS] = {0.0};
  // double weights_out[OUTPUT_LAYER_NEURONS][NEURONS] = {0.0};
  double input[INPUTS] = {0.3, 0.2, 0.5};
  double layer_sigma[LAYERS + 1][NEURONS] = {0.0};
  // TODO: Evtl. dasselbe wie bei den weights
  double output[OUTPUT_LAYER_NEURONS] = {0.0};
  double target = 0.9;

  /*Initialize bias and weights*/
  srand(time(NULL));
  for (int i = 0; i < LAYERS; i++)
  {
    for (int j = 0; j < NEURONS; j++)
    {
      if (i == 0)
      {
        for (int k = 0; k < INPUTS; k++)
        {
          weights[i][j][k] = (double)rand() / RAND_MAX;
        }
      }
      else if (i > 0 && i < LAYERS - 1)
      {
        for (int k = 0; k < NEURONS; k++)
        {
          weights[i][j][k] = (double)rand() / RAND_MAX;
        }
      }
      else if (i == LAYERS)
        bias[i][j] = (double)rand() / RAND_MAX;
    }
  }

  for (int i = 0; i < LAYERS + 1; i++)
  {
    for (int j = 0; j < NEURONS; j++)
    {
      for (int k = 0; k < NEURONS; k++)
      {
        weights[i][j][k] = (double)rand() / RAND_MAX;
      }
    }
  }

  // for (int i = 0; i < OUTPUT_LAYER_NEURONS; i++)
  // {
  //   for (int j = 0; j < NEURONS; j++)
  //   {
  //     weights_out[i][j] = (double)rand() / RAND_MAX;
  //   }
  // }

  /*Feed Forward*/
  int epoch = 0;
  while (epoch < MAX_EPOCHS)
  {
    epoch += 1;
    for (int i = 0; i < LAYERS + 1; i++)
    {
      for (int j = 0; j < NEURONS; j++)
      {
        if (i == 0)
        {
          double sum = 0.0;
          for (int k = 0; k < INPUTS; k++)
          {
            sum += input[k] * weights[i][j][k];
          }
          netinput[i][j] = sum;
          netoutput[i][j] = tanh(netinput[i][j]);
          continue;
        }
        else if ((i > 0) && (i < LAYERS))
        {
          double sum = 0.0;
          for (int k = 0; k < NEURONS; k++)
          {
            sum += netinput[i - 1][j] * weights[i - 1][j][k];
          }
          netinput[i][j] = sum;
          netoutput[i][j] = tanh(netinput[i][j]);
        }
        else
        {
          double sum = 0.0;
          for (int k = 0; k < NEURONS; k++)
          {
            sum += netinput[i - 1][j] * weights[i][j][k];
          }
          netinput[i][j] = sum;
          // tanh zwischen -1 und +1
          output[0] = sum;
        }
      }
    }

    // Backpropagation
    for (int i = LAYERS + 1; i >= 0; i--)
    {
      double sigma = 0.0;
      if (i == LAYERS + 1)
      {
        int j = 0;
        while (j < OUTPUT_LAYER_NEURONS)
        {
          sigma = (tanh_deriv(netinput[i][j])) * (output[j] - target);
          layer_sigma[i][j] = sigma;
          for (int k = 0; k < NEURONS; k++)
          {
            weights[i][j][k] += (-LEARNING_RATE * sigma * netoutput[i][k]);
          }
          j++;
        }
      }
      else if (i == LAYERS)
      {
        int j = 0;
        while (j < NEURONS)
        {
          double sum = 0.0;
          for (int k = 0; k < OUTPUT_LAYER_NEURONS; k++)
          {
            sum += layer_sigma[i + 1][k] * weights[i + 1][j][k];
          }
          sigma = (tanh_deriv(netinput[i][j])) * sum;
          layer_sigma[i][j] = sigma;
          for (int k = 0; k < NEURONS; k++)
          {
            weights[i][j][k] += (-LEARNING_RATE * sigma * netoutput[i][k]);
          }
          j++;
        }
      }
      else
      {
        int j = 0;
        while (j < NEURONS)
        {
          double sum = 0.0;
          for (int k = 0; k < NEURONS; k++)
          {
            sum += layer_sigma[i + 1][j] * weights[i + 1][j][k];
          }
          sigma = (tanh_deriv(netinput[i][j])) * sum;
          layer_sigma[i][j] = sigma;
          for (int k = 0; k < NEURONS; k++)
          {
            weights[i][j][k] += (-LEARNING_RATE * sigma * netoutput[i][k]);
          }
          j++;
        }
      }
    }
  }

  printf("Target: %f\nOutput: %f\n", target, output[0]);

  return 0;
}