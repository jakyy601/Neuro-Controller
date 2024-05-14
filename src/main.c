/**
 * @file main.c
 * @author Jakob Schatzl
 * @brief
 * @version 0.1
 * @date 2023-01-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "main.h"

#include "neuralController.h"

#define DT 0.1         // Time step
#define TOTAL_TIME 10  // Total simulation time in seconds

pthread_t feedInputThread;
pthread_mutex_t mutex;
// input_st input_s[2] = (input_st){0};

int main(int argc, const char* argv[]) {
    neuralControllerConfig_st ncConfig;
    long job = 0;
    float input[INPUTS - 1] = {0};
    float yn = 0;

    // System state
    double x = 0;      // Initial output
    double x_dot = 0;  // Initial rate of change of output

    ncConfig.inputs = ini_getl("Neural Network", "inputs", -1, inifile);
    ncConfig.hidden_layers = ini_getl("Neural Network", "hidden_layers", -1, inifile);
    ncConfig.layers = ncConfig.hidden_layers + 2;
    ncConfig.learning_rate = round(ini_getf("Neural Network", "learning_rate", -1, inifile) * 100) / 100;
    ncConfig.max_epochs = ini_getl("Neural Network", "max_epochs", -1, inifile);
    ncConfig.neurons = ini_getl("Neural Network", "neurons", -1, inifile);
    ncConfig.output_layer_neurons = ini_getl("Neural Network", "output_layer_neurons", -1, inifile);
    ncConfig.setpoint = round(ini_getf("Neural Network", "setpoint", -1, inifile) * 100) / 100;
    ncConfig.initialized = 0;

    double* error_array = (double*)calloc(ncConfig.max_epochs, sizeof(double));

    // input_st input[ncConfig.inputs];
    double output = 0.0;
    srand(time(NULL));

    float (*randFctPtr)();
    randFctPtr = &generateRandomInt;

    // pthread_mutex_init(&mutex, NULL);
    // pthread_create(&feedInputThread, NULL, feedInput, (void*)(&job));
    neuralController_Init(&ncConfig, randFctPtr);

    for (int i = 0; i < ncConfig.max_epochs; i++) {
        neuralController_Run(&ncConfig, &output, input);
        pt2(1, &x, &x_dot);
        input[0] = x;
        error_array[i] = (double)ncConfig.setpoint - x;
        if ((i % 10) == 0)
            printf("Epoch: %d Plant output: %f Error: %f u: %f \n", i, x, ncConfig.setpoint - x, output);
    }

    double x_values[ncConfig.max_epochs];
    double y_values[ncConfig.max_epochs];

    for (int i = 0; i < ncConfig.max_epochs; i++) {
        x_values[i] = (float)i;
        y_values[i] = *(error_array + i);
    }

    RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();

    DrawScatterPlot(imageRef, 600, 400, x_values, ncConfig.max_epochs, y_values, ncConfig.max_epochs, NULL);

    size_t length;
    double* pngData = ConvertToPNG(&length, imageRef->image);
    WriteToFile(pngData, length, "control.png");
    DeleteImage(imageRef->image);
    FreeAllocations();

    free(error_array);

    // return 42;
}

float generateRandomInt(void) {
    float low = -1.0;
    float high = 1.0;

    return (rand() / (double)(RAND_MAX)) / 10;
    // return ((rand() / (double)(RAND_MAX)) * fabs(low - high) + low) / 10;
}

// void* feedInput(void* arg) {
//     FILE* fptr = fopen("pt1.txt", "r");
//     char input_buffer[50] = {0};
//     int ret = 0;
//     char* pEnd = NULL;
//     while (1) {
//         if (input_s.available == false) {
//             ret = fscanf(fptr, "%s", &input_buffer);
//             if (ret == EOF) {
//                 break;
//             }
//             pthread_mutex_lock(&mutex);
//             input_s.value = (double)strtof(input_buffer, &pEnd);
//             input_s.available = true;
//             pthread_mutex_unlock(&mutex);
//         }
//     }
// }

float i_path(float yn, float u) {
    float K = 1;
    float T = 0.1;

    return yn + K * u * T;
}

void pt2(double control_signal, double* x, double* x_dot) {
    double x_ddot;
    // PT2 System Parameters
    double K = 1;        // Gain
    double zeta = 0.1;   // Damping ratio
    double omega_n = 2;  // Natural frequency

    // PT2 differential equation (double derivative of output)
    x_ddot = K * (-(2 * zeta * omega_n * (*x_dot) + omega_n * omega_n * (*x)) + control_signal);

    // Update state using Euler's method
    *x_dot += x_ddot * DT;
    *x += *x_dot * DT;
}