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

pthread_t feedInputThread;
pthread_mutex_t mutex;
// input_st input_s[2] = (input_st){0};

i2_st i2_path_st;
float K = 1;
float T = 0.1;
float yn1 = 0;
float yn2 = 0;

int main(int argc, const char* argv[]) {
    neuralControllerConfig_st ncConfig;
    long job = 0;
    float input[INPUTS - 1] = {0};

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
    double x[ncConfig.max_epochs];
    double y[ncConfig.max_epochs];
    // input_st input[ncConfig.inputs];
    double output = 0.0;
    srand(time(NULL));

    float (*randFctPtr)();
    randFctPtr = &generateRandomInt;

    // pthread_mutex_init(&mutex, NULL);
    // pthread_create(&feedInputThread, NULL, feedInput, (void*)(&job));
    neuralController_Init(&ncConfig, randFctPtr);

    for (int i = 0; i < ncConfig.max_epochs; i++) {
        pt1_path(output);
        // i2_path(output);
        input[0] = yn1;
        neuralController_Run(&ncConfig, &output, input);
        error_array[i] = (double)ncConfig.setpoint - yn1;
        if ((i % 10) == 0)
            printf("Epoch: %d Plant output: %f Error: %f u: %f \n", i, yn1, ncConfig.setpoint - yn1, output);
    }

    for (int i = 0; i < ncConfig.max_epochs; i++) {
        x[i] = (float)i;
        y[i] = *(error_array + i);
    }

    RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();

    DrawScatterPlot(imageRef, 600, 400, x, ncConfig.max_epochs, y, ncConfig.max_epochs, NULL);

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

    return (rand() / (double)(RAND_MAX)) * fabs(low - high) + low;
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

// void i2_path(float u) {
//     float K = 1;
//     float T = 0.1;

//     yn2 = yn2 + K * u * T;
//     yn1 = yn1 + K * yn2 * T;
// }

void pt1_path(float u) {
    float Kp = 1;
    float T1 = 1.0;
    float deltaT = 0.1;

    yn1 = yn1 + (Kp * u - yn1) * (deltaT / T1);
}