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
input_st input_s = (input_st){0};

void* feedInput(void* arg) {
    FILE* fptr = fopen("pt1.txt", "r");
    char input_buffer[50] = {0};
    int ret = 0;
    char* pEnd = NULL;
    while (1) {
        if (input_s.available == false) {
            ret = fscanf(fptr, "%s", &input_buffer);
            if (ret == EOF) {
                break;
            }
            pthread_mutex_lock(&mutex);
            input_s.value = (double)strtof(input_buffer, &pEnd);
            input_s.available = true;
            pthread_mutex_unlock(&mutex);
        }
    }
}

int main(int argc, const char* argv[]) {
    struct neuralControllerConfig ncConfig;
    long job = 0;

    ncConfig.inputs = ini_getl("Neural Network", "inputs", -1, inifile);
    ncConfig.hidden_layers = ini_getl("Neural Network", "hidden_layers", -1, inifile);
    ncConfig.layers = ncConfig.hidden_layers + 2;
    ncConfig.learning_rate = round(ini_getf("Neural Network", "learning_rate", -1, inifile) * 100) / 100;
    ncConfig.infinite = ini_getl("Neural Network", "infinite", -1, inifile);
    if (ncConfig.infinite == 1)
        ncConfig.max_epochs = -1;
    else
        ncConfig.max_epochs = ini_getl("Neural Network", "max_epochs", -1, inifile);
    ncConfig.one_shot = ini_getl("Neural Network", "one_shot", -1, inifile);
    if (ncConfig.one_shot == 1)
        ncConfig.max_epochs = 1;
    ncConfig.neurons = ini_getl("Neural Network", "neurons", -1, inifile);
    ncConfig.output_layer_neurons = ini_getl("Neural Network", "output_layer_neurons", -1, inifile);
    ncConfig.setpoint = round(ini_getf("Neural Network", "setpoint", -1, inifile) * 100) / 100;
    ncConfig.initialized = 0;

    double* error_array = calloc(ncConfig.max_epochs, sizeof(double));
    double x[ncConfig.max_epochs];
    double y[ncConfig.max_epochs];
    // input_st input[ncConfig.inputs];
    input_s.available = false;

    pthread_mutex_init(&mutex, NULL);
    pthread_create(&feedInputThread, NULL, feedInput, (void*)(&job));

    for (int i = 0; i < 500; i++) {
        learn_loop(&ncConfig, error_array, &input_s, time(NULL));
    }

    // for (int i = 0; i < ncConfig.max_epochs; i++) {
    //     x[i] = (float)i;
    //     y[i] = *(error_array + i);
    // }

    // RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();

    // DrawScatterPlot(imageRef, 600, 400, x, ncConfig.max_epochs, y, ncConfig.max_epochs, NULL);

    // size_t length;
    // double* pngData = ConvertToPNG(&length, imageRef->image);
    // WriteToFile(pngData, length, "control.png");
    // DeleteImage(imageRef->image);
    // FreeAllocations();

    // free(error_array);

    // return 42;
}