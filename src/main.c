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
#include "neuralController.h"

int main(void) {
    struct neuralControllerConfig ncConfig;

    ncConfig.inputs = ini_getl("Neural Network", "inputs", -1, inifile);
    ncConfig.hidden_layers = ini_getl("Neural Network", "hidden_layers", -1, inifile);
    ncConfig.layers = ncConfig.hidden_layers + 2;
    ncConfig.learning_rate = round(ini_getf("Neural Network", "learning_rate", -1, inifile) * 100) / 100;
    ncConfig.max_epochs = ini_getl("Neural Network", "max_epochs", -1, inifile);
    ncConfig.neurons = ini_getl("Neural Network", "neurons", -1, inifile);
    ncConfig.output_layer_neurons = ini_getl("Neural Network", "output_layer_neurons", -1, inifile);

    int layer_sizes[ncConfig.layers];
    for (int i = 0; i < ncConfig.layers; i++) {
        if (i == 0)
            layer_sizes[i] = ncConfig.inputs;
        else if (i == ncConfig.layers - 1)
            layer_sizes[i] = ncConfig.output_layer_neurons;
        else
            layer_sizes[i] = ncConfig.neurons;
    }

    double* error = malloc(ncConfig.max_epochs * sizeof(double));
    double x[ncConfig.max_epochs];
    double y[ncConfig.max_epochs];
    learn_loop(&ncConfig, error, layer_sizes);

    for (int i = 0; i < ncConfig.max_epochs; i++) {
        x[i] = (float)i;
        y[i] = *(error + i);
    }

    RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();

    DrawScatterPlot(imageRef, 600, 400, x, ncConfig.max_epochs, y, ncConfig.max_epochs, NULL);

    size_t length;
    double* pngData = ConvertToPNG(&length, imageRef->image);
    WriteToFile(pngData, length, "control.png");
    DeleteImage(imageRef->image);
    FreeAllocations();

    free(error);

    return 42;
}