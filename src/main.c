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
    ncConfig.layers = ini_getl("Neural Network", "layers", -1, inifile);
    ncConfig.learning_rate = round(ini_getf("Neural Network", "learning_rate", -1, inifile) * 100) / 100;
    ncConfig.max_epochs = ini_getl("Neural Network", "max_epochs", -1, inifile);
    ncConfig.neurons = ini_getl("Neural Network", "neurons", -1, inifile);
    ncConfig.output_layer_neurons = ini_getl("Neural Network", "output_layer_neurons", -1, inifile);

    double output[1] = {0.0};
    double* error = malloc(ncConfig.max_epochs * sizeof(double));
    double x[ncConfig.max_epochs];
    double y[ncConfig.max_epochs];
    learn_loop(&ncConfig, output, error);

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

    return 42;
}