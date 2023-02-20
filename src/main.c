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
#include <neural_controller.h>

int main(void) {
    struct neuralControllerConfig ncConfig;

    FILE* pConfig = fopen("config.json", "r");
    // cJSON* config_json = cJSON_Parse;

    ncConfig.inputs = 2;
    ncConfig.layers = 1;
    ncConfig.learning_rate = 0.1;
    ncConfig.max_epochs = 10000;
    ncConfig.neurons = 2;
    ncConfig.output_layer_neurons = 1;

    fclose(pConfig);

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
    WriteToFile(pngData, length, "plot.png");
    DeleteImage(imageRef->image);
    FreeAllocations();

    return 42;
}