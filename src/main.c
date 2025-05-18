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

pthread_t feedInputThread;
pthread_mutex_t mutex;

/**
 * @brief main loop for testing
 *
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, const char* argv[]) {
    // create the neural controller config
    neuralControllerConfig_st ncConfig;
    float input[INPUTS - 1] = {0};  // input array
    float yn = 0;                   // state of plant
    double output = 0.0;            // neural Network output

    // end pointer for strtol
    char* end;

    // initialize the neural controller config via command line arguments
    ncConfig.inputs = (int)strtol(argv[1], &end, 10);
    ncConfig.hidden_layers = (int)strtol(argv[2], &end, 10);
    ncConfig.layers = ncConfig.hidden_layers + 2;
    ncConfig.learning_rate = (float)roundf(strtof(argv[3], &end) * 100) / 100;
    ncConfig.max_epochs = (int)strtol(argv[4], &end, 10);
    ncConfig.neurons = (int)strtol(argv[5], &end, 10);
    ncConfig.output_layer_neurons = (int)strtol(argv[6], &end, 10);
    ncConfig.setpoint = (float)roundf(strtof(argv[7], &end) * 100) / 100;

    // array for all error values over n epochs
    float* error_array = (float*)calloc(ncConfig.max_epochs, sizeof(float));
    int* x_values = (int*)calloc(ncConfig.max_epochs, sizeof(int));

    // variables for write to file
    char filename[100];
    strncpy(filename, argv[8], sizeof(filename) - 1);
    filename[sizeof(filename) - 1] = '\0';  // Ensure null-termination
    int column = (int)strtol(argv[9], &end, 10);
    if (column <= 0) {
        fprintf(stderr, "Invalid column value: %d\n", column);
        return 1;
    }

    // set seed for rand() function
    srand(time(NULL));

    // function pointer to a function that generates random floats
    float (*randFctPtr)();
    randFctPtr = &generateRandomInt;

    // Initialize the neuralController
    neuralController_Init(&ncConfig, randFctPtr, "test.bin");

    /*Configuration for ZeroMQ as Sender/Server*/
    void *context = zmq_ctx_new();
    void *socket = zmq_socket(context, ZMQ_REP);
    int rc = zmq_bind(socket, "tcp://localhost:5555");
    if(rc != 0){
        printf("Error: %d\n");
        exit(EXIT_FAILURE);
    }
    char rxBuffer[15] = {0};
    char txBuffer[15] = {0};

    // Loop for testing over n epochs
    for (int i = 0; i < ncConfig.max_epochs; i++) {
        rc = zmq_recv(socket, rxBuffer, sizeof(rxBuffer), 0);
        yn = atof(rxBuffer);
        // set input for the next run
        input[0] = yn;
        // Run through feed forward + backpropagation
        neuralController_Run(&ncConfig, &output, input);
        /*Calculate next state of the I plant*/
        //yn = i_plant(yn, output);
        int ret = sprintf(txBuffer, "%f", output);
        rc = zmq_send(socket, txBuffer, ret, 0);
        // save error
        // error_array[i] = (double)ncConfig.setpoint - yn;
        error_array[i] = yn;
        // save epoch
        x_values[i] = i;
        // debug print message
        if ((i % 100) == 0) {
            printf("Epoch: %d Plant output: %f Error: %f u: %f \n", i, yn, ncConfig.setpoint - yn, output);
            fflush(stdout);
        }
    }

#if PlotGraph
    // Create image reference structure
    RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();

    // Plot
    DrawScatterPlot(imageRef, 600, 400, x_values, ncConfig.max_epochs, error_array, ncConfig.max_epochs, NULL);

    // Create PNG file and free allocations
    size_t length;
    double* pngData = ConvertToPNG(&length, imageRef->image);
    WriteToFile(pngData, length, "control.png");
    DeleteImage(imageRef->image);
    FreeAllocations();
#endif  // PlotGraph

#if WriteToFile

    FILE* file = fopen(filename, "r+");  // Open file in read+write mode
    if (file == NULL) {
        perror("Error opening results file");
        return 1;
    }else{
        int i = 0;
        char character;
        while ((character = fgetc(file)) != EOF) {
            if (character == '\n') {
                fprintf(file, ", %f\n", error_array[i]);
                i++;
            }
        }
    }

    fclose(file);

#endif  // WriteToFile

    //Write weights to file
    saveArrayToFile("test.bin");

    // free allocated memory
    free(error_array);
    free(x_values);

    return 42;
}

/**
 * @brief Generates a random float between 0 and 0.1
 *
 *
 * @param
 * @return
 */
float generateRandomInt(void) {
    return (rand() / (double)(RAND_MAX)) / 10;
}

/**
 * @brief Function that calculates the next step of an I plant
 *
 *
 * @param yn current state
 * @param u control variable
 * @return next state
 */
float i_plant(float yn, float u) {
    float K = 1;
    float T = 0.1;

    return yn + K * u * T;
}

double round_to_decimal_places(float value, int decimal_places) {
    double factor = pow(10, decimal_places);
    return round(value * factor) / factor;
}