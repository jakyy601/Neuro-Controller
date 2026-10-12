/**
 * @file neural_controller.h
 * @author Jakob Schatzl
 * @brief Public header for the activation function implementatitions
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026
 */
#ifndef ACTIVATION_H
#define ACTIVATION_H

/**************************************************************************************************
                             Section for public type definitions
***************************************************************************************************/
/**
 * @var activation_fn
 * @brief Function pointer used for activation function and derivative
 */
typedef float (*activation_fn) (float x);

/**
 * @enum activation_st
 * @brief Enumeration for the activation functions used in neuron_st
 * @var tanh Hyperbolic tangent, zero-centered, gradial output from -1 to 1 with diminishing gradient on higher values
 * @var sigmoid Sigmoid function, centerd at 0.5, gradial output from 0 to 1 with diminishing gradient on higher values
 * @var leaky_relu, negative input leads to very small negative output, positive values leads to linear output, very small gradient on negative values
 * @var linear, linear input equals linear output
 */
typedef struct activation {
    activation_fn forward;
    activation_fn backward;
} activation_st;

typedef enum activation_type{
    ACTIVATION_SIGMOID   = 0,
    ACTIVATION_TANH      = 1,
    ACTIVATION_LEAKY_RELU= 2,
    ACTIVATION_LINEAR    = 3
} activation_type_en;

/**************************************************************************************************
                             Section for public instances
***************************************************************************************************/
extern const activation_st ACT_SIGMOID;
extern const activation_st ACT_TANH;
extern const activation_st ACT_LEAKY_RELU;
extern const activation_st ACT_LINEAR;

/**************************************************************************************************
                        Section for public function prototypes
***************************************************************************************************/
float activation_forward(activation_type_en id, float x);
float activation_backward(activation_type_en id, float x);

#endif //ACTIVATION_H