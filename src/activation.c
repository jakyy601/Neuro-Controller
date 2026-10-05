/**
 * @file activation.c
 * @author Jakob Schatzl
 * @brief Implementation of activation functions for the neural controller
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026
 *
 */
/**************************************************************************************************
                                    Section for headers
***************************************************************************************************/
#include "activation.h"
#include "neuralController.h"

/**************************************************************************************************
                        Section for public function prototypes
***************************************************************************************************/
/**
 * @brief Sigmoid function
 * @param x x value
 * @return y value
 */
static double sigmoid(double x);

/**
 * @brief Derivative of the sigmoid function
 * @param x x value of the sigmoid function
 * @return y value of the sigmoid function
 */
static double dSigmoid(double x);

/**
 * @brief Derivative of the hyberbolic tangent
 * @param x x value of the dervative of the hyberbolic tangent
 * @return y value of the dervative of the hyberbolic tangent
 */
static double dTanh(double x);

/**
 * @brief Leaky ReLU function
 * @param x x value of the leaky ReLU function
 * @return y value of the leaky ReLU function
 */
static double leakyRelu(double x);

/**
 * @brief Derivative of the leaky ReLU function
 * @param x x value for the dervative of the leaky ReLU function
 * @return y value for the dervative of the leaky ReLU function
 */
static double dLeakyRelu(double x);

/**
 * @brief Linear function
 * @param x x value of the linear function
 * @return y value of the linear function
 */
static double linear(double x);

/**
 * @brief Derivative of the linear function
 * @param x x value for the dervative of the linear function (unused but required for consistency)
 * @return y value for the dervative of the linear function
 */
static double dLinear(double x);

/**************************************************************************************************
                                Section for static functions
***************************************************************************************************/
static double sigmoid(double x) {
    return 1 / (1 + exp(-x)); 
}

static double dSigmoid(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

static double dTanh(double x) {
    double th = tanh(x);
    return 1.0 - th * th;
}

static double leakyRelu(double x) {
    static const double alpha = 0.01;
    return x > 0.0 ? x : alpha * x;
}

static double dLeakyRelu(double x) {
    static const double alpha = 0.01;
    return x > 0.0 ? 1.0 : alpha;
}

static double linear(double x) {
    return x;
}

static double dLinear(double x) {
    /* For consistency only, as the function pointer will still need a parameter -> tell the compiler x is not used */
    (void)x;
    return 1.0;
}

/**************************************************************************************************
                                Section for public functions
***************************************************************************************************/


/**************************************************************************************************
                                    Constant instances
***************************************************************************************************/
const activation_st ACT_SIGMOID = {
    .forward = sigmoid,
    .backward = dSigmoid,
};

const activation_st ACT_TANH = {
    .forward = tanh,
    .backward = dTanh,
};

const activation_st ACT_LEAKY_RELU = {
    .forward = leakyRelu,
    .backward = dLeakyRelu,
};

const activation_st ACT_LINEAR = {
    .forward = linear,
    .backward = dLinear,
};