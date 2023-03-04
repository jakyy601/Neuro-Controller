/**
 * @file ncHelper.c
 * @author Jakob Schatzl
 * @brief Helper file for neural_controller.c
 * @version 0.1
 * @date 2023-02-25
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "ncHelper.h"

/**
 * @brief C function for the hyberbolic tangent
 *
 * @param x x value for the dervative of the hyberbolic tangent
 * @return y value for the dervative of the hyberbolic tangent
 */
double tanh_deriv(double x) {
    double th = tanh(x);
    return 1.0 - th * th;
}

/**
 * @brief Shuffle function
 *
 * Only effective if N is much smaller than RAND_MAX;
 * if this may not be the case, use a better random
 * number generator.
 */
void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

/**
 * @brief Sigmoid function
 *
 * @param x x value
 * @return y value
 */
double sigmoid(double x) { return 1 / (1 + exp(-x)); }

/**
 * @brief Derivative of the sigmoid function
 *
 *
 * @param x x value
 * @return y value
 */
double dSigmoid(double x) { return x * (1 - x); }

/**
 * @brief Double I path
 *
 * @param Ki K value for I element
 * @param T time constant
 * @param yn output of first I element
 * @param yn2 output of second I element
 * @param uk input for double I element
 *
 */
struct i2_path_s i2_path(double Ki, double T, double yn, double yn2, double uk) {
    struct i2_path_s i2;

    i2.yn_12 = yn2 + Ki * uk * T;
    i2.yn_1 = yn + Ki * i2.yn_12 * T;
    return i2;
}

/**
 * @brief Function to calculate the next I-Element value
 *
 * @param Ki integration constant
 * @param T time constant
 * @param y current I value
 * @param uk step value
 * @return next I value
 */
double i_path(double Ki, double T, double y, double uk) {
    return y + Ki * uk * T;
}

/**
 * @brief Function to calculate next PT1 value
 *
 * @param Kp gain cosntant
 * @param T1 time constant
 * @param deltaT time increment
 * @param uk step value
 * @param y current PT1 value
 * @return next PT1 value
 */
double pt1_path(double Kp, double T1, double deltaT, double uk, double y) {
    double ykp1 = y + (Kp * uk - y) * (deltaT / T1);
    return ykp1;
}