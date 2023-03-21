#ifndef ncHelper
#define ncHelper

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct i2_path_s {
    double yn_1;
    double yn_12;
};

void shuffle(int *array, size_t n);
double sigmoid(double x);
double dSigmoid(double x);
struct i2_path_s i2_path(double K, double T, double yn, double yn2, double uk);
double tanh_deriv(double x);
double pt1_path(double Kp, double T1, double deltaT, double uk, double y);
double i_path(double Ki, double T, double y, double uk);

#endif