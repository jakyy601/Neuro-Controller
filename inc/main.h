#ifndef main_h
#define main_h

#include <errno.h>
#include <time.h>
//#include <zmq.h>

#include "neuralController.h"
#include "activation.h"

#define PlotGraph false
#define WriteToFile false

typedef struct i2 {
    double yn;
    double yn2;
} i2_st;

double generateRandomInt(void);
double i_plant(double yn, double u);
double round_to_decimal_places(double value, int decimal_places);

static const char inifile[] = "F:/work/Neuro-Controller/cfg/config.ini";

#endif  // main_h