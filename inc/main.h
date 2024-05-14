#ifndef main_h
#define main_h

#include "minIni.h"
#include "pbPlots.h"
#include "pthread.h"
#include "supportLib.h"

typedef struct i2 {
    float yn;
    float yn2;
} i2_st;

void* feedInput(void* arg);
float generateRandomInt(void);
void pt2(double control_signal, double* x, double* x_dot);
void pt1_path(float u);
float i_path(float yn, float u);

static const char inifile[] = "F:/work/Neuro-Controller/cfg/config.ini";

#endif  // main_h