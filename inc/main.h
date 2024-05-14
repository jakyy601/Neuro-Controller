#ifndef main_h
#define main_h

#include "minIni.h"
#include "pbPlots.h"
#include "pthread.h"
#include "supportLib.h"

void* feedInput(void* arg);
float generateRandomInt(void);
void i2_path(float u);
void pt1_path(float u);

static const char inifile[] = "F:/work/Neuro-Controller/cfg/config.ini";

typedef struct i2 {
    float yn_1;
    float yn_12;
} i2_st;

#endif  // main_h