#include <stdio.h>

#include "ncHelper.h"
#include "pbPlots.h"
#include "supportLib.h"

int main(void) {
    int sampleSize = 50;
    double x[sampleSize];
    double Kp = 1.0;
    double T1 = 1.0;
    double deltaT = 0.1;
    double y[sampleSize];

    memset(y, 0, sampleSize * sizeof(double));
    memset(x, 0, sampleSize * sizeof(double));
    for (int i = 0; i < sampleSize; i++)
        x[i] = (double)i;
    double uk = 1.0;

    for (int i = 0; i < sampleSize - 1; i++) {
        y[i + 1] = (double)pt1_path(Kp, T1, deltaT, uk, y[i]);
    }

    RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();
    DrawScatterPlot(imageRef, 600, 400, x, sampleSize, y, sampleSize, NULL);
    size_t length;
    double* pngData = ConvertToPNG(&length, imageRef->image);
    WriteToFile(pngData, length, "pt1.png");
    DeleteImage(imageRef->image);
    FreeAllocations();

    double Ki = 1.0;
    double T = 0.1;
    double yn[sampleSize];
    double yn2[sampleSize];

    memset(yn, 0, sampleSize * sizeof(double));
    memset(yn2, 0, sampleSize * sizeof(double));

    for (int i = 0; i < sampleSize - 1; i++) {
        yn2[i + 1] = yn2[i] + Ki * uk * T;
        yn[i + 1] = yn[i] + Ki * yn2[i + 1] * T;
    }

    DrawScatterPlot(imageRef, 600, 400, x, sampleSize, yn, sampleSize, NULL);

    pngData = ConvertToPNG(&length, imageRef->image);
    WriteToFile(pngData, length, "i2_element.png");
    DeleteImage(imageRef->image);
    FreeAllocations();

    Ki = 2.0;
    T = 0.5;
    uk = 1.0;

    memset(y, 0, sampleSize * sizeof(double));

    for (int i = 0; i < sampleSize - 1; i++) {
        y[i + 1] = (double)i_path(Ki, T, uk, y[i]);
    }

    DrawScatterPlot(imageRef, 600, 400, x, sampleSize, y, sampleSize, NULL);
    pngData = ConvertToPNG(&length, imageRef->image);
    WriteToFile(pngData, length, "i_element.png");
    DeleteImage(imageRef->image);
    FreeAllocations();

    return 42;
}
