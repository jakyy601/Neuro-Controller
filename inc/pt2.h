#ifndef PT2_H
#define PT2_H

typedef struct PT2_Handle {
    double K;      // Static gain
    double T;      // Time constant [s]
    double D;      // Damping ratio
    double Ts;     // Simulation/control sampling time [s]

    double y;      // Plant output
    double dy;     // First derivative of y
} PT2_Handle_t;

void PT2_Init(PT2_Handle_t *h, double Ts);
void PT2_Reset(PT2_Handle_t *h, double y0, double dy0);
double PT2_Update(PT2_Handle_t *h, double u);

#endif /* PT2_H */