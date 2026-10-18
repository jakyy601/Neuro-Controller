#ifndef PID_H
#define PID_H

typedef struct PID_Handle {
    double kp;
    double ki;
    double kd;
    double ts;

    double out_min;
    double out_max;

    double integral;
    double measurement_prev;
    double output_prev;

    unsigned int initialized;
} PID_Handle_t;

void PID_Init(PID_Handle_t *pid,
              double kp,
              double ki,
              double kd,
              double ts,
              double out_min,
              double out_max);

double PID_Update(PID_Handle_t *pid,
                 double setpoint,
                 double measurement);

void PID_Reset(PID_Handle_t *pid);

#endif /* PID_H */