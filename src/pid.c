#include "pid.h"

void PID_Init(PID_Handle_t *pid,
              double kp,
              double ki,
              double kd,
              double ts,
              double out_min,
              double out_max)
{
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
    pid->ts = ts;

    pid->out_min = out_min;
    pid->out_max = out_max;

    pid->integral = 0.0f;
    pid->measurement_prev = 0.0f;
    pid->output_prev = 0.0f;
    pid->initialized = 0u;
}

void PID_Reset(PID_Handle_t *pid)
{
    pid->integral = 0.0f;
    pid->measurement_prev = 0.0f;
    pid->output_prev = 0.0f;
    pid->initialized = 0u;
}

double PID_Update(PID_Handle_t *pid,
                 double setpoint,
                 double measurement)
{
    double error = setpoint - measurement;

    /* Derivative on measurement avoids derivative kick on setpoint steps. */
    double derivative = 0.0f;

    if (pid->initialized != 0u) {
        derivative = (pid->measurement_prev - measurement) / pid->ts;
    } else {
        pid->measurement_prev = measurement;
        pid->initialized = 1u;
    }

    /* Tentative integral update. */
    double integral_new =
        pid->integral + pid->ki * pid->ts * error;

    /* Unsaturated PID output. */
    double output_unsat =
        pid->kp * error +
        integral_new +
        pid->kd * derivative;

    /* Saturate plant/DAC command. */
    double output_sat = output_unsat;

    if (output_sat < pid->out_min) {
        output_sat = pid->out_min;
    } else if (output_sat > pid->out_max) {
        output_sat = pid->out_max;
    }

    /*
     * Conditional anti-windup:
     * integrate only when doing so does not push an already-saturated
     * output farther into saturation.
     */
    if ((output_unsat == output_sat) ||
        ((error > 0.0f) && (output_sat < pid->out_max)) ||
        ((error < 0.0f) && (output_sat > pid->out_min))) {
        pid->integral = integral_new;
    }

    pid->measurement_prev = measurement;
    pid->output_prev = output_sat;

    return output_sat;
}