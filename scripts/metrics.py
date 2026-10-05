import numpy as np
import pandas as pd
from matplotlib import colormaps
from neuralControllerClass import NeuralController
from system import I, PT2

def calculate_metrics(
    data,
    setpoint,
    settling_tolerance=0.02,
    settling_duration=0.0,
):
    """
    Calculate tracking metrics for one run.

    Expected columns:
        data[:, 0] = time
        data[:, 1] = plant output y
        data[:, 2] = controller output u
    """

    t = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1], dtype=float)
    u = np.asarray(data[:, 2], dtype=float)

    if not (len(t) == len(y) == len(u)):
        raise ValueError("t, y and u must have equal lengths")

    if len(t) < 2:
        raise ValueError("At least two samples are required")

    if np.any(np.diff(t) <= 0):
        raise ValueError("Time values must be strictly increasing")

    reference = np.full_like(y, setpoint)

    # Tracking error
    error = reference - y

    # Integral error metrics
    ise = np.trapezoid(error**2, t)
    iae = np.trapezoid(np.abs(error), t)

    # Final error
    final_error = abs(error[-1])

    # Mean absolute error over the final 10% of the run
    tail_start = int(0.9 * len(error))
    final_mean_abs_error = np.mean(np.abs(error[tail_start:]))

    # Overshoot for a positive setpoint
    overshoot_amount = max(0.0, np.max(y) - setpoint)

    if abs(setpoint) > 1e-12:
        overshoot_percent = (
            100.0 * overshoot_amount / abs(setpoint)
        )
    else:
        overshoot_percent = np.nan

    # Settling band
    band = settling_tolerance * max(abs(setpoint), 1e-12)
    inside_band = np.abs(y - setpoint) <= band

    settling_time = np.inf

    for index in range(len(t)):
        if not inside_band[index]:
            continue

        end_time = t[index] + settling_duration
        end_index = np.searchsorted(t, end_time)

        if end_index >= len(t):
            end_index = len(t) - 1

        if np.all(inside_band[index:end_index + 1]):
            settling_time = t[index]
            break

    # Control effort metrics
    control_energy = np.trapezoid(u**2, t)
    max_abs_control = np.max(np.abs(u))
    mean_abs_control = np.mean(np.abs(u))

    return {
        "ise": ise,
        "iae": iae,
        "final_error": final_error,
        "final_mean_abs_error": final_mean_abs_error,
        "overshoot_amount": overshoot_amount,
        "overshoot_percent": overshoot_percent,
        "settling_time": settling_time,
        "control_energy": control_energy,
        "max_abs_control": max_abs_control,
        "mean_abs_control": mean_abs_control,
    }


def add_setpoint_column(data, setpoint):
    t = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]

    r = np.full_like(t, setpoint)

    return np.column_stack((t, r, y, u))

def min_max_normalize(series):
    minimum = series.min()
    maximum = series.max()

    if np.isclose(minimum, maximum):
        return pd.Series(0.0, index=series.index)

    return (series - minimum) / (maximum - minimum)


def min_max_normalize(series):
    minimum = series.min()
    maximum = series.max()

    if np.isclose(minimum, maximum):
        return pd.Series(0.0, index=series.index)

    return (series - minimum) / (maximum - minimum)

def calculate_average_metrics(metrics_df):
    """
    Average the metrics over the repetitions for every p2 value.
    """

    average_metrics = (
        metrics_df
        .groupby("p2", as_index=False)
        .agg({
            "ise": ["mean", "std"],
            "iae": ["mean", "std"],
            "final_error": ["mean", "std"],
            "final_mean_abs_error": ["mean", "std"],
            "overshoot_percent": ["mean", "std"],
            "settling_time": ["mean", "std"],
            "control_energy": ["mean", "std"],
            "max_abs_control": ["mean", "std"],
        })
    )

    # Flatten MultiIndex column names
    average_metrics.columns = [
        "_".join(column).strip("_")
        for column in average_metrics.columns
    ]

    return average_metrics

def run_experiments(
    p2_values,
    repetitions,
    test_parameters,
    controller_type=3,
    simulation_steps=6000,
    initial_learning_rate=0.01,
    setpoint=1.0,
    option=True,
):
    """
    Execute all experiments and return plot data and metric results.
    """

    colors = colormaps["viridis"](
        np.linspace(0.1, 0.9, len(p2_values))
    )

    experiment_data = []
    metric_results = []

    for p2, color in zip(p2_values, colors):
        runs = []

        for _ in range(repetitions):
            data = test_parameters(
                hidden_layers=2,
                neurons=p2,
                max_epochs=simulation_steps,
                learning_rate=initial_learning_rate,
                setpoint=setpoint,
                isJordan=option,
            )
            runs.append(data)

        experiment_data.append({
            "p2": p2,
            "color": color,
            "runs": runs,
        })

        # Calculate metrics for every individual repetition
        for repetition, run in enumerate(runs, start=1):
            metrics = calculate_metrics(
                run,
                setpoint=setpoint,
            )

            metric_results.append({
                "p2": p2,
                "repetition": repetition,
                **metrics,
            })

    metrics_df = pd.DataFrame(metric_results)

    return experiment_data, metrics_df

def testParameters(hidden_layers: int, neurons: int, max_epochs: int, learning_rate: float, setpoint: float, isJordan: bool):
    dt = 0.001
    t = np.arange(0.0, max_epochs * dt, dt) # array for time
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    i_p = PT2(K=1.0, T=0.1, D=0.3, dt=dt)
    # pt2 = PT2(K=1.0, T=1.0, D=0.3, dt=dt)
    setpoint_rel = setpoint

    # A class is needed to run the neural controller module as the python refcount garbage collector
    # looses the c pointer and therefore just frees the memory.
    neuralController = NeuralController(hidden_layers, neurons, learning_rate, setpoint, isJordan)
    for i in range(max_epochs-1):
        # learning_rate_rel = lr.step_decay_lr(i, learning_rate, 1000, 0.5)
        # learning_rate_rel = lr.linear_decay_lr(i, learning_rate, 0.0001, max_epochs)
        learning_rate_rel = learning_rate
        u[i+1] = neuralController.run(y[i], learning_rate_rel, setpoint_rel)
        y[i+1] = i_p.step(u[i+1])
        if (i%1000) == 0:
            print(f"Epoch: {i} Plant output: {y[i]} u: {u[i]} Error: {neuralController.ncConfig.setpoint - y[i]} Learning Rate: {learning_rate_rel}")
        # if ((i%(max_epochs/2)) == 0) and (i > 0):
        #     setpoint_rel = 0.5

    data = np.column_stack((t,y,u))
    del neuralController
    return data