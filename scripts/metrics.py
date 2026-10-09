import numpy as np
import pandas as pd
import learnrate as lr
from matplotlib import colormaps
from neuralControllerClass import NeuralController, activation_type_en
from system import I, PT2

def calculate_metrics(
    data,
    setpoint,
    settling_tolerance=0.02,
):
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise TypeError(
            "calculate_metrics() expects one 2-D NumPy array "
            "with columns [t, y, u]. "
            f"Received type={type(data).__name__}, "
            f"shape={getattr(data, 'shape', None)}"
        )

    if data.shape[1] < 2:
        raise ValueError(
            "data must contain at least columns [t, y]"
        )

    t = data[:, 0]
    y = data[:, 1]

    if len(t) < 2:
        raise ValueError(
            "At least two samples are required"
        )

    if np.any(np.diff(t) <= 0):
        raise ValueError(
            "Time values must be strictly increasing"
        )

    error = setpoint - y
    ise = np.trapezoid(error**2, t)

    final_value = y[-1]

    if setpoint >= 0.0:
        peak_value = np.max(y)
        max_overshoot = max(
            0.0,
            peak_value - setpoint,
        )
    else:
        peak_value = np.min(y)
        max_overshoot = max(
            0.0,
            setpoint - peak_value,
        )

    if abs(setpoint) > 1e-12:
        overshoot_percent = (
            100.0
            * max_overshoot
            / abs(setpoint)
        )
    else:
        overshoot_percent = np.nan

    band = settling_tolerance * max(
        abs(setpoint),
        1e-12,
    )

    inside_band = np.abs(y - setpoint) <= band
    settling_time = np.inf

    for i in range(len(t)):
        if inside_band[i] and np.all(inside_band[i:]):
            settling_time = t[i]
            break

    return {
        "ise": ise,
        "final_value": final_value,
        "max_overshoot": max_overshoot,
        "overshoot_percent": overshoot_percent,
        "settling_time": settling_time,
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
    return (
        metrics_df
        .groupby("activation", as_index=False)
        .agg(
            ise_mean=("ise", "mean"),
            final_value_mean=("final_value", "mean"),
            max_overshoot_mean=(
                "max_overshoot",
                "mean",
            ),
            overshoot_percent_mean=(
                "overshoot_percent",
                "mean",
            ),
            settling_time_mean=(
                "settling_time",
                "mean",
            ),
        )
    )

def run_experiments(
    activation_tests,
    repetitions,
    test_parameters,
    hidden_layers=1,
    neurons=16,
    simulation_steps=10000,
    initial_learning_rate=0.01,
    setpoint=0.5,
    option=True,
):
    colors = colormaps["viridis"](
        np.linspace(
            0.1,
            0.9,
            len(activation_tests),
        )
    )

    experiment_data = []
    metric_results = []

    for test, color in zip(activation_tests, colors):
        name, activation_hidden, activation_output = test
        runs = []

        for _ in range(repetitions):
            data = test_parameters(
                hidden_layers=hidden_layers,
                neurons=neurons,
                max_epochs=simulation_steps,
                learning_rate=initial_learning_rate,
                setpoint=setpoint,
                isJordan=option,
                activation_hidden=activation_hidden,
                activation_output=activation_output,
            )

            runs.append(data)

        experiment_data.append({
            "name": name,
            "activation_hidden": activation_hidden,
            "activation_output": activation_output,
            "color": color,
            "runs": runs,
        })

        for repetition, run in enumerate(runs, start=1):
            run_metrics = calculate_metrics(
                run,
                setpoint=setpoint,
            )

            metric_results.append({
                "activation": name,
                "repetition": repetition,
                **run_metrics,
            })

    metrics_df = pd.DataFrame(metric_results)

    return experiment_data, metrics_df

def create_short_test_report(average_metrics):
    lines = [
        ";".join([
            "Activation",
            "ISE mean",
            "Final value mean",
            "Maximum overshoot mean",
            "Overshoot mean [%]",
            "Settling time mean [s]",
        ])
    ]

    for _, row in average_metrics.iterrows():
        lines.append(";".join([
            str(row["activation"]),
            f"{row['ise_mean']:.6g}",
            f"{row['final_value_mean']:.6f}",
            f"{row['max_overshoot_mean']:.6f}",
            f"{row['overshoot_percent_mean']:.3f}",
            f"{row['settling_time_mean']:.4f}",
        ]))

    return "\n".join(lines)

def testParameters(
    hidden_layers: int,
    neurons: int,
    max_epochs: int,
    learning_rate: float,
    setpoint: float,
    isJordan: bool,
    activation_hidden,
    activation_output,
):
    dt = 0.001

    t = np.arange(max_epochs, dtype=float) * dt
    y = np.zeros(max_epochs, dtype=float)
    u = np.zeros(max_epochs, dtype=float)

    pt2 = PT2(
        K=1.0,
        T=0.5,
        D=1.0,
        dt=dt,
    )

    if activation_hidden == activation_type_en.ACTIVATION_LEAKY_RELU:
        learning_rate = learning_rate / 10

    step_size = 2000
    gamma = 0.1
    min_learning_rate = learning_rate * gamma

    neuralController = NeuralController(
        hidden_layers=hidden_layers,
        neurons=neurons,
        learning_rate=learning_rate,
        setpoint=setpoint,
        isJordan=isJordan,
        activation_hidden=activation_hidden,
        activation_output=activation_output,
    )

    for i in range(max_epochs - 1):
        learning_rate_rel = lr.step_decay_lr(
            i,
            learning_rate,
            step_size,
            gamma,
        )

        learning_rate_rel = max(
            min_learning_rate,
            learning_rate_rel,
        )

        u[i + 1] = neuralController.run(
            y[i],
            learning_rate_rel,
            setpoint,
        )

        y[i + 1] = pt2.step(u[i + 1])

        if i % 1000 == 0:
            error = setpoint - y[i]

            print(
                f"Epoch: {i} | "
                f"Setpoint: {setpoint:.3f} | "
                f"Plant output: {y[i]:.6f} | "
                f"u: {u[i]:.6f} | "
                f"Error: {error:.6f} | "
                f"Learning rate: {learning_rate_rel:.8f}"
            )

    data = np.column_stack((t, y, u))

    del neuralController

    return data