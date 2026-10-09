import matplotlib.pyplot as plt
import numpy as np


def plot_response(
    ax,
    runs,
    label,
    color,
    signal_column,
):
    """
    Plot individual runs in the background and their
    pointwise average in the foreground.
    """

    t = runs[0][:, 0]

    signal_all = np.asarray([
        run[:, signal_column]
        for run in runs
    ])

    for signal in signal_all:
        ax.plot(
            t,
            signal,
            color=color,
            alpha=0.20,
            linewidth=1.0,
        )

    signal_average = signal_all.mean(axis=0)

    ax.plot(
        t,
        signal_average,
        color=color,
        linewidth=2.5,
        label=label,
        zorder=10,
    )

    return t, signal_all, signal_average


def configure_plot(ax, title, y_label):
    """Apply common formatting to a plot."""

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend(title="Activation functions")


def plot_all_experiments(experiment_data):
    """
    Create one figure for y(t) and one figure for u(t).

    Expected experiment_data entries:
        {
            "name": str,
            "color": color,
            "runs": list of arrays,
        }

    Expected run columns:
        data[:, 0] = t
        data[:, 1] = y
        data[:, 2] = u
    """

    fig_y, ax_y = plt.subplots(figsize=(10, 6))
    fig_u, ax_u = plt.subplots(figsize=(10, 6))

    for experiment in experiment_data:
        label = experiment["name"]

        plot_response(
            ax=ax_y,
            runs=experiment["runs"],
            label=label,
            color=experiment["color"],
            signal_column=1,
        )

        plot_response(
            ax=ax_u,
            runs=experiment["runs"],
            label=label,
            color=experiment["color"],
            signal_column=2,
        )

    configure_plot(
        ax_y,
        title="Plant output for activation functions",
        y_label="Plant output y(t)",
    )

    configure_plot(
        ax_u,
        title="Controller output for activation functions",
        y_label="Controller output u(t)",
    )

    fig_y.tight_layout()
    fig_u.tight_layout()

    return fig_y, fig_u