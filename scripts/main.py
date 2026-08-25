import gc
from neuralControllerClass import NeuralController
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from system import PT1, PT2, I


def calculateLearningRate(t, t0, k, lr_min, lr_max):
        return lr_min + (lr_max - lr_min) / (1 + np.exp((t - t0) / k))

def testParameters(hidden_layers: int, neurons: int, max_epochs: int, learning_rate: float, setpoint: float, isJordan: bool):
    dt = 0.002
    t = np.arange(0.0, max_epochs * dt, dt) # array for time
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    pt2 = PT2(K=1.0, T=1.0, D=0.3, dt=dt)

    # A class is needed to run the neural controller module as the python refcount garbage collector
    # looses the c pointer and therefore just frees the memory.
    neuralController = NeuralController(hidden_layers, neurons, max_epochs, learning_rate, setpoint, isJordan)
    for i in range(neuralController.ncConfig.max_epochs-1):
        learning_rate_rel = calculateLearningRate(i, max_epochs*0.4, max_epochs*0.08, learning_rate/10, learning_rate)
        # learning_rate_rel = learning_rate
        u[i+1] = neuralController.run(y[i], learning_rate_rel)
        y[i+1] = pt2.step(u[i+1])
        if (i%1000) == 0:
            print(f"Epoch: {i} Plant output: {y[i]} u: {u[i]} Error: {neuralController.ncConfig.setpoint - y[i]} Learning Rate: {learning_rate_rel}")

    data = np.column_stack((t,y,u))
    del neuralController
    return data

def main():
    gc.disable()

    p2_values = range(4, 20)
    repetitions = 10

    # One color per p2 value
    colors = colormaps["viridis"](
        np.linspace(0.1, 0.9, len(p2_values))
    )

    fig_y, ax_y = plt.subplots()
    fig_u, ax_u = plt.subplots()

    for p2, color in zip(p2_values, colors):
        runs = []

        # Repeat this p2 value ten times
        for _ in range(repetitions):
            data = testParameters(
                3,
                p2,
                20000,
                0.01,
                1.0,
                True
            )
            runs.append(data)

        # Extract values from all runs
        t = runs[0][:, 0]
        y_all = np.array([run[:, 1] for run in runs])
        u_all = np.array([run[:, 2] for run in runs])

        # Plot individual runs in the background
        for y, u in zip(y_all, u_all):
            ax_y.plot(
                t,
                y,
                color=color,
                alpha=0.20,
                linewidth=1
            )

            ax_u.plot(
                t,
                u,
                color=color,
                alpha=0.20,
                linewidth=1
            )

        # Calculate and plot averages in the foreground
        y_average = y_all.mean(axis=0)
        u_average = u_all.mean(axis=0)

        ax_y.plot(
            t,
            y_average,
            color=color,
            linewidth=2.5,
            label=f"p2 = {p2}",
            zorder=10
        )

        ax_u.plot(
            t,
            u_average,
            color=color,
            linewidth=2.5,
            label=f"p2 = {p2}",
            zorder=10
        )

    ax_y.set_title("y(t) for different p2 values")
    ax_y.set_xlabel("Time")
    ax_y.set_ylabel("y(t)")
    ax_y.grid(True)
    ax_y.legend(title="Average runs")

    ax_u.set_title("u(t) for different p2 values")
    ax_u.set_xlabel("Time")
    ax_u.set_ylabel("u(t)")
    ax_u.grid(True)
    ax_u.legend(title="Average runs")

    fig_y.tight_layout()
    fig_u.tight_layout()

    plt.show()
    

if __name__ == "__main__":
    main()
