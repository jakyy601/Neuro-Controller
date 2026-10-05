import gc
from neuralControllerClass import NeuralController
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from system import PT1, PT2, I
import learnrate as lr
import plot
import metrics

def main():
    gc.disable()

    p2_values = list(range(6, 20))
    repetitions = 10

    experiment_data, metrics_df = metrics.run_experiments(
        p2_values=p2_values,
        repetitions=repetitions,
        test_parameters=metrics.testParameters,
        controller_type=3,
        simulation_steps=6000,
        initial_learning_rate=0.01,
        setpoint=1.0,
        option=True,
    )

    # Plot all runs and averages
    plot.plot_all_experiments(experiment_data)

    # Calculate mean and standard deviation per p2
    average_metrics = metrics.calculate_average_metrics(metrics_df)

    print("\nMetrics for every individual run:")
    print(metrics_df.to_string(index=False))

    print("\nMean and standard deviation per p2:")
    print(average_metrics.to_string(index=False))

    plt.show()
    

if __name__ == "__main__":
    main()
