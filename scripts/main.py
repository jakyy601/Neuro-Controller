import gc
from neuralControllerClass import NeuralController
import numpy as np
import matplotlib.pyplot as plt
from system import PT1, PT2, I


def calculateLearningRate(t, t0, k, lr_min, lr_max):
        return lr_min + (lr_max - lr_min) / (1 + np.exp((t - t0) / k))

def testParameters(hidden_layers: int, neurons: int, max_epochs: int, learning_rate: float, setpoint: float, isJordan: bool):
    dt = 0.002
    t = np.arange(0.0, max_epochs * dt, dt) # array for time
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    pt2 = PT2(K=1.0, T=1.0, D=0.3, dt=dt)

    #A class is needed to run the neural controller module as the python refcount garbage collector
    #looses the c pointer and therefore just frees the memory.
    neuralController = NeuralController(hidden_layers, neurons, max_epochs, learning_rate, setpoint, isJordan)
    for i in range(neuralController.ncConfig.max_epochs-1):
        learning_rate_rel = calculateLearningRate(i, max_epochs*0.4, max_epochs*0.08, learning_rate/10, learning_rate)
        u[i+1] = neuralController.run(y[i], learning_rate_rel)
        y[i+1] = pt2.step(u[i+1])
        if (i%1000) == 0:
            print(f"Epoch: {i} Plant output: {y[i]} u: {u[i]} Error: {neuralController.ncConfig.setpoint - y[i]} Learning Rate: {learning_rate_rel}")

    data = np.column_stack((t,y,u))
    del neuralController
    return data

def main():
    gc.disable()
    yn = 0.0
    u = 0.0
    data = np.array([])

    data = testParameters(2, 8, 10000, 0.01, 1.0, True)
    t = data[:,0]
    y = data[:,1]
    u = data[:,2]
    plt.figure()
    plt.plot(t, y, label="y(t)")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(t, u, label="u(t)")
    plt.legend()
    plt.grid(True)
    plt.show()        
    

if __name__ == "__main__":
    main()
