import gc
from neuralControllerClass import NeuralController
import numpy as np
import matplotlib.pyplot as plt
import control as ct

def i_plant(yn1: float, u: float) -> float:
    K = 1.0
    T = 0.1
    return yn1 + (K * u * T)

def main():
    gc.disable()
    yn = 0.0
    u = 0.0
    #A class is needed to run the neural controller module as the python refcount garbage collector
    #looses the c pointer and therefore just frees the memory.
    neuralController = NeuralController()
    for i in range(neuralController.ncConfig.max_epochs):
        u = neuralController.run(yn)
        yn = i_plant(yn, u)
        if (i%100) == 0:
            print(f"Epoch: {i} Plant output: {yn} u: {u} Error: {neuralController.ncConfig.setpoint - yn}")
            
    del neuralController

if __name__ == "__main__":
    main()

    