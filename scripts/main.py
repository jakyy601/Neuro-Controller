import gc
from neuralControllerClass import NeuralController
import numpy as np
import matplotlib.pyplot as plt
from system import PT1, PT2, I

def main():
    gc.disable()
    yn = 0.0
    u = 0.0
    #A class is needed to run the neural controller module as the python refcount garbage collector
    #looses the c pointer and therefore just frees the memory.
    neuralController = NeuralController(2, 8, 5000, 0.01, 1.0, True)
    pt2 = PT2(K=1.0, T=1.0, D=0.3, dt=0.01)
    for i in range(neuralController.ncConfig.max_epochs):
        u = neuralController.run(yn)
        yn = pt2.step(u)
        if (i%100) == 0:
            print(f"Epoch: {i} Plant output: {yn} u: {u} Error: {neuralController.ncConfig.setpoint - yn}")
            
    del neuralController

if __name__ == "__main__":
    main()

    