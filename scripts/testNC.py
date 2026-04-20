import gc
from neuralControllerClass import NeuralController

def main():
    gc.disable()
    #A class is needed to run the neural controller module as the python refcount garbage collector
    #looses the c pointer and therefore just frees the memory.
    neuralController = NeuralController()
    neuralController.run()

if __name__ == "__main__":
    main()