import ctypes

neuralController = ctypes.CDLL('./libNeuralController.so')

neuralController.neuralController_Init()