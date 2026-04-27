import ctypes
import random

lib = ctypes.CDLL("./scripts/neuralControllerInterface.dll") 

class arch_st(ctypes.Structure):
    _fields_ = [
        ("topology", ctypes.POINTER(ctypes.c_int)),
        ("total_neurons", ctypes.c_int),
        ("total_weights", ctypes.c_int),
    ]

class neuralControllerConfig_st(ctypes.Structure):
    _fields_ = [
        ("hidden_layers",        ctypes.c_int),
        ("layers",               ctypes.c_int),
        ("neurons",              ctypes.c_int),
        ("output_layer_neurons", ctypes.c_int),
        ("inputs",               ctypes.c_int),
        ("max_epochs",           ctypes.c_int),
        ("initialized",          ctypes.c_int),
        ("learning_rate",        ctypes.c_double),
        ("setpoint",             ctypes.c_double),
        ("arch",                 arch_st),
    ]

class control_st(ctypes.Structure):
    _fields_ = [
        ("act_old", ctypes.c_double),
        ("act_new", ctypes.c_double),
        ("rating", ctypes.c_double),
        ("input", ctypes.POINTER(ctypes.c_double)),
        ("input_old", ctypes.POINTER(ctypes.c_double)),
    ]

class neuron_st(ctypes.Structure):
    _fields_ = [
        ("netinput", ctypes.c_double),
        ("netoutput", ctypes.c_double),
        ("bias", ctypes.c_double),
        ("sigma", ctypes.c_double),
    ]

FCTPTR_TYPE = ctypes.CFUNCTYPE(ctypes.c_float)

@FCTPTR_TYPE
def random_ctypes_float() -> float:
    return random.uniform(a=0.01, b=0.1)

class NeuralController:
    lib.neuralController_Init.restype  = ctypes.c_int
    lib.neuralController_Init.argtypes = [
    ctypes.POINTER(neuralControllerConfig_st),                                          # ncConfig
    ctypes.POINTER(control_st),                                                         # control
    FCTPTR_TYPE,                                                                        # fctPtr
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))),    # pWeight
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(neuron_st))),                          # pNeuron
    ]

    lib.neuralController_Run.restype = ctypes.c_int
    lib.neuralController_Run.argtypes = [
        ctypes.POINTER(neuralControllerConfig_st),                                      # ncConfig
        ctypes.POINTER(control_st),                                                     # control
        ctypes.POINTER(ctypes.c_double),                                                # pOutput
        ctypes.POINTER(ctypes.c_double),                                                 # pInput
        ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),                # weight
        ctypes.POINTER(ctypes.POINTER(neuron_st)),                                      # neuron
    ]

    lib.neuralController_Free.argtypes = [
        ctypes.POINTER(neuralControllerConfig_st),                                      # ncConfig
        ctypes.POINTER(control_st),                                                     # control
        ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),                # weight
        ctypes.POINTER(ctypes.POINTER(neuron_st)),                                      # neuron
    ]

    def __init__(self):
        self.ncConfig = neuralControllerConfig_st(
            hidden_layers = 2,
            layers = 4,
            neurons = 8,
            output_layer_neurons = 1,
            inputs = 2,
            max_epochs = 5000,
            initialized = 0,
            learning_rate = 0.01,
            setpoint = 1.0,
            arch = arch_st(),
        )
        self.control = control_st()

        self.weightPtr = ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
        self.weights = self.weightPtr()

        self.neuronPtr = ctypes.POINTER(ctypes.POINTER(neuron_st))
        self.neurons = self.neuronPtr()


        self.result = lib.neuralController_Init(
        ctypes.byref(self.ncConfig),
        ctypes.byref(self.control),
        random_ctypes_float,
        ctypes.byref(self.weights),
        ctypes.byref(self.neurons),
        )
        self.yn = (ctypes.c_double)()
        self.output = (ctypes.c_double)()

    def run(self, input: float) -> float:
        self.control.input[0] = ctypes.c_double(input)
        lib.neuralController_Run(ctypes.byref(self.ncConfig), ctypes.byref(self.control), ctypes.byref(self.output), self.control.input, self.weights, self.neurons)
        return self.output.value

    def __del__(self):
        lib.neuralController_Free(ctypes.byref(self.ncConfig), ctypes.byref(self.control), self.weights, self.neurons)

