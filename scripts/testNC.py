import ctypes
import random

# --- struct definitions (fill in actual fields) ---

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


# --- initialize neuralControllerConfig_st --- #
ncConfig = neuralControllerConfig_st(
    hidden_layers = 2,
    layers = 4,
    neurons = 8,
    output_layer_neurons = 1,
    inputs = 2,
    max_epochs = 5000,
    initialized = 0,
    learning_rate = 0.01,
    setpoint = 0.0,
    arch = arch_st(),
)

# --- variable declarations ---

ncConfigPtr = ctypes.pointer(ncConfig)

control = control_st()
control_ptr = ctypes.pointer(control)

FCTPTR_TYPE = ctypes.CFUNCTYPE(ctypes.c_float)

@FCTPTR_TYPE
def random_ctypes_float(low=0.01, high=0.1):
    return random.uniform(low, high)

def i_plant(yn: float, u: float) -> float:
    K = 1.0
    T = 0.1
    return yn + K * u * T

weight_val  = ctypes.c_double(0.0)
weight_p1   = ctypes.pointer(weight_val)            # double*
weight_p2   = ctypes.pointer(weight_p1)             # double**
weight_p3   = ctypes.pointer(weight_p2)             # double***
weight_p4   = ctypes.pointer(weight_p3)             # double****

neuron      = neuron_st()
neuron_p1   = ctypes.pointer(neuron)                # neuron_st*
neuron_p2   = ctypes.pointer(neuron_p1)             # neuron_st**
neuron_p3   = ctypes.pointer(neuron_p2)             # neuron_st***

# --- function binding ---
lib = ctypes.CDLL("./scripts/neuralControllerInterface.so") 

lib.neuralController_Init.restype  = ctypes.c_int
lib.neuralController_Init.argtypes = [
    ctypes.POINTER(neuralControllerConfig_st),  # ncConfig
    ctypes.POINTER(control_st),                 # control
    FCTPTR_TYPE,                                # fctPtr
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))),  # pWeight
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(neuron_st))),                        # pNeuron
]

# --- call ---
result = lib.neuralController_Init(
    ncConfigPtr,
    control_ptr,
    random_ctypes_float,
    weight_p4,
    neuron_p3,
)

inputs = (ctypes.c_double * 2)()
yn = 0.0
output = (ctypes.c_double)(0.0)
outputPtr = ctypes.pointer(output)

for i in range(ncConfig.max_epochs):
    inputs[0] = yn
    lib.neuralController_Run(ncConfigPtr, control_ptr, outputPtr, inputs, weight_p3, neuron_p2)
    yn = i_plant(yn, output.value)
    if (i%100) == 0:
        print(f"Epoch: {i} Plant output: {yn} u: {ncConfig.setpoint - yn} Error: {output}")

lib.neuralController_Free(ncConfigPtr, control_ptr, weight_p3, neuron_p2)
