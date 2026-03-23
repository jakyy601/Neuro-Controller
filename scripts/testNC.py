import sys
import neuralControllerInterface

cfg = neuralControllerInterface.NeuralController(hidden_layers=2, layers=4, neurons=8,
    output_layer_neurons=1, inputs=10,
    max_epochs=5000, learning_rate=0.001, setpoint=1.0)
print(cfg)