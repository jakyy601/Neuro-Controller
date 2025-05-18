import subprocess
import threading
import os
import csv

file = ".\\build\\main.exe"

csv_file_name = "test.csv"

column = 2

inputs = 2
hidden_layers = 3
learning_rate = 0.01
max_epochs = 6000
neurons = 8
output_layer_neurons = 1
setpoint = 1.0

args = ' ' + str(inputs) + ' ' + str(hidden_layers) + ' ' + str(learning_rate) + ' ' + str(max_epochs) + ' ' + str(neurons) + ' ' + str(output_layer_neurons) + ' ' + str(setpoint) + ' ' + csv_file_name + ' ' + str(column)

process_command = file + args

if os.path.exists(csv_file_name):
    reset = input(f"File {csv_file_name} already exists. Do you want to reset it? (y/n): ")
    if reset.lower() != 'y':
        raise Exception(f"File {csv_file_name} already exists and was not reset.")
    else:
        with open(csv_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for i in range(1, max_epochs + 1):
                writer.writerow([i])
else:
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(1, max_epochs + 1):
            writer.writerow([i])

print(process_command)

process = subprocess.Popen(process_command, bufsize=0, shell=True)

process.wait()