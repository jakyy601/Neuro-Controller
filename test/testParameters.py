import subprocess
import threading
import os

file = ".\\build\\main.exe"

args = ' 2 3 0.01 6000 8 1 1.0'

process_command = file + args

print(process_command)

process = subprocess.Popen(process_command, bufsize=0, shell=True)

process.wait()