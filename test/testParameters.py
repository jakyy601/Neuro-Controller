import subprocess
import threading

def stdout_log():
    for line in iter(process.stdout.readline, ''):
        if not(thread2.is_alive()):
            break
        print(line, end='\n')

def monitor_process():
    process.wait()

file = "build\\main.exe"

args = ['2 3 0.01 6000 8 1 1.0']

process = subprocess.Popen(file, args, stdout=subprocess.PIPE)

thread1 = threading.Thread(target=stdout_log)
thread2 = threading.Thread(target=monitor_process)

thread1.start()
thread2.start()
