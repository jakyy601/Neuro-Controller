import subprocess
import numpy as np
import struct

Kp = 1.0
T1 = 1.0
deltaT = 0.1
y = [0]
k = 0
uk = 1

p = subprocess.Popen(['build/test3.exe'], stdin=subprocess.PIPE)

while(True):
    ykp1 = y[k] + (Kp * uk - y[k])*(deltaT/T1)
    y.append(ykp1)
    data = struct.pack('d', ykp1)
    p.stdin.write(data)
    k = k + 1
    if k == 100:
        break

p.terminate()
