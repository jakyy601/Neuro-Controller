import numpy as np

Kp = 1.0
T1 = 1.0
deltaT = 0.1
y = [0]
k = 0
uk = 1

file = open("pt1.txt", "w")

while(True):
    ykp1 = y[k] + (Kp * uk - y[k])*(deltaT/T1)
    y.append(ykp1)
    file.write(f"{repr(ykp1)}\n")
    k = k + 1
    if k == 1000:
        break

file.close()
