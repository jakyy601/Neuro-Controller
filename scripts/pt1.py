import numpy as np
import zmq
import struct

Kp = 1.0
T1 = 1.0
deltaT = 0.1
y = [0]
k = 0
uk = 0

#file = open("pt1.txt", "w")
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

while(True):
    ykp1 = y[k] + (Kp * uk - y[k])*(deltaT/T1)
    y.append(ykp1)
    #file.write(f"{repr(ykp1)}\n")
    ykp1_bytes = struct.pack('f', ykp1)
    print(ykp1_bytes)
    socket.send_string(str(ykp1_bytes))
    rx = socket.recv_string()
    uk = float(rx)
    k = k + 1
    if k == 1000:
        break

#file.close()
