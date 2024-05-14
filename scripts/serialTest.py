import serial
import struct
import time

ser = serial.Serial('COM5', 115200, timeout=1)
Kp = 1.0
T1 = 1.0
deltaT = 0.1
y = [0]
k = 0
uk = 1
tx_buff = [0] * 20

while(True):
    ykp1 = y[k] + (Kp * uk - y[k])*(deltaT/T1)
    y.append(ykp1)
    k = k + 1
    float_str = str(ykp1)
    buff = list(float_str.encode('utf-8'))
    for idx, byte in enumerate(buff):
        tx_buff[idx] = buff[idx]
    ser.write(tx_buff)
    print(f"Sent {tx_buff} over UART")
    time.sleep(0.02)
    #rx = ser.read(20)
    #print(f"Got message {rx}")
    #rx = 0