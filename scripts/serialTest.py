import serial

ser = serial.Serial('COM8', 115200, timeout=1)


while(True):
    ser.write(b'Test')
    rx = ser.read(50)
    print(rx)