import cv2
import os
import numpy as np
import serial
import glob
import time

# Video capture setup
cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
global light_state
light_state=0

# Function to find available serial ports
def get_available_ports():
    return glob.glob('/dev/tty[A-Za-z]*')

# Try to open a serial port
ser = None
available_ports = get_available_ports()
print(f"Available ports: {available_ports}")

for port in available_ports:
    try:
        ser = serial.Serial(
            port=port,
            baudrate=2400,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1
        )
        print(f"Successfully opened serial port: {port}")
        break
    except (OSError, serial.SerialException):
        print(f"Failed to open serial port: {port}")

if ser is None:
    print("Could not open any serial port. PTZ control will be unavailable.")

# Pelco-D protocol command structure
def pelco_command(address, command1, command2, data1, data2):
    msg = bytearray([0xFF, address, command1, command2, data1, data2])
    msg.append(sum(msg[1:]) % 256)  # Checksum
    return msg

# PTZ control functions
# PTZ control functions
def pan_left(speed= 0x15):
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x04, speed, speed))
        print("Pan left command sent")
    else:
        print("Pan left command - No serial port available")

def pan_right(speed= 0x15):
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x02, speed, speed))
        print("Pan right command sent")
    else:
        print("Pan right command - No serial port available")

def tilt_up(speed= 0x10):
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x08, speed, speed))
        print("Tilt up command sent")
    else:
        print("Tilt up command - No serial port available")

def tilt_down(speed= 0x10):
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x10, speed, speed))
        print("Tilt down command sent")
    else:
        print("Tilt down command - No serial port available")



def up_right(speed=0x3F):
    if ser:
        speed=0x3F
        ser.write(pelco_command(0x01, 0x00, 0x0A, speed, speed))
        print("Up-right command sent")
    else:
        print("UP-rifht command - No serial port available")
def up_left(speed=0x3F):
    if ser:
        speed=0x3F
        ser.write(pelco_command(0x01, 0x00, 0x0C, speed, speed))
        print("up-left command sent")
    else:
        print("up-left command - No serial port available")
def down_right(speed=0x3F):
    if ser:
        speed=0x3F
        ser.write(pelco_command(0x01, 0x00, 0x12, speed, speed))
        print("Down right command sent")
    else:
        print("Down right command - No serial port available")
def down_left(speed=0x3F):
    if ser:
        speed=0x3F
        ser.write(pelco_command(0x01, 0x00, 0x14, speed, speed))
        print("Tilt up command sent")
    else:
        print("Tilt up command - No serial port available")
def light():
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x09, 0x00, 0x02))
        print("Light command sent")
    else:
        print("Light on command - No serial port available")

def light_on():
    global light_state
    if light_state==0:
        light()
        light_state = (light_state + 1) % 3
    elif light_state==1:
        None
    elif light_state==2:
        light()
        time.sleep(0.8)
        light()
        light_state = (light_state + 2) % 3
    print(light_state)

def light_off():
    global light_state
    if light_state==0:
        None
        
    elif light_state==1:
        light()
        time.sleep(0.8)
        light()
        light_state = (light_state + 2) % 3
    elif light_state==2:
        light()
        light_state = (light_state + 1) % 3
def stop():
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x00, 0x00, 0x00))
        print("Stop command sent")
    else:
        print("Stop command - No serial port available")

# Settings
write = False
display = True
os.makedirs('test', exist_ok=True)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue

    print(frame.shape)
    frame = cv2.resize(frame, (640, 640))

    if display:
        cv2.imshow("Camera", frame)

    if write and i % 100 == 0 and i < 5000:
        cv2.imwrite(f"test/frame{i}.jpg", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        stop()
        light_off()
        break
    elif key == ord('a'):
        pan_left()
    elif key == ord('d'):
        pan_right()
    elif key == ord('w'):
        tilt_up()
    elif key == ord('s'):
        tilt_down()
    elif key == ord(' '):
        stop()
    elif key == ord('z'):
        up_left()
    elif key == ord('v'):
        up_right()
    elif key == ord('x'):
        down_left()
    elif key == ord('c'):
        down_right()
    elif key == ord('o'):
        light()
        print(light_state)
    elif key == ord('b'):
        light()
    elif key == ord('f'):
        light_off()
        print(light_state)

    i += 1

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
