import sys
import argparse
import time
import cv2
import numpy as np
import serial
import glob
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaFont,cudaDrawRect

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
def pan_left(speed= 0x20):
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x04, speed, speed))
        print("Pan left command sent")
    else:
        print("Pan left command - No serial port available")

def pan_right(speed= 0x20):
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x02, speed, speed))
        print("Pan right command sent")
    else:
        print("Pan right command - No serial port available")

def tilt_up(speed= 0x25):
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x08, speed, speed))
        print("Tilt up command sent")
    else:
        print("Tilt up command - No serial port available")

def tilt_down(speed= 0x25):
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
def light_on(speed=0x3F):
    if ser:
        speed=0x32
        ser.write(pelco_command(0x01, 0x00, 0x09, 0x00, 0x02))
        print("Light on command sent")
    else:
        print("Light on command - No serial port available")
def light_off(speed=0x3F):
    if ser:
        speed=0x32
        ser.write(pelco_command(0x01, 0x00, 0x09, 0x00, 0x02))
        print("Light off command sent")
    else:
        print("Light off command - No serial port available")

def stop():
    if ser:
        ser.write(pelco_command(0x01, 0x00, 0x00, 0x00, 0x00))
        print("Stop command sent")
    else:
        print("Stop command - No serial port available")

# Function to control PTZ based on keypoint position
def track_keypoint(frame_center, keypoint):
    if keypoint is None:
        return
    curr=""
    x, y = keypoint.x, keypoint.y
    center_x, center_y = frame_center
    thresholdpan= 240 # Adjust this value to change sensitivity
    thresholdtilt= 180
    if x < center_x - thresholdpan:
        curr="panleft"
        pan_left(0x20)
    elif x > center_x + thresholdpan:
        pan_right(0x20)
        curr="panright"    

    elif y < center_y - thresholdtilt:
        tilt_up(0x25)
        curr="tiltup"
    elif y > center_y + thresholdtilt:
        tilt_down(0x25)
        curr="tiltdown"
    else:
        stop()
def track_keypoint2(frame_center, keypoint):
    if keypoint is None:
        return
    curr=""
    prev=keypoint
    x, y = keypoint.x, keypoint.y
    center_x, center_y = frame_center
    thresholdpanfast= 240 
    thresholdtiltfast= 180
    thresholdpanslow= 80
    thresholdtiltslow= 40
    if x < center_x - thresholdpanfast:
        curr="panleft"
        pan_left(0x20)
    elif x > center_x + thresholdpanfast:
        pan_right(0x20)
        curr="panright"    

    elif y < center_y - thresholdtiltfast:
        tilt_up(0x25)
        curr="tiltup"
    elif y > center_y + thresholdtiltfast:
        tilt_down(0x25)
        curr="tiltdown"
    elif x < center_x - thresholdpanslow and x > center_x - thresholdpanfast:
        curr="panleft"
        pan_left(0x10)
    elif x > center_x + thresholdpanslow and x < center_x + thresholdpanfast:
        pan_right(0x10)
        curr="panright"    

    elif y < center_y - thresholdtiltslow and y > center_y - thresholdtiltfast:
        tilt_up(0x10)
        curr="tiltup"
    elif y > center_y + thresholdtiltslow and  y < center_y + thresholdtiltfast:
        tilt_down(0x10)
        curr="tiltdown"
    else:
        stop()

        

    # if  prev!=curr: #fc%10==0: 
    #     stop()
    #     prev=curr
    #     print("ST",curr)
    return 0

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.2, help="minimum detection threshold to use")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# Load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# Create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()
trackedposeid=0
font = cudaFont()

# Process frames until EOS or the user exits
while True:
    # Capture the next image
    img = input.Capture()

    if img is None:  # timeout
        continue

    # Perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)
    if len(poses)<1:
        stop()
    # Print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print("IN POSE")
        # print(pose.Keypoints)
        # print('Links', pose.Links)

        # Track the nose keypoint (ID 0)
        if len(pose.Keypoints) > 0 and pose.ID==0:
            nose = pose.Keypoints[0]
            print(pose)
            # trackedposeid=pose.ID
            frame_center = (img.width / 2, img.height / 2)
            cudaDrawRect(img, (pose.Left,pose.Top,pose.Right,pose.Bottom), (255,127,0,200))
            # font.OverlayText(img, text="Person is far", 
            # x=5, y=5 + 1 * (font.GetSize() + 5),
            # color=font.White, background=font.Gray40)
            track_keypoint(frame_center, nose)

    # Render the image
    output.Render(img)

    # Update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # Print out performance info
    net.PrintProfilerTimes()

    # Update frame count
    frame_count += 1

    # Exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

# Calculate and print the overall FPS
end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time
print("Overall FPS: {:.2f}".format(fps))

# Cleanup
if ser:
    ser.close()
