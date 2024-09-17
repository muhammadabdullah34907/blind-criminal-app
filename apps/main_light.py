import sys
import argparse
import time
import cv2
import numpy as np
import serial
import glob
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, cudaFont
from ultralytics import YOLO
import jetson_utils
import threading
from pynput import keyboard

exit_flag = False

def on_press(key):
    global exit_flag
    try:
        if key.char == 'q':  # Check if 'q' key is pressed
            print("Key 'q' pressed. Exiting...")
            exit_flag = True
            return False  # Stop listener
    except AttributeError:
        pass

def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# Start the listener in a separate thread to stop
keyboard_thread = threading.Thread(target=start_keyboard_listener)
keyboard_thread.start()

#Serial ports and connecting to them for ptz commnands

def get_available_ports():
    return glob.glob('/dev/tty[A-Za-z]*')

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


# Pelco-D protocol commands structure to control our PTZ camera
def pelco_command(address, command1, command2, data1, data2):
    msg = bytearray([0xFF, address, command1, command2, data1, data2])
    msg.append(sum(msg[1:]) % 256)  # Checksum
    return msg

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



def up_right(speed=0x10):
    if ser:
        speed=0x10
        ser.write(pelco_command(0x01, 0x00, 0x0A, speed, speed))
        print("Up-right command sent")
    else:
        print("UP-rifht command - No serial port available")
def up_left(speed=0x10):
    if ser:
        speed=0x10
        ser.write(pelco_command(0x01, 0x00, 0x0C, speed, speed))
        print("up-left command sent")
    else:
        print("up-left command - No serial port available")
def down_right(speed=0x10):
    if ser:
        speed=0x10
        ser.write(pelco_command(0x01, 0x00, 0x12, speed, speed))
        print("Down right command sent")
    else:
        print("Down right command - No serial port available")
def down_left(speed=0x10):
    if ser:
        speed=0x10
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


# # Function to control PTZ based on keypoint position
# def track_keypoint2(frame_center, keypoint):
#     if keypoint is None:
#         return

#     x, y = keypoint.x, keypoint.y
#     center_x, center_y = frame_center
#     threshold_x = 50  # Adjust this value to change horizontal sensitivity
#     threshold_y = 50  # Adjust this value to change vertical sensitivity

#     diff_x = x - center_x
#     diff_y = y - center_y

#     curr = "stop"

#     if abs(diff_x) > threshold_x and abs(diff_y) > threshold_y:
#         # Diagonal movements
#         if diff_x < 0 and diff_y < 0:
#             up_left()
#             curr = "up_left"
#         elif diff_x < 0 and diff_y > 0:
#             down_left()
#             curr = "down_left"
#         elif diff_x > 0 and diff_y < 0:
#             up_right()
#             curr = "up_right"
#         else:  # diff_x > 0 and diff_y > 0
#             down_right()
#             curr = "down_right"
#     elif abs(diff_x) > threshold_x:
#         # Horizontal movements
#         if diff_x < 0:
#             pan_left()
#             curr = "pan_left"
#         else:
#             pan_right()
#             curr = "pan_right"
#     elif abs(diff_y) > threshold_y:
#         # Vertical movements
#         if diff_y < 0:
#             tilt_up()
#             curr = "tilt_up"
#         else:
#             tilt_down()
#             curr = "tilt_down"
#     else:
#         stop()
#         curr = "stop"

#     return curr
# Function to control PTZ based on keypoint position
def track_keypoint(frame_center, keypoint):
    if keypoint is None:
        return
    curr=""
    x, y = keypoint.x, keypoint.y
    center_x, center_y = frame_center
    thresholdpan = 120 # Adjust this value to change sensitivity
    thresholdtilt=120
    if x < center_x - thresholdpan:
        curr="panleft"
        pan_left()
    elif x > center_x + thresholdpan:
        pan_right()
        curr="panright"    

    elif y < center_y - thresholdtilt:
        tilt_up()
        curr="tiltup"
    elif y > center_y + thresholdtilt:
        tilt_down()
        curr="tiltdown"
    else:
        stop()
        light_on

    # if  prev!=curr: #fc%10==0: 
    #     stop()
    #     prev=curr
    #     print("ST",curr)
    return prev
# Parse command line arguments
parser = argparse.ArgumentParser(description="Run pose estimation and weapon detection on a video stream.")
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="densenet121-body", help="pre-trained model to load for pose estimation")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

args = parser.parse_known_args()[0]

# pose estimation model
pose_net = poseNet(args.network, sys.argv, args.threshold)

#weapon detection model
weapon_model_path = 'weapon.engine'
weapon_model = YOLO(weapon_model_path)

#  video sources & outputs
input_source = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
font=cudaFont()

WEAPON_CLASS = 2
global light_state
light_state=0
frame_count = 0
start_time = time.time()
weapon_detected = False
prev=''
#tracked_poseid
tracked_poseid=None
# Process frames 
tracked_pose=None
while not exit_flag:
    # Capture 
    img = input_source.Capture()

    if img is None:  
        continue
    frame_count=frame_count+1
    # Convert to CV2 image for weapon detection
    frame = np.array(img, copy=True, order='C')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    #success, frame = cv2.VideoCapture(args.input).read()
    # Perform pose estimation (with overlay)
    poses = pose_net.Process(img, overlay=args.overlay)
    if len(poses)<1:
        stop()
        light_off()
    # Perform weapon detection
    if weapon_detected==False:
        weapon_results = weapon_model(frame)
        
        
        weapon_center = None
        # Process weapon detections

        for result in weapon_results[0].boxes:
            cls = int(result.cls[0].item())
            conf = result.conf[0].item()
            
            if cls == WEAPON_CLASS and conf > 0.5:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                weapon_detected = True
                weapon_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                print("Gun detected")
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                jetson_utils.cudaDrawRect(img, (x1,y1,x2,y2), (255,127,0,200))

                # Display label
                label = f'Weapon: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        

    # tracked_pose = None
    min_distance = float('inf')
    #print("detected {:d} objects in image".format(len(poses)))
    if weapon_detected and tracked_poseid==None:
        font.OverlayText(img, text=f"Gun", 
                         x=5, y=5  * (font.GetSize() + 5),
                         color=font.White, background=font.Gray40)
        jetson_utils.cudaDrawRect(img, (x1,y1,x2,y2), (255,127,0,200))

        for pose in poses:
            print("poseid=",pose.ID)
            # Find the hand keypoint closest to the weapon
            for keypoint in pose.Keypoints:
                print(keypoint)
                if keypoint.ID in [9, 10]:  # 9 and 10 are hand keypoints
                    distance = ((keypoint.x - weapon_center[0])**2 + (keypoint.y - weapon_center[1])**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        tracked_pose = pose
                        tracked_poseid=pose.ID
                # else:
                #     if pose.ID==tracked_poseid:
                #         tracked_pose=pose

 # Track the face of the person holding the weapon
    for pose in poses:
        if tracked_poseid==pose.ID:
            tracked_pose=pose
        else:
            stop()
    if tracked_pose and len(tracked_pose.Keypoints) > 0:
        #face_keypoint = tracked_pose.Keypoints[0]  # 0 is the nose or a face keypoint
        face_keypoint = tracked_pose.Keypoints[0]
        print(face_keypoint)
        frame_center = (img.width / 2, img.height / 2)
        track_keypoint(frame_center, face_keypoint)

    # Render the image
    output.Render(img)

    # Update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, pose_net.GetNetworkFPS()))

    # Print out performance info
    pose_net.PrintProfilerTimes()



    # Exit on input/output EOS
    if not input_source.IsStreaming() or not output.IsStreaming():
        break

# Calculate and print the overall FPS
end_time = time.time()
stop()
light_off()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time
print("Overall FPS: {:.2f}".format(fps))

# Cleanup
if ser:
    ser.close()