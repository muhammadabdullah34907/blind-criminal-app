import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open the video file or camera
cap = cv2.VideoCapture("test.mp4")
start=time.time()
# Variables for FPS calculation
fps_start_time = time.time()
fps = 0
frame_count = 0

# Define the frame size (optional, for downscaling)
frame_width = 640
frame_height = 480

# Process every nth frame
frame_skip = 5  # Adjust as needed
current_frame = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame count
    frame_count += 1
    current_frame += 1

    # Downscale the frame (optional)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Process every nth frame
    if current_frame % frame_skip == 0:
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect pose landmarks
        result = pose.process(frame_rgb)
        
        # Draw the pose landmarks on the frame
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Calculate FPS
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    if time_diff >= 1:
        fps = frame_count / time_diff
        fps_start_time = time.time()
        frame_count = 0

    # Display FPS on frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Pose Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
end=time.time()
print(end-start)
# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
