import cv2
import time
start=time.time()
# Open the video file or camera
cap = cv2.VideoCapture("test.mp4")

# Variables for FPS calculation
fps_start_time = time.time()
fps = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame count
    frame_count += 1

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
    cv2.imshow('Video Feed', frame)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
end=time.time()
print(end-start)
# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
