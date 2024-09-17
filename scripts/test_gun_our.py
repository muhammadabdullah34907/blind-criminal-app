import cv2
from ultralytics import YOLO
import time
# Load the custom detection model
model_path = 'weapon.engine'  # Replace with the path to your .pt file
model = YOLO(model_path)

# Initialize video capture
video_path = "/dev/video0" 
video_path = "test.mp4"   # Use 0 for webcam, or specify a video file path
cap = cv2.VideoCapture(video_path)

# Define the class index for weapon (adjust if necessary)
WEAPON_CLASS = 2  # Assuming weapon is class index 2 (0: person, 1: face, 2: weapon)
frame_count = 0
start_time = time.time()
while cap.isOpened():
    frame_count=frame_count+1
    success, frame = cap.read()
    if not success:
        break
    print(frame)
    # Perform detection
    results = model(frame)

    # Process detections
    for result in results[0].boxes:
        cls = int(result.cls[0].item())
        conf = result.conf[0].item()
        
        # Only process weapon detections
        if cls == WEAPON_CLASS and conf > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for weapons
            
            # Display label
            label = f'Weapon: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Weapon Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time
print("Overall FPS: {:.2f}".format(fps))
# Release resources
cap.release()
cv2.destroyAllWindows()

print("Video processing completed.")