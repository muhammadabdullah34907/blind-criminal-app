import cv2
from ultralytics import YOLO

# Load the gun detection model
model_path = 'best.pt'  # Ensure this path is correct
model = YOLO(model_path)

# Initialize video capture
video_path = "test.mp4" # Use 0 for webcam, or specify a video file path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Detect guns in the frame using YOLO
    results = model(frame)

    # Draw bounding boxes for detected guns
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        confidence = result.conf[0].item()
        
        if confidence > 0.5:  # Confidence threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Gun: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Gun Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Video processing completed.")