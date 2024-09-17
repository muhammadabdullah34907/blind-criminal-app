import cv2
from inference_sdk import InferenceHTTPClient
import numpy as np

from roboflow import Roboflow

rf = Roboflow(api_key="3icBo65l1BV3g5wb0s1Y")
project = rf.workspace().project("gun-d8mga/2")
model = project.version(2).model

job_id, signed_url = model.predict_video(
    "test.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

# # Initialize the RoboFlow inference client
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="3icBo65l1BV3g5wb0s1Y"
# )

# # Define your model ID
# model_id = "gun-d8mga/2"

# # Open video capture (use 0 for webcam or provide a video file path)
# video_path = 'test.mp4'  # Replace with your video file path or use 0 for webcam
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# while cap.isOpened():
#     # Read frame-by-frame
#     success, frame = cap.read()
#     if not success:
#         break
    
#     # Convert frame to JPG format (since the client expects image data in this format)
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     img_bytes = img_encoded.tobytes()
    
#     # Perform inference
#     result = CLIENT.infer(img_bytes, model_id=model_id)
    
#     # Extract predictions
#     predictions = result['predictions']

#     # Draw bounding boxes for detected objects
#     for prediction in predictions:
#         x_center = prediction['x']
#         y_center = prediction['y']
#         width = prediction['width']
#         height = prediction['height']
#         confidence = prediction['confidence']
#         class_name = prediction['class']

#         # Calculate top-left and bottom-right coordinates
#         x1 = int(x_center - width / 2)
#         y1 = int(y_center - height / 2)
#         x2 = int(x_center + width / 2)
#         y2 = int(y_center + height / 2)

#         # Draw bounding box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame with bounding boxes
#     cv2.imshow('Detections', frame)
    
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and destroy all windows
# cap.release()
# cv2.destroyAllWindows()
