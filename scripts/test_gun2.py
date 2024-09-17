from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="3icBo65l1BV3g5wb0s1Y"
)

result = CLIENT.infer('test.jpg', model_id="gun-d8mga/2")
print(result)
print("done")
import cv2
import json
import os

# Sample response from RoboFlow model API
response = {
    'inference_id': '32617318-eb5f-4b69-9be4-dd1e5c8681e7',
    'time': 0.05716037199999846,
    'image': {'width': 1920, 'height': 1080},
    'predictions': [
        {'x': 892.5, 'y': 588.75, 'width': 1016.25, 'height': 960.0, 'confidence': 0.8026094436645508, 'class': 'PERSON', 'class_id': 1, 'detection_id': '6f11fc55-80c1-470b-a33a-ce3bbe1e4f4a'},
        {'x': 1273.125, 'y': 281.25, 'width': 345.0, 'height': 281.25, 'confidence': 0.6573598384857178, 'class': '0', 'class_id': 0, 'detection_id': 'b83ab52b-4811-410c-882d-17b42ea7bd3c'}
    ]
}
response=result



# Load image
img_name = 'test.jpg'  # Replace with your image name
img_path = os.path.join(img_name)
frame = cv2.imread(img_path)

# Draw bounding boxes for detected objects
for prediction in response['predictions']:
    x_center, y_center = prediction['x'], prediction['y']
    width, height = prediction['width'], prediction['height']
    confidence = prediction['confidence']
    class_name = prediction['class']

    # Calculate top-left and bottom-right coordinates
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Detections', frame)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
