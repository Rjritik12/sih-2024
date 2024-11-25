import cv2
from ultralytics import YOLO
import numpy as np
import yt_dlp

def start_livestream(stream_url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'noplaylist': True,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(stream_url, download=False)
            video_url = info_dict['formats'][0]['url']  # Get the video URL
    except Exception as e:
        print(f"Error extracting video URL: {e}")
        return

   
    stream = cv2.VideoCapture(stream_url)

    # Check if the video stream opened successfully
    if not stream.isOpened():
        print("Error: Could not open video stream.")
        return

    # Create a window to display the video frame
    cv2.namedWindow('FRAME')

    # Load the YOLO model
    model = YOLO("../YOLO-Weights/yolov8n.pt")
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    while True:
        # Capture the next video frame
        success, img = stream.read()

        # If the frame is not empty, perform object detection
        if success:
            # Perform detection using YOLO
            results = model(img, stream=True)

            # Initialize counter for detected persons
            count = 0

            # Loop through results and draw bounding boxes
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = box.conf[0].item()
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Count the number of people detected
                    if class_name == 'person':
                        count += 1

            # Display the number of people detected
            cv2.putText(img, f'Count: {count}', (50, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))

            # Add a rectangle and label for detection
            cv2.rectangle(img, (10, 10), (200, 50), (0, 0, 255), 2)
            cv2.putText(img, "Person Detection", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display the video frame
            cv2.imshow('FRAME', img)

            # If the user presses the 'Esc' key, break out of the loop
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            print("Error: Could not read frame.")
            break

    # Release the video stream
    stream.release()

    # Destroy all windows
    cv2.destroyAllWindows()
if __name__ == "__main__":
    stream_url = input("Please enter the stream URL: ")
    start_livestream(stream_url)

    
