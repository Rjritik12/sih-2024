import cv2
import cvlib as cv
import winsound

# Read the YouTube video stream
stream = cv2.VideoCapture('https://www.youtube.com/live/LKse2ijG6QE?si=fEmxeHM1oWKnQM98')

# Create a window to display the video frame
cv2.namedWindow('FRAME')

# Initialize the object detection model 
model = cv.detect_common_objects

# Initialize the counter for the number of people detected
count = 0

while True:
    # Capture the next video frame
    success, frame = stream.read()

    # If the frame is not empty, resize it and perform object detection
    if success:
        frame = cv2.resize(frame, (640, 480))
        bbox, label, conf = model(frame)

        # Draw bounding boxes around the detected objects
        frame = cv2.draw_bbox(frame, bbox, label, conf)

        # Count the number of people detected
        c = label.count('person')

        # Display the number of people detected
        cv2.putText(frame, str(c), (50, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))

        # Play a beep sound if more than one person is detected
        if label == 'person' and c > 1:
            winsound.siren(1000, 500)

        # Add the following code to the above code
        cv2.rectangle(frame, (10, 10), (100, 30), (0, 0, 255), 2)
        cv2.putText(frame, "Person Detection", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the video frame
        cv2.imshow('FRAME', frame)

    # If the user presses the 'Esc' key, break out of the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video stream
stream.release()

# Destroy all windows
cv2.destroyAllWindows()
