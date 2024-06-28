import cv2
import numpy as np
import easyocr
from picamera2 import Picamera2
import time
import threading

""""
The yolov4-tiny.cfg IS NOT the default one. It has been adapted to correctly match my model classes. 
As there is only 1 class, I updated the filters using the formula given in the Darknet Repo (YOLOV4Tiny Darknet): (nbclass+5)x3 which in my case gets us to 18.
"""
config_path = 'cfg/yolov4-tiny.cfg'
weights_path = 'models/yolov4-tiny/custom-yolov4-tiny-detector_last.weights'
names_path = 'cfg/obj.names'

# Load the network weights and classes
net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, 'r') as f:
    classes = f.read().splitlines()

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Initialize the raspberry pi camera module with picamera2  
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (1280, 720)}, lores={"size": (640, 480)}, display="lores")
picam2.preview_configuration.main.format = "RGB888"
picam2.configure(config)
picam2.start()

time.sleep(1)  # Allow camera to adjust

# Initialize shared variables and synchronization primitives
# lower or higher the frame_skip value to respectively lower or higher the annotation refresh rate
frame = None
results = None
lock = threading.Lock()
frame_skip = 5 
frame_counter = 0

# Frame processing thread
def process_frame():
    global frame, results
    while True:
        with lock:
            if frame is None:
                continue
            img = frame.copy()
        height, width, _ = img.shape

        # Pre-process image and perform YOLO inference
        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Parse YOLO outputs
        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        new_results = []
        if class_ids:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if indexes:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = (0, 255, 0)

                    # Perform OCR on detected box
                    crop_img = img[y:y+h, x:x+w]
                    ocr_result = reader.readtext(crop_img)
                    ocr_text = " ".join([res[1] for res in ocr_result])

                    new_results.append((x, y, w, h, label, confidence, ocr_text, color))

        with lock:
            results = new_results

# Start frame processing thread
thread = threading.Thread(target=process_frame)
thread.daemon = True
thread.start()

prev_time = time.time()

# Main loop for capturing and displaying frames
while True:
    start_time = time.time()
    with lock:
        frame = picam2.capture_array()
        display_frame = frame.copy()
        current_results = results

    # Annotate and display results
    if current_results:
        for (x, y, w, h, label, confidence, ocr_text, color) in current_results:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, label + " " + str(round(confidence, 2)), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            ocr_text_position_y = y + h + 20
            if ocr_text_position_y + 30 > display_frame.shape[0]:
                ocr_text_position_y = y - 30
            cv2.putText(display_frame, ocr_text, (x, ocr_text_position_y), cv2.FONT_HERSHEY_PLAIN, 4, color, 2) 
            # cv2.putText(display_frame, ocr_text, (x,ocr_text_position_y), cv2.FONT_HERSHEY_PLAIN, TEXT_SIZE(INT), color, (INT))

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Annotated Image", display_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1
    if frame_counter % frame_skip == 0:
        with lock:
            frame = display_frame

picam2.stop()
cv2.destroyAllWindows()
