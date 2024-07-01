import cv2
import numpy as np
import easyocr
from picamera2 import Picamera2
import time
import threading
import argparse 
import os

""""
The yolov4-tiny.cfg IS NOT the default one. It has been adapted to correctly match my model classes. 
As there is only 1 class, I updated the filters using the formula given in the Darknet Repo (YOLOV4Tiny Darknet): (nbclass+5)x3 which in my case gets us to 18.
"""

parser = argparse.ArgumentParser(description="Real-time object detection and OCR with YOLOv4-tiny and EasyOCR.")
parser.add_argument('--export', type=str, help='Directory to export annotated images and OCR results.')
parser.add_argument('--no-skip', action='store_true', help='Perform on every frame without skipping. /!\ will decrease performance significantly')
args = parser.parse_args()

config_path = 'cfg/yolov4-tiny.cfg'
weights_path = 'models/yolov4-tiny/custom-yolov4-tiny-detector_last.weights'
names_path = 'cfg/obj.names'
CONFIDENCE_THRESHOLD = 0.3 # higher or lower the confidence if you want more detections, less accurate or revert 
# Load the network weights and classes
net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, 'r') as f:
    classes = f.read().splitlines()

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Initialize the raspberry pi camera module with picamera2  
picam2 = Picamera2() 
config = picam2.create_still_configuration(main={"size": (640, 480)}, lores={"size": (320, 240)}, display="lores") # change res here
picam2.preview_configuration.main.format = "XRGB8888"
picam2.configure(config)
picam2.start()

time.sleep(1)  # Allow camera to adjust

# Initialize shared variables and synchronization primitives
# lower or higher the frame_skip value to respectively lower or higher the annotation refresh rate
frame = None
results = None
lock = threading.Lock()
frame_skip = 10 if not args.no_skip else 1
frame_counter = 0
last_saved_time = 0
save_interval = 5  # Save only if more than 5 seconds have passed since the last save
last_saved_results = None


def save_results(export_dir, frame, results):
    global last_saved_time, last_saved_results
    current_time = time.time()
    if current_time - last_saved_time > save_interval and results_changed(results, last_saved_results):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        img_path = os.path.join(export_dir, f"annotated_{timestamp}.png")
        txt_path = os.path.join(export_dir, f"ocr_{timestamp}.txt")

        cv2.imwrite(img_path, frame)
        with open(txt_path, 'w') as f:
            for (x, y, w, h, label, confidence, ocr_text, color) in results:
                f.write(f"Label: {label}, Confidence: {confidence}\n")
                f.write(f"OCR Text: {ocr_text}\n\n")

        last_saved_time = current_time
        last_saved_results = results

def results_changed(new_results, last_results):
    if last_results is None:
        return True
    if len(new_results) != len(last_results):
        return True
    for new, last in zip(new_results, last_results):
        if new[:6] != last[:6]:
            return True
    return False

def preprocess_for_ocr(image):
    # grey level conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # denoise
    denoised = cv2.fastNlMeansDenoising(binary)
    return denoised            

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
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Parse YOLO outputs
        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) to prevent multi-boxes overlapping eachother
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
                    preprocessed_crop = preprocess_for_ocr(crop_img)
                    ocr_result = reader.readtext(preprocessed_crop)
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

    if current_results:  # Ensure current_results is not None
        for (x, y, w, h, label, confidence, ocr_text, color) in current_results:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, label + " " + str(round(confidence, 2)), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            ocr_text_position_y = y + h + 20
            if ocr_text_position_y + 30 > display_frame.shape[0]:
                ocr_text_position_y = y - 30
            cv2.putText(display_frame, ocr_text, (x, ocr_text_position_y), cv2.FONT_HERSHEY_PLAIN, 4, color, 2)

        if args.export:
            save_results(args.export, display_frame, current_results)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Annotated Image", display_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1
    if frame_counter % frame_skip == 0:
        with lock:
            frame = display_frame
picam2.stop()
cv2.destroyAllWindows()