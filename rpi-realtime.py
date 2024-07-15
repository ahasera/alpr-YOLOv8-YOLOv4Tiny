import cv2
import numpy as np
import easyocr
import pytesseract
from picamera2 import Picamera2
import time
import argparse 
import os
import threading
from queue import Queue, Empty
#Picamera2.set_logging(Picamera2.DEBUG)
#uncomment line above for additional DEBUG if you have picamera problems

""""
The yolov4-tiny.cfg IS NOT the default one. It has been adapted to correctly match my model classes. 
As there is only 1 class, I updated the filters using the formula given in the Darknet Repo (YOLOV4Tiny Darknet): (nbclass+5)x3 which in my case gets us to 18.
"""
"""
Command-line Argument Parsing
-----------------------------
Available options are:
- Exporting annotated images to path of yout choice
- Skipping frames to improve performance (not recommended)
- Choosing between Tesseract and EasyOCR for text recognition (tesseract recommended for performance, EasyOCR for accuracy)
- Enabling multi-threading for improved processing speed (I'd recommend to use it only if you plan to export frames as asynchronous image processing will lead annotations to pop and disappear)
- Disabling OCR to focus solely on plate detection
"""
parser = argparse.ArgumentParser(description="Real-time object detection and OCR with YOLOv4-tiny and configurable OCR.")
parser.add_argument('--export', type=str, help='Directory to export annotated images.')
parser.add_argument('--skip', action='store_true', help='Skip frames (process every 10th frame)') # can change this value below 
parser.add_argument('--tesseract', action='store_true', help='Use Tesseract instead of EasyOCR')
parser.add_argument('--multi-thread', action='store_true', help='Enable multi-threading for processing')
parser.add_argument('--no-ocr', action='store_true', help='Disable OCR and only perform detection')
args = parser.parse_args()

"""
YOLO Model Configuration
------------------------
This section loads the YOLOv4-tiny model for license plate detection.
The model has been customized for single-class detection (license plates),
with the configuration adjusted accordingly.
"""
config_path = 'cfg/yolov4-tiny.cfg'
weights_path = 'models/yolov4-tiny/custom-yolov4-tiny-detector_last.weights'
names_path = 'cfg/obj.names'
CONFIDENCE_THRESHOLD = 0.3 # higher or lower the confidence if you want more detections, less accurate or revert (only plate detections, OCR accuracy is not meant to be modified for now)
# Load the network weights and classes
net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, 'r') as f:
    classes = f.read().splitlines()


"""
OCR Initialization
------------------
Initialize the Optical Character Recognition (OCR) system based on user preferences. Mandatory for plates annotations
If OCR is enabled, it sets up EasyOCR by default, but --tesseract is recommended for better performance on less powerful devices.
"""
if not args.no_ocr:
    if args.tesseract:
        ocr = pytesseract
    else:
        ocr = easyocr.Reader(['en'])

"""
Camera Initialization
---------------------
Set up the Raspberry Pi camera using the Picamera2 library.
Higher resolutions can help you detect from farther but will lower performance significantly
"""
picam2 = Picamera2() 
config = picam2.create_still_configuration(main={"size": (640, 480)}, buffer_count=8) # change res and buffer count there, lores removed because not relevant
picam2.configure(config)
picam2.start()

time.sleep(2)  # Allow camera to adjust

"""
Image Validation
----------------
This function checks if an image is valid and suitable for processing.
It ensures that the image is not None, has a non-zero size, and has at least
two dimensions (height and width). This validation is crucial to prevent
crashes and errors when processing invalid or corrupted image data.

Parameters:
    image (numpy.ndarray): The input image to validate.

Returns:
    bool: True if the image is valid, False otherwise.
"""
def is_valid_image(image):
    return image is not None and image.size != 0 and len(image.shape) >= 2 and image.shape[0] > 0 and image.shape[1] > 0


    
"""
Image Preprocessing for OCR
---------------------------
This function prepares the detected license plate image for OCR processing.
It converts the image to grayscale, applies thresholding, and reduces noise
to improve OCR accuracy. The function now includes error handling and
image validation to ensure robust operation.

Parameters:
    image (numpy.ndarray): The input image containing the license plate.

Returns:
    numpy.ndarray: The preprocessed image ready for OCR, or None if preprocessing fails.
"""
def preprocess_for_ocr(image):
    if not is_valid_image(image):
        print("Invalid image received in preprocess_for_ocr")
        return None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(binary)
        return denoised
    except cv2.error as e:
        print(f"Error in preprocess_for_ocr: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in preprocess_for_ocr: {e}")
        return None

"""
Frame Processing
----------------
This function is the core of the license plate detection system. It performs the following steps:
1. Validates the input frame.
2. Prepares the input frame for the YOLO model.
3. Runs the YOLO model to detect license plates.
4. Applies non-maximum suppression to filter overlapping detections.
5. If OCR is enabled, it performs text recognition on each detected plate.

The function now includes additional error checking and handling to ensure
robust operation even with potentially problematic input frames.

Parameters:
    frame (numpy.ndarray): The input frame from the camera.

Returns:
    list: A list of tuples, each containing information about a detected license plate
          (bounding box coordinates, label, confidence, OCR text, and color for visualization).
"""
def process_frame(frame):
    if not is_valid_image(frame):
        print("Invalid frame received in process_frame")
        return []

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

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

    results = []
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if isinstance(indexes, np.ndarray) and indexes.size > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)

            if not args.no_ocr:
                # Ensure the crop coordinates are within the frame boundaries
                crop_x = max(0, min(x, width - 1))
                crop_y = max(0, min(y, height - 1))
                crop_w = min(w, width - crop_x)
                crop_h = min(h, height - crop_y)
                
                if crop_w > 0 and crop_h > 0:
                    crop_img = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    if is_valid_image(crop_img):
                        preprocessed_crop = preprocess_for_ocr(crop_img)
                        if preprocessed_crop is not None:
                            if args.tesseract:
                                ocr_text = ocr.image_to_string(preprocessed_crop).strip()
                            else:
                                ocr_result = ocr.readtext(preprocessed_crop)
                                ocr_text = " ".join([res[1] for res in ocr_result])
                        else:
                            ocr_text = "OCR Failed"
                    else:
                        ocr_text = "Invalid crop"
                else:
                    ocr_text = "Invalid crop size"
            else:
                ocr_text = ""

            results.append((x, y, w, h, label, confidence, ocr_text, color))

    return results

"""
Multi-threaded Frame Processing
-------------------------------
This function runs in a separate thread when multi-threading is enabled.
It continuously processes frames from the input queue and puts the results
in the output queue. This allows for parallel processing of frames, potentially
improving overall system performance. Keep in mind though that as it processes images asynchronously, 
annotations can pop and disappear real fast. It is recommended to use it with the export option 
allowing for best overall performance and export in another thread detected plates on vehicles, 
snapshotted in the folder of your choice. 

The function now includes additional error handling and logging to help
identify and diagnose issues in the multi-threaded processing pipeline.

Parameters:
    input_queue (Queue): Queue containing frames to be processed.
    output_queue (Queue): Queue to store processed results.
    stop_event (threading.Event): Event to signal when to stop processing.
"""
def process_frame_thread(input_queue, output_queue, stop_event):
    while not stop_event.is_set():
        try:
            timestamp, frame = input_queue.get(timeout=1)
            results = process_frame(frame)
            output_queue.put((timestamp, frame, results))
            input_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            print(f"Error in processing thread: {e}")

def export_thread(export_queue, export_dir, stop_event):
    while not stop_event.is_set() or not export_queue.empty():
        try:
            frame, timestamp, results = export_queue.get(timeout=1)
            
            # Save full annotated image
            full_filename = os.path.join(export_dir, f"annotated_{timestamp}.png")
            cv2.imwrite(full_filename, frame)
            
            # Save cropped license plates
            cropped_dir = os.path.join(export_dir, "cropped")
            
            # Assure-toi que le dossier cropped_dir existe
            os.makedirs(cropped_dir, exist_ok=True)
            
            for idx, (x, y, w, h, label, confidence, _, _) in enumerate(results):
                crop = frame[y:y+h, x:x+w]
                crop_filename = os.path.join(cropped_dir, f"crop_{timestamp}_{idx}.png")
                cv2.imwrite(crop_filename, crop)
            
        except Empty:
            continue
        except Exception as e:
            print(f"Error in export thread: {e}")
            
frame_count = 0
start_time = time.time()
stop_event = threading.Event()

if args.multi_thread:
    input_queue = Queue(maxsize=2) 
    output_queue = Queue(maxsize=2)  # Reduced queue size to 2 
    processing_thread = threading.Thread(target=process_frame_thread, args=(input_queue, output_queue, stop_event))
    processing_thread.start()

if args.export:
    os.makedirs(args.export, exist_ok=True)
    export_queue = Queue()
    export_thread = threading.Thread(target=export_thread, args=(export_queue, args.export, stop_event))
    export_thread.start()
"""
Main Loop
-------------------
This is the primary execution loop of rpi-realtime
It continuously captures frames from the camera, processes them (either in
the main thread or a separate thread), and displays the results annotated.

Key features:
- Supports frame skipping for performance optimization
- Handles multi-threaded processing if enabled
- Manages result synchronization in multi-threaded mode
- Displays detected license plates and OCR results on the frame
- Calculates and displays FPS in term stdout 
- Supports exporting of annotated frames (useful in combi. with multi-thread)
"""
try:
    last_processed_timestamp = 0
    while True:
        frame = picam2.capture_array()
        if not is_valid_image(frame):
            print("Invalid frame captured, skipping...")
            continue
        current_timestamp = time.time()
        results = []  # Initialize results as an empty list
        
        if not args.skip or frame_count % 10 == 0:
            if args.multi_thread:
                if input_queue.full():
                    try:
                        input_queue.get_nowait()  # Remove oldest frame if queue is full
                    except Empty:
                        pass
                input_queue.put((current_timestamp, frame))
                
                try:
                    while not output_queue.empty():
                        timestamp, processed_frame, new_results = output_queue.get_nowait()
                        if timestamp > last_processed_timestamp:
                            last_processed_timestamp = timestamp
                            frame = processed_frame
                            results = new_results
                            break
                except Empty:
                    pass
            else:
                results = process_frame(frame)
            
            for (x, y, w, h, label, confidence, ocr_text, color) in results:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if not args.no_ocr:
                    ocr_y = y + h + 20
                    cv2.putText(frame, ocr_text, (x, ocr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if args.export and results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            export_queue.put((frame.copy(), timestamp, results))

        cv2.imshow("Live detection (press 'q' to quit program)", frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            print(f"FPS: {fps:.2f}")
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    stop_event.set()
    if args.multi_thread:
        processing_thread.join()
    if args.export:
        export_thread.join()
    picam2.stop()
    cv2.destroyAllWindows()