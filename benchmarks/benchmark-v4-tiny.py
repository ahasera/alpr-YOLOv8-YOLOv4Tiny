import os
import cv2
import numpy as np
import easyocr
import pytesseract
import time
import argparse
from preprocessing import preprocess_image
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse arguments
parser = argparse.ArgumentParser(description='Benchmark Vehicle and License Plate Detection with YOLOv4-Tiny')
parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for detection')
parser.add_argument('--input-size', type=int, default=416, help='Input size for YOLO models')
args = parser.parse_args()

# Initialize YOLOv4-Tiny model
config_path = '../cfg/yolov4-tiny.cfg'
weights_path = '../models/yolov4-tiny/custom-yolov4-tiny-detector_last.weights'
names_path = '../cfg/obj.names'

try:
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, 'r') as f:
        classes = f.read().splitlines()
    logging.info("YOLOv4-Tiny model loaded successfully")
except Exception as e:
    logging.error(f"Error while loading YOLOv4-Tiny model: {e}")
    exit(1)

try:
    easyocr_reader = easyocr.Reader(['en'])
except Exception as e:
    logging.error(f"Error initializing EasyOCR: {e}")
    exit(1)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    return wrapper

@measure_time
def detect_plate(image, net, confidence, input_size):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence_score = scores[class_id]
            if confidence_score > confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence_score))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, 0.4)
    
    if isinstance(indices, (tuple, list)):  # For older versions of OpenCV
        indices = np.array(indices).flatten()
    
    return [(boxes[i], confidences[i], class_ids[i]) for i in indices]

@measure_time
def perform_easyocr(image, reader):
    return reader.readtext(image)

@measure_time
def perform_pytesseract(image):
    return pytesseract.image_to_string(image)

def process_image(image_path, net, args, preprocess=False, ocr_method='easyocr'):
    if preprocess:
        image = preprocess_image(image_path)
    else:
        image = cv2.imread(image_path)
    
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return [{
            'image': os.path.basename(image_path),
            'model': 'YOLOv4-Tiny',
            'plate_time': 0,
            'ocr_time': 0,
            'ocr_result': 'Failed to load image',
            'confidence': 0,
            'preprocessed': preprocess,
            'ocr_method': ocr_method
        }]

    detections, plate_time = detect_plate(image, net, args.confidence, args.input_size)

    if len(detections) == 0:
        return [{
            'image': os.path.basename(image_path),
            'model': 'YOLOv4-Tiny',
            'plate_time': plate_time,
            'ocr_time': 0,
            'ocr_result': 'No plate detected',
            'confidence': 0,
            'preprocessed': preprocess,
            'ocr_method': ocr_method
        }]

    results = []
    for (box, confidence, class_id) in detections:
        x, y, w, h = box
        plate = image[y:y+h, x:x+w]
        
        if ocr_method == 'easyocr':
            ocr_result, ocr_time = perform_easyocr(plate, easyocr_reader)
            ocr_text = " ".join([res[1] for res in ocr_result])
        else:  # pytesseract
            ocr_text, ocr_time = perform_pytesseract(plate)
        
        results.append({
            'image': os.path.basename(image_path),
            'model': 'YOLOv4-Tiny',
            'plate_time': plate_time,
            'ocr_time': ocr_time,
            'ocr_result': ocr_text,
            'confidence': confidence,
            'preprocessed': preprocess,
            'ocr_method': ocr_method
        })
    
    return results

def benchmark_folder(input_folder, net, args):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in os.listdir(input_folder) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    all_results = []
    
    for image_path in images:
        full_image_path = os.path.join(input_folder, image_path)
        
        # Configuration des 4 scénarios pour chaque image
        scenarios = [
            {'preprocess': False, 'ocr_method': 'easyocr'},
            {'preprocess': True, 'ocr_method': 'easyocr'},
            {'preprocess': False, 'ocr_method': 'pytesseract'},
            {'preprocess': True, 'ocr_method': 'pytesseract'}
        ]
        
        for scenario in scenarios:
            results = process_image(full_image_path, net, args, preprocess=scenario['preprocess'], ocr_method=scenario['ocr_method'])
            all_results.extend(results)
    
    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(all_results)
    df.to_csv('benchmark_results_yolov4_tiny.csv', index=False)
    logging.info(f"Benchmark results saved to benchmark_results_yolov4_tiny.csv. Total rows: {len(df)}")
    
    expected_rows = len(images) * 4
    if len(df) != expected_rows:
        logging.warning(f"Expected {expected_rows} rows, but got {len(df)} rows. Some images might have failed processing.")

if __name__ == "__main__":
    input_folder = '../data/input'
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    
    benchmark_folder(input_folder, net, args)