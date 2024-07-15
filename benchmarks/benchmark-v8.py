import os
import cv2
from ultralytics import YOLO
import easyocr
import pytesseract
import time
import argparse
import logging
import pandas as pd
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse arguments
parser = argparse.ArgumentParser(description='Benchmark Vehicle and License Plate Detection')
parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for detection')
parser.add_argument('--input-size', type=int, default=416, help='Input size for YOLO models')
parser.add_argument('--use-gpu', action='store_true', help='Use GPU for processing if available')
args = parser.parse_args()

# Initialize models and OCR readers
try:
    model_vehicle = YOLO('../models/yolov8/yolov8n.pt')
    model_plate = YOLO('../models/yolov8/best.pt')
    if args.use_gpu and torch.cuda.is_available():
        model_vehicle.to('cuda')
        model_plate.to('cuda')
        logging.info("Using GPU for YOLO models")
    else:
        logging.info("Using CPU for YOLO models")
except Exception as e:
    logging.error(f"Error while loading YOLO models: {e}")
    exit(1)

try:
    easyocr_reader = easyocr.Reader(['en'], gpu=args.use_gpu)
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
def detect_vehicle(image, model, confidence, input_size):
    results = model(image, conf=confidence, imgsz=input_size)
    return results

@measure_time
def detect_plate(image, model, confidence, input_size):
    results = model(image, conf=confidence, imgsz=input_size)
    return results

@measure_time
def perform_easyocr(image, reader):
    return reader.readtext(image)

@measure_time
def perform_pytesseract(image):
    return pytesseract.image_to_string(image)

def process_image(image_path, args, ocr_method='easyocr'):
    image = cv2.imread(image_path)
    
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return [{
            'image': os.path.basename(image_path),
            'model': 'YOLOv8',
            'vehicle_time': 0,
            'plate_time': 0,
            'ocr_time': 0,
            'ocr_result': 'Failed to load image',
            'confidence': 0,
            'ocr_method': ocr_method
        }]

    results = []

    # Detect vehicle
    vehicle_results, vehicle_time = detect_vehicle(image, model_vehicle, args.confidence, args.input_size)
    
    vehicle_detected = False
    for vehicle_result in vehicle_results:
        for bbox in vehicle_result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = bbox
            if int(class_id) == 2:  # Car class
                vehicle_detected = True
                vehicle = image[int(y1):int(y2), int(x1):int(x2)]
                
                # Detect plate in vehicle
                plate_results, plate_time = detect_plate(vehicle, model_plate, args.confidence, args.input_size)
                
                if len(plate_results) == 0:
                    results.append({
                        'image': os.path.basename(image_path),
                        'model': 'YOLOv8',
                        'vehicle_time (in ms)': vehicle_time,
                        'plate_time (in ms)': plate_time,
                        'ocr_time (in ms)': 0,
                        'ocr_result': 'No plate detected in vehicle',
                        'confidence': score,
                        'ocr_method': ocr_method
                    })
                else:
                    for plate_result in plate_results:
                        for plate_bbox in plate_result.boxes.data.tolist():
                            px1, py1, px2, py2, plate_score, plate_class_id = plate_bbox
                            plate = vehicle[int(py1):int(py2), int(px1):int(px2)]
                            
                            if ocr_method == 'easyocr':
                                ocr_result, ocr_time = perform_easyocr(plate, easyocr_reader)
                                ocr_text = " ".join([res[1] for res in ocr_result])
                            else:  # pytesseract
                                ocr_text, ocr_time = perform_pytesseract(plate)
                            
                            results.append({
                                'image': os.path.basename(image_path),
                                'model': 'YOLOv8',
                                'vehicle_time': vehicle_time,
                                'plate_time': plate_time,
                                'ocr_time': ocr_time,
                                'ocr_result': ocr_text,
                                'confidence': plate_score,
                                'ocr_method': ocr_method
                            })

    # If no vehicle detected, try to detect plate in the whole image
    if not vehicle_detected:
        plate_results, plate_time = detect_plate(image, model_plate, args.confidence, args.input_size)
        
        if len(plate_results) == 0:
            results.append({
                'image': os.path.basename(image_path),
                'model': 'YOLOv8',
                'vehicle_time': vehicle_time,
                'plate_time': plate_time,
                'ocr_time': 0,
                'ocr_result': 'No plate detected in full image',
                'confidence': 0,
                'ocr_method': ocr_method
            })
        else:
            for plate_result in plate_results:
                for plate_bbox in plate_result.boxes.data.tolist():
                    x1, y1, x2, y2, plate_score, plate_class_id = plate_bbox
                    plate = image[int(y1):int(y2), int(x1):int(x2)]
                    
                    if ocr_method == 'easyocr':
                        ocr_result, ocr_time = perform_easyocr(plate, easyocr_reader)
                        ocr_text = " ".join([res[1] for res in ocr_result])
                    else:  # pytesseract
                        ocr_text, ocr_time = perform_pytesseract(plate)
                    
                    results.append({
                        'image': os.path.basename(image_path),
                        'model': 'YOLOv8',
                        'vehicle_time': vehicle_time,
                        'plate_time': plate_time,
                        'ocr_time': ocr_time,
                        'ocr_result': ocr_text,
                        'confidence': plate_score,
                        'ocr_method': ocr_method
                    })

    return results

def benchmark_folder(input_folder, args):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in os.listdir(input_folder) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    all_results = []
    
    for image_path in images:
        full_image_path = os.path.join(input_folder, image_path)
        
        # Process with EasyOCR
        results = process_image(full_image_path, args, ocr_method='easyocr')
        all_results.extend(results)
        
        # Process with Pytesseract
        results = process_image(full_image_path, args, ocr_method='pytesseract')
        all_results.extend(results)
    
    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(all_results)
    df.to_csv('benchmark_results_yolov8.csv', index=False)
    logging.info(f"Benchmark results saved to benchmark_results_yolov8.csv. Total rows: {len(df)}")
    
    
    expected_rows = len(images) * 2
    if len(df) != expected_rows:
        logging.warning(f"Expected {expected_rows} rows, but got {len(df)} rows. Some images might have multiple detections, or some may have failed.")

if __name__ == "__main__":
    input_folder = '../data/input'
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    
    benchmark_folder(input_folder, args)