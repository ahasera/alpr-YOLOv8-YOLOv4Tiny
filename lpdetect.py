import os
import cv2
from ultralytics import YOLO
import easyocr
from tqdm import tqdm
import time
import gc
import multiprocessing as mp
import argparse
from preprocessing import preprocess_image
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import re 
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse arguments
parser = argparse.ArgumentParser(description='Vehicle and License Plate Detection with optional preprocessing')
parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing of images')
parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for detection')
parser.add_argument('--input-size', type=int, default=416, help='Input size for YOLO models')
parser.add_argument('--output-format', choices=['csv', 'json'], default='csv', help="Output format for results")
parser.add_argument('--use-gpu', action='store_true', help='Use GPU for processsing if available')
args = parser.parse_args()


"""
YoloV8 model initialisation 
First one is pretrained coco model 
Second one is yours or mine, for license plate object detection. 
"""
try:
    model_vehicle = YOLO('models/yolov8/yolov8n.pt')
    model_plate = YOLO('models/yolov8/best.pt')  # YOLOv8 model for plate detection
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
    reader = easyocr.Reader(['en'], gpu=args.use_gpu)
except Exception as e:
    logging.error(f"Error initializing EasyOCR: {e}")
    exit(1)
    
# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def apply_conversion(text, conversion_dict):
    return ''.join(conversion_dict.get(char, char) for char in text)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    return wrapper

def is_valid_plate(text):
    # Add your country-specific license plate format validation here
    # This is a simple example, adjust as needed
    pattern = r'^[A-Z0-9]{5,8}$'
    return bool(re.match(pattern, text))

def adjust_text_position(image, text_x, text_y, text, font_scale, thickness, used_positions):
    """
    Adjust text position to avoid overlapping with the image boundaries and other texts.
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    image_height, image_width = image.shape[:2]
    
    # Adjust if the text goes beyond the right edge of the image
    if text_x + text_width > image_width:
        text_x = image_width - text_width - 10
    
    # Adjust if the text goes above the top edge of the image
    if text_y - text_height < 0:
        text_y = text_height + 10
    
    # Check and adjust for overlapping with used positions
    while any(abs(text_x - used_x) < text_width and abs(text_y - used_y) < text_height for used_x, used_y in used_positions):
        text_y += text_height + 10
        if text_y + text_height > image_height:
            text_y = text_height + 10
            text_x += text_width + 10
            if text_x + text_width > image_width:
                text_x = 10
    
    used_positions.append((text_x, text_y))
    return text_x, text_y
"""
detect_and_recognize will first : 

    1. Read the image from the provided image_path.
    2. Optionally preprocess the image.
    3. Use the YOLOv8 model trained on the COCO dataset to detect vehicles in the image. If no vehicles are detected, proceed to step 5 using the full image.
    4. For each detected vehicle, crop the vehicle region from the image.
    5. Use the specialized YOLOv8 model to detect license plates within the cropped vehicle image.
    6. For each detected license plate, crop the license plate region from the vehicle image.
    7. Save the cropped license plate image to the cropped folder.
    8. Use EasyOCR to perform optical character recognition (OCR) on the cropped license plate image.
    9. Annotate the original image with bounding boxes around detected vehicles and license plates, and add the recognized text with confidence scores.
    10. Adjust the text position to avoid overlapping with other annotations and ensure readability.
    11. Save the annotated image to the specified output folder.
    12. Log the results of the OCR, including the recognized text and confidence scores, to the provided log file.
    13. Handle and log any exceptions that occur during the processing of the image.
"""
def format_inference_time(start_time):
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    return f"{inference_time:.2f}"

def detect_and_recognize(image_path, output_folder, cropped_folder, log_file, args):
    try:
        # Read and optionally preprocess the image with --preprocess
        if args.preprocess:
            image = preprocess_image(image_path)
        else:
            image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        used_positions = []
        
        with open(log_file, 'a') as log:
            log.write(f"Processing {image_path}:\n")
            # Measure vehicle detection time
            start_time = time.perf_counter()
            results_vehicle = model_vehicle(image, conf=args.confidence, imgsz=args.input_size)
            vehicle_time = (time.perf_counter() - start_time) * 1000 
            log.write(f"Vehicle detection time: {vehicle_time} ms\n")
            vehicle_detected = False
            
            for result in results_vehicle:
                for bbox in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = bbox
                    if int(class_id) == 2:  # We only specify car class in the coco model for object detection
                        vehicle_detected = True
                        # crop image to keep vehicle
                        vehicle = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Measure plate detection time
                        start_time = time.perf_counter()
                        results_plate = model_plate(vehicle, conf=args.confidence, imgsz=args.input_size)
                        plate_time = (time.perf_counter() - start_time) * 1000
                        log.write(f"Plate detection time: {plate_time:.2f} ms\n")
                        
                        for result_plate in results_plate:
                            for bbox_plate in result_plate.boxes.data.tolist():
                                px1, py1, px2, py2, pscore, pclass_id = bbox_plate
                                # crop to get license plate 
                                plate = vehicle[int(py1):int(py2), int(px1):int(px2)]
                                
                                # Save the cropped license plate image
                                plate_filename = os.path.join(cropped_folder, f"plate_{os.path.basename(image_path)}")
                                cv2.imwrite(plate_filename, plate)
                                
                                # Measure OCR time
                                start_time = time.perf_counter()
                                ocr_result = reader.readtext(plate)
                                ocr_time = (time.perf_counter() - start_time) * 1000
                                log.write(f"OCR time: {ocr_time:.2f} ms\n")
                                
                                """
                                this for loop will annotate each image
                                """
                                for (bbox_ocr, text, prob) in ocr_result:
                                    # prevent text from overlapping 
                                    text_x_position = int(px1) + int(x1)
                                    text_y_position = int(py1) + int(y1) - 10
                                    if text_y_position < 10:
                                        text_y_position = int(py2) + int(y1) + 20
                                    text_x_position, text_y_position = adjust_text_position(
                                        image, text_x_position, text_y_position, f"{text} ({prob:.2f})", 0.9, 2, used_positions)
                                    
                                    cv2.rectangle(image, (int(px1) + int(x1), int(py1) + int(y1)),
                                                  (int(px2) + int(x1), int(py2) + int(y1)), (0, 255, 0), 2)
                                    cv2.putText(image, f"{text} ({prob:.2f})", (text_x_position, text_y_position),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                    
                                    log.write(f"  Detected plate: {text} with confidence: {prob}\n")
                                    
                                # Annotate on the original image
                                cv2.rectangle(image, (int(px1) + int(x1), int(py1) + int(y1)),
                                              (int(px2) + int(x1), int(py2) + int(y1)), (255, 0, 0), 2)
                                
                        # vehicule rectangle annotation 
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            if not vehicle_detected: # will allow to run the second model if the first one did not detect a vehicle. Can cause problm in some situations with many objects nearby
                start_time = time.time()
                results_plate = model_plate(image, conf=args.confidence, imgsz=args.input_size)
                plate_time = format_inference_time(start_time)
                log.write(f"Plate detection time (full image): {plate_time} ms\n")
                
                for result_plate in results_plate:
                    for bbox_plate in result_plate.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = bbox_plate
                        # Crop to get the license plate
                        plate = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Save the cropped license plate image
                        plate_filename = os.path.join(cropped_folder, f"plate_{os.path.basename(image_path)}")
                        cv2.imwrite(plate_filename, plate)
                        
                        # Measure OCR time
                        start_time = time.time()
                        ocr_result = reader.readtext(plate)
                        ocr_time = format_inference_time(start_time)
                        log.write(f"OCR time: {ocr_time} ms\n")
                        
                        # OCR annotation on the input image
                        for (bbox_ocr, text, prob) in ocr_result:
                            text_x_position = int(x1)
                            text_y_position = int(y1) - 10
                            if text_y_position < 10:
                                text_y_position = int(y2) + 20
                            text_x_position, text_y_position = adjust_text_position(
                                image, text_x_position, text_y_position, f"{text} ({prob:.2f})", 0.9, 2, used_positions)
                            
                            cv2.rectangle(image, (int(x1), int(y1)),
                                          (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(image, f"{text} ({prob:.2f})", (text_x_position, text_y_position),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            log.write(f"  Detected plate: {text} with confidence: {prob}\n")
                            
                        # detection annotation on input image
                        cv2.rectangle(image, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (255, 0, 0), 2)
                        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        annotated_image_path = os.path.join(output_folder, "annotated_" + os.path.basename(image_path))
        cv2.imwrite(annotated_image_path, image)
        logging.info(f"Annotated image saved to: {annotated_image_path}")

        gc.collect()
    
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")

def process_image(args):
    image_path, output_folder, cropped_folder, log_file, detection_args = args
    detect_and_recognize(image_path, output_folder, cropped_folder, log_file, detection_args)
    return image_path

def process_folder(input_folder, output_folder, cropped_folder, log_file, args):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in os.listdir(input_folder) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)
    
    total_images = len(images)
    processed_images = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = []
        for image_path in images:
            full_image_path = os.path.join(input_folder, image_path)
            futures.append(executor.submit(process_image, (full_image_path, output_folder, cropped_folder, log_file, args)))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            processed_images += 1
            logging.info(f"Processed {future.result()}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Generate summary report
    with open('summary_report.txt', 'w') as report:
        report.write("License Plate Detection Summary Report\n")
        report.write("======================================\n\n")
        report.write(f"Total images processed: {total_images}\n")
        report.write(f"Total processing time: {total_time:.2f} seconds\n")
        if total_images > 0:
            report.write(f"Average time per image: {total_time/total_images:.2f} seconds\n")
        else:
            report.write("No images were processed.\n")
    
    logging.info("Processing completed. Summary report generated in summary_report.txt")

input_folder = 'data/input'
output_folder = 'data/output'
cropped_folder = 'data/cropped'
log_file = 'detection_log.txt'

if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    
process_folder(input_folder, output_folder, cropped_folder, log_file, args)