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
# Parse arguments
parser = argparse.ArgumentParser(description='Vehicle and License Plate Detection with optional preprocessing')
parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing of images')
args = parser.parse_args()

"""
YoloV8 model initialisation 
First one is pretrained coco model 
Second one is yours or mine, for license plate object detection. 
"""
try:
    model_vehicle = YOLO('models/yolov8/yolov8n.pt')
except Exception as e:
    print(f"Error while loading yolov8n base model: {e}")
    exit(1)

try:
    model_plate = YOLO('models/yolov8/best.pt')  # YoloV8 model for plate detection 
except Exception as e:
    print(f"Error while loading license-plate model: {e}")
    exit(1)
    
reader = easyocr.Reader(['en'])

"""
The adjust_text_position is here to prevent text going over each other 
--> made to simplify the reading of the easyocr results, as they can be multiple results for 1 plate (with additional characters being recognized, in different orders)
"""

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
    3. Use the YOLOv8 model trained on the COCO dataset to detect vehicles in the image.
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

def detect_and_recognize(image_path, output_folder, cropped_folder, log_file, preprocess):
    try:
        # Read and optionally preprocess the image with --preprocess
        if preprocess:
            image = preprocess_image(image_path)
        else:
            image = cv2.imread(image_path)
        
        used_positions = []
        
        with open(log_file, 'a') as log:
            log.write(f"Processing {image_path}:\n")
            # Measure vehicle detection time
            start_time = time.time()
            results_vehicle = model_vehicle(image)
            vehicle_time = format_inference_time(start_time)
            log.write(f"Vehicle detection time: {vehicle_time} ms\n")
            vehicle_detected = False
            
            for result in results_vehicle:
                for bbox in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = bbox
                    if class_id == 2:  # We only specify car class in the coco model for object detection
                        vehicle_detected = True
                        # crop image to keep vehicle
                        vehicle = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Measure plate detection time
                        start_time = time.time()
                        results_plate = model_plate(vehicle)
                        plate_time = format_inference_time(start_time)
                        log.write(f"Plate detection time: {plate_time} ms\n")
                        
                        for result_plate in results_plate:
                            for bbox_plate in result_plate.boxes.data.tolist():
                                px1, py1, px2, py2, pscore, pclass_id = bbox_plate
                                # crop to get license plate 
                                plate = vehicle[int(py1):int(py2), int(px1):int(px2)]
                                
                                # Save the cropped license plate image
                                plate_filename = os.path.join(cropped_folder, f"plate_{os.path.basename(image_path)}")
                                cv2.imwrite(plate_filename, plate)
                                
                                # Measure OCR time
                                start_time = time.time()
                                ocr_result = reader.readtext(plate)
                                ocr_time = format_inference_time(start_time)
                                log.write(f"OCR time: {ocr_time} ms\n")
                                
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
                                    
                                    log.write(f"  Detected plate: {text} with confidence: {prob}\n") # write console results to logs
                                    
                                # Annotate on the original image
                                cv2.rectangle(image, (int(px1) + int(x1), int(py1) + int(y1)),
                                              (int(px2) + int(x1), int(py2) + int(y1)), (255, 0, 0), 2)
                                
                        # vehicule rectangle annotation 
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            if not vehicle_detected: # will allow to run the second model if the first one did not detect a vehicle. Can cause problm in some situations with many objects nearby
                start_time = time.time()
                results_plate = model_plate(image)
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
        print(f"Annotated image saved to: {annotated_image_path}")

        time.sleep(1)
        gc.collect()
    
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error processing {image_path}: {str(e)}\n")
        print(f"Error processing {image_path}: {str(e)}")

def process_image(image_path, output_folder, cropped_folder, log_file, preprocess):
    detect_and_recognize(image_path, output_folder, cropped_folder, log_file, preprocess)
    gc.collect() 

"""
process_folder will process the images in the input folder using the multiprocessing and will log the inference times. 
"""
def process_folder(input_folder, output_folder, cropped_folder, log_file, preprocess):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in os.listdir(input_folder) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)
    
    with mp.Pool(mp.cpu_count()) as pool:
        for image_path in tqdm(images, desc="Processing images"):
            full_image_path = os.path.join(input_folder, image_path)
            pool.apply_async(process_image, args=(full_image_path, output_folder, cropped_folder, log_file, preprocess))
        pool.close()
        pool.join()
    
input_folder = 'data/input'
output_folder = 'data/output'
cropped_folder = 'data/cropped'
log_file = 'detection_log.txt'

if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    
process_folder(input_folder, output_folder, cropped_folder, log_file, args.preprocess)