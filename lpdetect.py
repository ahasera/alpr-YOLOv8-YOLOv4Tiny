import os
import cv2
from ultralytics import YOLO
import easyocr
from tqdm import tqdm
import time

"""
YoloV8 model initialisation 
First one is pretrained coco model 
Second one is yours or mine, for license plate object detection. 
"""
try:
    model_vehicle = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error while loading yolov8n base model: {e}")
    exit(1)

try:
    model_plate = YOLO('best.pt')  # YoloV8 model for plate detection 
except Exception as e:
    print(f"Error while loading license-plate model: {e}")
    exit(1)
    
reader = easyocr.Reader(['en'])

"""
The adjust_text_position is here to prevent text going over eachother 
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
    2. Use the YOLOv8 model trained on the COCO dataset to detect vehicles in the image.
    3. For each detected vehicle, crop the vehicle region from the image.
    4. Use the specialized YOLOv8 model to detect license plates within the cropped vehicle image.
    5. For each detected license plate, crop the license plate region from the vehicle image.
    6. Use EasyOCR to perform optical character recognition (OCR) on the cropped license plate image.
    7. Annotate the original image with bounding boxes around detected vehicles and license plates, and add the recognized text with confidence scores.
    8. Adjust the text position to avoid overlapping with other annotations and ensure readability.
    9. Save the annotated image to the specified output folder.
    10. Log the results of the OCR, including the recognized text and confidence scores, to the provided log file.
    11. Handle and log any exceptions that occur during the processing of the image.
"""

vehicle_inference_times = []
plate_inference_times = []
ocr_inference_times = []

def detect_and_recognize(image_path, output_folder, log_file):
    try:
        image = cv2.imread(image_path)
        
        # Measure vehicle detection time
        start_time = time.time()
        results_vehicle = model_vehicle(image)
        vehicle_time = (time.time() - start_time) * 1000  # in milliseconds
        vehicle_inference_times.append(vehicle_time)
        
        used_positions = []
        
        with open(log_file, 'a') as log:
            log.write(f"Processing {image_path}:\n")
        
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
                        plate_time = (time.time() - start_time) * 1000  # in milliseconds
                        plate_inference_times.append(plate_time)
                        
                        for result_plate in results_plate:
                            for bbox_plate in result_plate.boxes.data.tolist():
                                px1, py1, px2, py2, pscore, pclass_id = bbox_plate
                                # crop to get license plate 
                                plate = vehicle[int(py1):int(py2), int(px1):int(px2)]
                                
                                # Measure OCR time
                                start_time = time.time()
                                ocr_result = reader.readtext(plate)
                                ocr_time = (time.time() - start_time) * 1000  # in milliseconds
                                ocr_inference_times.append(ocr_time)
                                
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
            
            if not vehicle_detected:
                # No vehicle detected, try to detect plates in the entire image
                start_time = time.time()
                results_plate = model_plate(image)
                plate_time = (time.time() - start_time) * 1000  # in milliseconds
                plate_inference_times.append(plate_time)
                
                for result_plate in results_plate:
                    for bbox_plate in result_plate.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = bbox_plate
                        # Crop to get the license plate
                        plate = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Measure OCR time
                        start_time = time.time()
                        ocr_result = reader.readtext(plate)
                        ocr_time = (time.time() - start_time) * 1000  # in milliseconds
                        ocr_inference_times.append(ocr_time)
                        
                        # Annotate OCR results on the original image
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
                            
                            # Log OCR results
                            log.write(f"  Detected plate: {text} with confidence: {prob}\n")
                            
                        # Annotate the plate on the original image
                        cv2.rectangle(image, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (255, 0, 0), 2)
                        
        # Save the annotated image
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        annotated_image_path = os.path.join(output_folder, "annotated_" + os.path.basename(image_path))
        cv2.imwrite(annotated_image_path, image)
        print(f"Annotated image saved to: {annotated_image_path}")
    
    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error processing {image_path}: {str(e)}\n")
        print(f"Error processing {image_path}: {str(e)}")

def process_folder(input_folder, output_folder, log_file):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in os.listdir(input_folder) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    for image_path in tqdm(images, desc="Processing images"):
        full_image_path = os.path.join(input_folder, image_path)
        detect_and_recognize(full_image_path, output_folder, log_file)
    
    # Calculate and display average inference times
    avg_vehicle_time = sum(vehicle_inference_times) / len(vehicle_inference_times) if vehicle_inference_times else 0
    avg_plate_time = sum(plate_inference_times) / len(plate_inference_times) if plate_inference_times else 0
    avg_ocr_time = sum(ocr_inference_times) / len(ocr_inference_times) if ocr_inference_times else 0

    print(f"\nAverage inference times (in ms):")
    print(f"Vehicle detection: {avg_vehicle_time:.2f} ms")
    print(f"Plate detection: {avg_plate_time:.2f} ms")
    print(f"OCR: {avg_ocr_time:.2f} ms")

input_folder = 'input'
output_folder = 'output'
log_file = 'detection_log.txt'

process_folder(input_folder, output_folder, log_file)
