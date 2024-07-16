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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse arguments
parser = argparse.ArgumentParser(description='Vehicle and License Plate Detection with optional preprocessing')
parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing of images, see preprocessing.py')
parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for detection')
parser.add_argument('--input-size', type=int, default=416, help='Input size for YOLO models, must be 32 multiple')
args = parser.parse_args()

"""
YoloV8 model initialisation 
First one is pretrained coco model 
Second one is yours or mine, for license plate object detection. 
"""
try:
    model_vehicle = YOLO('models/yolov8/yolov8n.pt')
except Exception as e:
    logging.error(f"Error while loading yolov8n base model: {e}")
    exit(1)

try:
    model_plate = YOLO('models/yolov8/best.pt')  # YOLOv8 model for plate detection
except Exception as e:
    logging.error(f"Error while loading license-plate model: {e}")
    exit(1)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def adjust_text_position(image, text_x, text_y, text, font_scale, thickness, used_positions):
    """
    Adjust text position to avoid overlapping with image boundaries and other texts.
    
    Args:
        image (np.array): The input image.
        text_x (int): Initial x-coordinate for text placement.
        text_y (int): Initial y-coordinate for text placement.
        text (str): The text to be placed.
        font_scale (float): Font scale for the text.
        thickness (int): Thickness of the text.
        used_positions (list): List of previously used text positions.
    
    Returns:
        tuple: Adjusted (x, y) coordinates for text placement.
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    image_height, image_width = image.shape[:2]
    
    # Adjust if text goes beyond right edge of the image
    if text_x + text_width > image_width:
        text_x = image_width - text_width - 10
    
    # Adjust if text goes above top edge of the image
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

def format_inference_time(start_time):
    """
    Calculate and format the inference time. May not be very accurate with multi-threading implementation.
    
    Args:
        start_time (float): The start time of the inference.
    
    Returns:
        str: Formatted inference time in milliseconds.
    """
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    return f"{inference_time:.2f}"

def detect_and_recognize(image_path, output_folder, cropped_folder, log_file, args):
    """
    Detect vehicles and license plates in an image, perform OCR, and annotate the results.
    
    This function performs the following steps:
    1. Read and optionally preprocess the image.
    2. Detect vehicles using YOLOv8.
    3. For each detected vehicle, detect license plates.
    4. For each detected license plate, perform OCR.
    5. Annotate the image with bounding boxes and OCR results.
    6. Save the annotated image and cropped license plates.
    7. Log the results and processing times.
    
    Args:
        image_path (str): Path to the input image.
        output_folder (str): Path to save annotated images.
        cropped_folder (str): Path to save cropped license plates.
        log_file (str): Path to the log file.
        args (argparse.Namespace): Command-line arguments.
    """
    try:
        # Read and optionally preprocess the image
        if args.preprocess:
            image = preprocess_image(image_path)
        else:
            image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        used_positions = []
        
        with open(log_file, 'a') as log:
            log.write(f"Processing {image_path}:\n")
            
            # Detect vehicles and measure detection time 
            start_time = time.time()
            results_vehicle = model_vehicle(image, conf=args.confidence, imgsz=args.input_size)
            vehicle_time = format_inference_time(start_time)
            log.write(f"Vehicle detection time: {vehicle_time} ms\n")
            vehicle_detected = False
            
            for result in results_vehicle:
                for bbox in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = bbox
                    if int(class_id) == 2:  # Class ID 2 represents cars in COCO dataset, as we only need this one in our case.
                        vehicle_detected = True
                        vehicle = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Detect license plates within the vehicle region, and measure deteciton time
                        start_time = time.time()
                        results_plate = model_plate(vehicle, conf=args.confidence, imgsz=args.input_size)
                        plate_time = format_inference_time(start_time)
                        log.write(f"Plate detection time: {plate_time} ms\n")
                        
                        for result_plate in results_plate:
                            for bbox_plate in result_plate.boxes.data.tolist():
                                px1, py1, px2, py2, pscore, pclass_id = bbox_plate
                                plate = vehicle[int(py1):int(py2), int(px1):int(px2)]
                                # Save the cropped license plate image
                                plate_filename = os.path.join(cropped_folder, f"plate_{os.path.basename(image_path)}")
                                cv2.imwrite(plate_filename, plate)
                                
                                # Perform OCR on the license plate, and measure OCR time
                                start_time = time.time()
                                ocr_result = reader.readtext(plate)
                                ocr_time = format_inference_time(start_time)
                                log.write(f"OCR time: {ocr_time} ms\n")
                                
                                # Annotate the image with OCR results
                                for (bbox_ocr, text, prob) in ocr_result:
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
                                    
                                # Annotate the license plate on the original image
                                cv2.rectangle(image, (int(px1) + int(x1), int(py1) + int(y1)),
                                              (int(px2) + int(x1), int(py2) + int(y1)), (255, 0, 0), 2)
                                
                        # Annotate the vehicle on the original image
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # If no vehicle is detected, will allow to run the second model if the first one did not detect a vehicle. Can cause problem in some situations with many objects nearby
            if not vehicle_detected:
                start_time = time.time()
                results_plate = model_plate(image, conf=args.confidence, imgsz=args.input_size)
                plate_time = format_inference_time(start_time)
                log.write(f"Plate detection time (full image): {plate_time} ms\n")
                
                for result_plate in results_plate:
                    for bbox_plate in result_plate.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = bbox_plate
                        plate = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Save the cropped license plate image
                        plate_filename = os.path.join(cropped_folder, f"plate_{os.path.basename(image_path)}")
                        cv2.imwrite(plate_filename, plate)
                        
                        # Perform OCR on the license plate, and measure time
                        start_time = time.time()
                        ocr_result = reader.readtext(plate)
                        ocr_time = format_inference_time(start_time)
                        log.write(f"OCR time: {ocr_time} ms\n")
                        
                        # Annotate the image with OCR results
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
                            
                        # Annotate the license plate on the original image
                        cv2.rectangle(image, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (255, 0, 0), 2)
                        
        # Save the annotated image
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        annotated_image_path = os.path.join(output_folder, "annotated_" + os.path.basename(image_path))
        cv2.imwrite(annotated_image_path, image)
        logging.info(f"Annotated image saved to: {annotated_image_path}")

        # Clean up to free memory
        gc.collect()
    
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")

def process_image(args):
    """
    Wrapper function to process a single image.
    
    This function is designed to be used with multiprocessing.
    It unpacks the arguments and calls detect_and_recognize.
    
    Args:
        args (tuple): A tuple containing image path and other necessary arguments.
    
    Returns:
        str: The path of the processed image.
    """
    image_path, output_folder, cropped_folder, log_file, detection_args = args
    detect_and_recognize(image_path, output_folder, cropped_folder, log_file, detection_args)
    return image_path

def process_folder(input_folder, output_folder, cropped_folder, log_file, args):
    """
    Process all images in the input folder using multiprocessing.
    
    This function:
    1. Identifies all valid image files in the input folder.
    2. Creates necessary output directories.
    3. Uses a ProcessPoolExecutor to process images in parallel.
    4. Tracks progress using tqdm.
    5. Generates a summary report of the processing.
    
    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to save annotated images.
        cropped_folder (str): Path to save cropped license plates.
        log_file (str): Path to the log file.
        args (argparse.Namespace): Command-line arguments.
    """
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
        report.write(f"Average time per image: {total_time/total_images:.2f} seconds\n")
    
    logging.info("Processing completed. Summary report generated in summary_report.txt")

# Main execution
if __name__ == "__main__":
    # Define input and output directories
    input_folder = 'data/input'
    output_folder = 'data/output'
    cropped_folder = 'data/cropped'
    log_file = 'detection_log.txt'

    # Ensure input folder exists
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    
    # Process all images in the input folder
    process_folder(input_folder, output_folder, cropped_folder, log_file, args)