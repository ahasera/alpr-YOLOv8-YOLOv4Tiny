"""
License Plate Detection and OCR for Raspberry Pi

This script performs license plate detection and optical character recognition (OCR)
on images using a YOLOv4-tiny model and EasyOCR. It is optimized for use on Raspberry Pi.

The script processes all images in a specified input folder, detects license plates,
performs OCR on the detected plates, and saves annotated images and cropped plate images.

Usage:
    python3 rpi-lp.py [--preprocess] [--confidence CONFIDENCE] [--input-size INPUT_SIZE]

Arguments:
    --preprocess: Apply image preprocessing before detection (optional)
    --confidence: Confidence threshold for detection (default: 0.5)
    --input-size: Input size for YOLO model, must be multiple of 32 (default: 416)

Dependencies:
    - OpenCV (cv2)
    - NumPy
    - EasyOCR
    - Custom preprocessing module (preprocessing.py)

Author: ahasera
Date: 25/07/2024
"""

import cv2
import numpy as np
import os
import easyocr
import time
import argparse
import logging
from preprocessing import preprocess_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse arguments
parser = argparse.ArgumentParser(description='License Plate Detection and OCR for Raspberry Pi')
parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing of images')
parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for detection')
parser.add_argument('--input-size', type=int, default=416, help='Input size for YOLO model, must be multiple of 32')
args = parser.parse_args()

# Paths
config_path = 'cfg/yolov4-tiny.cfg'
weights_path = 'models/yolov4-tiny/custom-yolov4-tiny-detector_last.weights'
names_path = 'cfg/obj.names'
input_folder = 'data/input'
output_folder = 'data/output'
cropped_folder = 'data/cropped'
log_file = 'detection_log_rpi.txt'

# Check for required files
for path in [config_path, weights_path, names_path]:
    if not os.path.exists(path):
        logging.error(f"Error: file not found {path}")
        exit(1)

# Load the network
try:
    net = cv2.dnn.readNet(weights_path, config_path)
    logging.info("License plate detection model loaded successfully")
except Exception as e:
    logging.error(f"Error loading license plate detection model: {e}")
    exit(1)

# Load class names
with open(names_path, 'r') as f:
    classes = f.read().splitlines()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Create necessary directories
for folder in [output_folder, input_folder, cropped_folder]:
    os.makedirs(folder, exist_ok=True)

def format_inference_time(start_time):
    """
    Calculate and format the inference time.

    Args:
        start_time (float): The start time of the inference.

    Returns:
        str: Formatted inference time in milliseconds.
    """
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    return f"{inference_time:.2f}"

def detect_plates(img, confidence_threshold):
    """
    Detect license plates in the image using YOLOv4-tiny.

    Args:
        img (numpy.ndarray): Input image.
        confidence_threshold (float): Confidence threshold for detections.

    Returns:
        tuple: Lists of bounding boxes, confidences, and class IDs of detected plates.
    """
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (args.input_size, args.input_size), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

def adjust_text_position(image, text_x, text_y, text, font_scale, thickness, used_positions):
    """
    Adjust text position to avoid overlapping with image boundaries and other texts.
    
    Args:
        image (numpy.ndarray): The input image.
        text_x (int): Initial x-coordinate for text placement.
        text_y (int): Initial y-coordinate for text placement.
        text (str): The text to be placed.
        font_scale (float): Font scale for the text.
        thickness (int): Thickness of the text.
        used_positions (list): List of previously used text positions.
    
    Returns:
        tuple: Adjusted (x, y) coordinates for text placement.
    """
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    image_height, image_width = image.shape[:2]
    
    if text_x + text_width > image_width:
        text_x = image_width - text_width - 10
    if text_y - text_height < 0:
        text_y = text_height + 10
    
    while any(abs(text_x - used_x) < text_width and abs(text_y - used_y) < text_height for used_x, used_y in used_positions):
        text_y += text_height + 10
        if text_y + text_height > image_height:
            text_y = text_height + 10
            text_x += text_width + 10
            if text_x + text_width > image_width:
                text_x = 10
    
    used_positions.append((text_x, text_y))
    return text_x, text_y

def process_image(image_path):
    """
    Process a single image for license plate detection and OCR.

    This function performs the following steps:
    1. Read and optionally preprocess the image.
    2. Detect license plates using YOLOv4-tiny.
    3. Perform OCR on detected license plates.
    4. Annotate the image with bounding boxes and OCR results.
    5. Save the annotated image and cropped license plates.

    Args:
        image_path (str): Path to the input image.
    """
    logging.info(f"Processing image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Error reading image {image_path}")
        return

    if args.preprocess:
        try:
            preprocessed_img = preprocess_image(image_path)
            if preprocessed_img is not None and len(preprocessed_img.shape) == 3:
                img = preprocessed_img
                logging.info("Preprocessing applied successfully")
            else:
                logging.warning("Preprocessing failed, using original image")
        except Exception as e:
            logging.error(f"Error during preprocessing of {image_path}: {e}")
            logging.warning("Using original image due to preprocessing error")

    start_time = time.time()
    boxes, confidences, class_ids = detect_plates(img, args.confidence)
    detection_time = format_inference_time(start_time)

    logging.info(f"Detection time: {detection_time} ms")
    logging.info(f"Number of detections: {len(boxes)}")

    if len(boxes) > 0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, 0.4)
        
        if isinstance(indexes, tuple):
            indexes = indexes[0] if len(indexes) > 0 else []

        used_positions = []
        
        for i in indexes.flatten() if isinstance(indexes, np.ndarray) else indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            
            # Crop and save plate image
            plate_img = img[y:y+h, x:x+w]
            plate_filename = os.path.join(cropped_folder, f"plate_{i}_{os.path.basename(image_path)}")
            cv2.imwrite(plate_filename, plate_img)

            # Perform OCR
            start_ocr_time = time.time()
            ocr_result = reader.readtext(plate_img)
            ocr_time = format_inference_time(start_ocr_time)

            ocr_text = " ".join([f"{res[1]} ({res[2]:.2f})" for res in ocr_result])
            logging.info(f"OCR result: {ocr_text}")

            # Annotate image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text_x, text_y = adjust_text_position(img, x, y - 10, label, 0.5, 2, used_positions)
            cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            text_x, text_y = adjust_text_position(img, x, y + h + 20, ocr_text, 0.5, 2, used_positions)
            cv2.putText(img, ocr_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Log results
            logging.info(f"Plate detected: {label}")
            logging.info(f"OCR result: {ocr_text}")
            logging.info(f"OCR time: {ocr_time} ms")

    else:
        logging.info("No license plates detected in this image.")

    output_image_path = os.path.join(output_folder, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, img)
    logging.info(f"Annotated image saved at: {output_image_path}")

def main():
    """
    Main function to process all images in the input folder.

    This function:
    1. Identifies all valid image files in the input folder.
    2. Processes each image using the process_image function.
    3. Logs the overall processing status.
    """
    logging.info("Starting license plate detection and OCR process")
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        process_image(image_path)

    logging.info("All images processed successfully")

if __name__ == "__main__":
    main()