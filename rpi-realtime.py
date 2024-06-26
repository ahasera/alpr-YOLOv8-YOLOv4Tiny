import cv2
import numpy as np
import easyocr
from picamera2 import Picamera2, Preview
import time

""""
The yolov4-tiny.cfg IS NOT the default one. It has been adapted to correctly match my model classes. 
As there is only 1 class, I updated the filters using the formula given in the Darknet Repo (YOLOV4Tiny Darknet): (nbclass+5)x3 which in my case gets us to 18.
"""
config_path = 'cfg/yolov4-tiny.cfg'
weights_path = 'custom-yolov4-tiny-detector_last.weights' # you can change this by your own weights
names_path = 'cfg/obj.names' # if you have more classes than mine don't forget to update them here. 

# Load the network and the weights
net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, 'r') as f:
    classes = f.read().splitlines()

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Initialize the raspberry pi camera module with picamera2  
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (808, 606)}, display="lores")
picam2.preview_configuration.main.format = "RGB888"
picam2.configure(config)
picam2.start()

time.sleep(2)

# Capture and process images in a loop
while True:
    img = picam2.capture_array()
    height, width, _ = img.shape

    # Pre-process the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Analyze the results
    class_ids = []
    confidences = []
    boxes = []
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

    # If detections are made
    if len(class_ids) > 0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indexes) > 0:
            for i in indexes.flatten():
                if i < len(class_ids) and i < len(boxes):
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = (0, 255, 0)

                    # Extract text within the box
                    crop_img = img[y:y+h, x:x+w]
                    ocr_result = reader.readtext(crop_img)
                    ocr_text = " ".join([res[1] for res in ocr_result])

                    # Annotate the image with YOLO results
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

                    # Add OCR text below the detection box
                    ocr_text_position_y = y + h + 20
                    if ocr_text_position_y + 30 > height:
                        ocr_text_position_y = y - 30

                    cv2.putText(img, ocr_text, (x, ocr_text_position_y), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Display the annotated image
    cv2.imshow("Annotated Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
picam2.stop()
cv2.destroyAllWindows()
