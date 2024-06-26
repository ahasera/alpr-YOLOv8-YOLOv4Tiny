import cv2
import numpy as np
import os
import easyocr

config_path = 'cfg/yolov4-tiny.cfg'
weights_path = 'custom-yolov4-tiny-detector_last.weights'
names_path = 'cfg/obj.names'
input_folder = 'input'
output_folder = 'output'

if not os.path.exists(config_path):
    print(f"Erreur: fichier de configuration non trouvé: {config_path}")
    exit(1)

if not os.path.exists(weights_path):
    print(f"Erreur: fichier de poids non trouvé: {weights_path}")
    exit(1)

if not os.path.exists(names_path):
    print(f"Erreur: fichier de noms non trouvé: {names_path}")
    exit(1)

# Load the network and the weights
net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, 'r') as f:
    classes = f.read().splitlines()
    
# EasyOCR init 
reader = easyocr.Reader(['en'])

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# iterate through the folder 
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)

        # charge image and pre-process 
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # reusult analysis
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5: # you can lower or higher this value depending on the accuracy you want 
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # verify detections 
        if len(class_ids) > 0:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if len(indexes) > 0:
                for i in indexes.flatten():
                    if i < len(class_ids) and i < len(boxes):
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = confidences[i]
                        color = (0, 255, 0)
                        
                        # extract cropped image text
                        crop_img = img[y:y+h, x:x+w]
                        ocr_result = reader.readtext(crop_img)
                        ocr_text = " ".join([res[1] for res in ocr_result])

                        # annotation
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
                        
                        ocr_text_position_y = y + h + 20
                        if ocr_text_position_y + 30 > height:  # if ocr annotation overlaps, change its position upper
                            ocr_text_position_y = y - 30
                        cv2.putText(img, ocr_text, (x, ocr_text_position_y), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        cv2.imwrite(output_image_path, img)
        print(f"Image processed and saved at: {output_image_path}")
    else:
        print(f"ignored : not supported image: {filename}")

print("Program exited successfully")