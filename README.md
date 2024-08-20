#  alpr-YOLOv8-YOLOv4Tiny
![License](https://img.shields.io/badge/license-GNU-blue.svg) 
##  Project Description

This project is a subproject made in the context of an internship. Its goal is to provide tools that allows to detect and extract the characters of license plates. At the end, the purpose of this project for me is to compare different solutions, models accuracy and test in different conditions.

##  Table of Contents

- [Project Description](#project-description)
- [Table of Contents](#table-of-contents)
- [Content](#content)
- [Installation](#installations)
- [Usage](#usage)
- [Models](#models)
- [License](#license)
- [Acknowledgements](#Acknowledgements)

##  Content
 
In this repo, you will find different script with different use cases :

-  `lpdetect.py` : This script aims at recognizing first vehicles, and then the license plate on the vehicle.

--> Two models are used :

1. The YOLOv8 COCO pretrained model, in which we will keep the vehicle classes only for the annotation and detection.

2. Your own pretrained YOLOv8 ALPR model or mine.

-  `rpi-lp.py` : This one is quite similar to the other, although it uses a YOLOv4Tiny model.

-  `rpi-realtime.py` : This script aims at using the Raspberry Pi camera module to detect with a realtime preview the license plates. It uses the same YOLOv4Tiny model.

>  Note that the realtime has been tested on a Raspberry PI 4 4GB. The process is quite heavy due to processing plate detection, OCR and annotation.

-  `clean.py` : This tool is just used in order to quicly clean input, output folders and truncating logs, all together or separately.  

##  Installations

You first need to meet dependencies requirements.

  

####  Ubuntu, Debian based systems:

```
sudo apt update
sudo apt install libcap-dev libjpeg-devel -y
```

####  CENTOS, Fedora, Alma, RHEL based systems:

```
sudo dnf update -y
sudo dnf install python3-devel libcap-devel -y
```

Please use a virtual environment to avoid conflicts with your package manager :

```
python3 -m venv venv
```

  

  

Then install the requirements with pip :

```
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

>  Please note that if you want to use the realtime detection on your Raspberry Pi, I recommend you to first install picamera2 module if not installed with APT : `sudo apt install python3-picamera2` and then create your venv with this command `python3 -m venv --system-site-packages venv` to avoid errors using the module within the venv.

## Usage
### Batch images 
You can run batch processing programs with these simple commands (run once to download COCO pretrained and create data folders): 
```
python3 lpdetect.py 
python3 rpi-lp.py
```
Add `--preprocess` option for preprocessing steps (detailed here)

Example usage of `clean.py` :
```
python3 clean.py data/output data/cropped --truncate-log detection_log.txt
```

### Realtime
For now, only detection on RaspberryPi is available. 
Available options are:

* `--export` Exporting annotated images to path of yout choice

* `--skip` Skipping frames to improve performance (not recommended)

* `--tesseract` Choosing between Tesseract and EasyOCR for text recognition (tesseract recommended for performance, EasyOCR for accuracy)

* `--multi-thread` Enabling multi-threading for improved processing speed (I'd recommend to use it only if you plan to export frames as asynchronous image processing will lead annotations to pop and disappear)

* `--no-ocr` Disabling OCR to focus solely on plate detection


##  MODELS

The models we used are both trained on the same dataset for a legitimate comparison between models accuracy and performance. The notebooks used for the training are available in the notebooks folder.

The dataset we used for training is available on the roboflow universe [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)

>  The notebooks are made to import data with roboflow.

# Demo
### Batch YOLOv8
![batch-yolov8-demo](demo/batch.gif)

### Live Realtime on RaspberryPi4 

![realtime-demo](demo/rpi-realtime.gif)
# License
This project is licensed under the GNU v3.0 License - see the [LICENSE](LICENSE) file for details.

# Acknowledgements
I would like to address my thanks to the CRISES Group and my internship tutor in the Universitat Rovira i Virgili for the opportunity to work on this project by providing me the necessary resources and support. 
```
```