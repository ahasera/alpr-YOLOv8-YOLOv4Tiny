from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, Response, jsonify
import os
import subprocess
import shutil
import cv2 

app = Flask(__name__)
camera = None

# Configuration
INPUT_FOLDER = 'data/input'
OUTPUT_FOLDER = 'data/output'
CROPPED_FOLDER = 'data/cropped'
LOG_FILE = 'detection_log.txt'

"""
Standalone functions
"""
def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # Assurez-vous de libérer la caméra avant de la réallouer
    return camera

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/images/<folder>/<path:filename>')
def send_image(folder, filename):
    if folder == "output":
        directory = OUTPUT_FOLDER
    elif folder == "cropped":
        directory = CROPPED_FOLDER
    else:
        return "Folder not found", 404  # Sécurité pour ne pas accéder à d'autres dossiers
    return send_from_directory(directory, filename)

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'POST':
        model = request.form['model']
        files = request.files.getlist('files')
        preprocess = 'preprocess' in request.form
        confidence = float(request.form['confidence'])

        # Clear previous results
        clear_directory(INPUT_FOLDER)
        clear_directory(OUTPUT_FOLDER)
        clear_directory(CROPPED_FOLDER)

        # Save uploaded files
        if files:
            for file in files:
                if file.filename:
                    file_path = os.path.join(INPUT_FOLDER, file.filename)
                    file.save(file_path)
        else:
            return render_template('batch.html', error="No files uploaded")

        # Process images
        if model == 'yolov8':
            script = 'lpdetect.py'
        else:
            script = 'rpi-lp.py'

        cmd = [
            'python', script,
            '--confidence', str(confidence)
        ]
        if preprocess:
            cmd.append('--preprocess')

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            return render_template('batch.html', error="Error processing images. Please check if you uploaded valid image files.")

        return redirect(url_for('results'))

    return render_template('batch.html')

@app.route('/results')
def results():
    output_images = os.listdir(OUTPUT_FOLDER)
    cropped_images = os.listdir(CROPPED_FOLDER)
    return render_template('results.html', output_images=output_images, cropped_images=cropped_images)

@app.route('/start_camera')
def start_camera():
    get_camera()  # Just call to initialize the camera
    return jsonify({"status": "Camera started"})
@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status": "Camera stopped"})
@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/video_feed')
def video_feed():
    """route for video stream."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        camera = get_camera()
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/clean', methods=['GET', 'POST'])
def clean():
    if request.method == 'POST':
        cmd = ['python', 'clean.py']
        if 'input' in request.form:
            cmd.append(INPUT_FOLDER)
        if 'output' in request.form:
            cmd.append(OUTPUT_FOLDER)
        if 'cropped' in request.form:
            cmd.append(CROPPED_FOLDER)
        if 'log' in request.form:
            cmd.extend(['--truncate-log', LOG_FILE])
        
        subprocess.run(cmd, check=True)
        return render_template('clean.html', message="Cleaned successfully!")
    return render_template('clean.html')

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == '__main__':
    for folder in [INPUT_FOLDER, OUTPUT_FOLDER, CROPPED_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    app.run(debug=True, threaded=True)
    
