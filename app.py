from flask import Flask, render_template, request, redirect, url_for, session
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from csv import writer
from flask_material import Material
import imutils
from flask import Flask, render_template, request, url_for, session, redirect, flash
import pandas as pd  
import numpy as np
from tensorflow.keras.models import load_model
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
import pygame
import threading
# ML Pkg

UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
Material(app)

class_names=['Normal', 'Tired']
img_height = 224
img_width = 224
app.secret_key = 'your_secret_key_here'
secret_key="1sdassssssss"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# parameters for loading data and images

loaded_model = load_model("Eyeneww.h5")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def play_alert_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.mp3")
    pygame.mixer.music.play()
# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Enter your database connection details below

@app.route('/')
def index():
    return render_template("login.html")

@app.route('/home')
def home():
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():

    return render_template('about.html')
    # User is not loggedin redirect to login page

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/',methods=['GET', 'POST'])
def login():
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        
                # If account exists in accounts table in out database
        if username=="admin" and password=="admin":
            return render_template('index.html')
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)


# Function to extract landmarks
def extract_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

# Function to predict fatigue using the loaded model
def predict_fatigue_with_loaded_model(image_path, model):
    image = cv2.imread(image_path)
    landmarks = extract_landmarks(image)
    if landmarks is None:
        print("No face detected in the image.")
        return
    landmarks = landmarks / 255.0  # Normalize landmarks
    landmarks = landmarks.reshape(1, 68, 2)
    prediction = model.predict(landmarks)
    class_label = "Normal" if np.argmax(prediction) == 0 else "Tired"
    print(f"Prediction: {class_label}")
    return class_label

# Load shape predictor model
shape_predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
# Function to detect a face in the image using shape_predictor
def detect_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    if len(faces) == 0:
        return False  # No face detected
    
    for face in faces:
        landmarks = shape_predictor(gray, face)
        if landmarks.num_parts == 68:
            return True  # Face with landmarks detected
    return False


@app.route('/upload_image', methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if face exists in the image
        if not detect_face(path):
            flash('No face detected in the image. Please upload a valid image.')
            return render_template('upload.html', res=0,filename=filename)

        # Load the trained model
        model = Sequential([
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(2)
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        model.load_weights("Eyeneww.h5")

        # Prepare image for prediction
        img = keras.preprocessing.image.load_img(
            path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Make prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        return render_template('upload.html', aclass=class_names[np.argmax(score)], ascore=100 * np.max(score), res=1, filename=filename)
    
    flash('Invalid file type')
    return redirect(request.url)

import time
@app.route('/upload_image1',methods=["POST"])
def upload_image1():
    thresh = 0.25  # EAR threshold for closed eyes
    frame_check = 50
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Landmark model

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)

    flag = 0
    blink_count = 0
    start_time = time.time()
    blink_rate_threshold = 12  # Minimum healthy blink rate per minute

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  # Convert to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw contours around the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Blink detection logic
            if ear < thresh:  # Eye closed
                flag += 1
            else:
                if flag >= 3:  # Minimum frames for a blink
                    blink_count += 1
                    flag = 0

            # Display blink count on the frame
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Calculate blink rate every minute
        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            blink_rate = blink_count
            blink_count = 0
            start_time = time.time()

            # Alert for low blink rate
            if blink_rate < blink_rate_threshold:
                threading.Thread(target=play_alert_sound).start()
                cv2.putText(frame, "LOW BLINK RATE! TAKE A BREAK!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    return render_template('index.html')


# @app.route('/upload_image1',methods=["POST"])
# def upload_image1():
	

#     thresh = 0.25
#     frame_check = 50
#     detect = dlib.get_frontal_face_detector()
#     predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat") # Dat file is the crux of the code

#     (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
#     (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#     cap = cv2.VideoCapture(0)
#     flag = 0

#     while True:
#         ret, frame = cap.read()
#         frame = imutils.resize(frame, width=450)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         subjects = detect(gray, 0)
#         for subject in subjects:

#             shape = predict(gray, subject)
#             shape = face_utils.shape_to_np(shape) # Converting to NumPy Array
#             leftEye = shape[lStart:lEnd]
#             rightEye = shape[rStart:rEnd]
#             leftEAR = eye_aspect_ratio(leftEye)
#             rightEAR = eye_aspect_ratio(rightEye)
#             ear = (leftEAR + rightEAR) / 2.0
#             leftEyeHull = cv2.convexHull(leftEye)
#             rightEyeHull = cv2.convexHull(rightEye)
#             cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#             cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#             if ear < thresh:
#                 flag += 1
#                 if flag >= frame_check:
#                     cv2.putText(frame, "****************ALERT!****************", (10, 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     cv2.putText(frame, "****************ALERT!****************", (10, 325),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     threading.Thread(target=play_alert_sound).start()
                    
#             else:
#                 flag = 0
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break

#     cv2.destroyAllWindows()
#     cap.release()
#     return render_template('index.html')


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
	app.run(debug=True)
