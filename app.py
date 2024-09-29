
import base64
from flask import Flask, render_template, jsonify, request
import os
import cv2
import face_recognition
import numpy as np
app = Flask(__name__)

# Directory to save face images
FACE_DIR = 'faces'
os.makedirs(FACE_DIR, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'POST':
        student_name = request.form['student_name']
        face_image_data = request.form['face_image']

        # Process the base64 face image
        header, encoded = face_image_data.split(',')
        image_data = base64.b64decode(encoded)

        # Save the image to a file
        face_image_path = os.path.join(FACE_DIR, f"{student_name}.png")
        with open(face_image_path, 'wb') as f:
            f.write(image_data)

        return f"Enrolled: {student_name} with face image saved!"

    return render_template('enroll.html')


@app.route('/add_attendance')
def add_attendance():
    data = request.json
    face_image_data = data['face_image']

    # Decode the captured face image
    img_data = face_image_data.split(",")[1]
    img_data = np.frombuffer(base64.b64decode(img_data), np.uint8)
    image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    # Get face encoding for the captured image
    captured_face_encoding = face_recognition.face_encodings(image)

    if not captured_face_encoding:
        return jsonify({"message": "No face detected."}), 400

    # Compare captured face with stored face encodings
    for filename in os.listdir('faces'):
        if filename.endswith('.jpg'):
            stored_image = face_recognition.load_image_file(os.path.join('faces', filename))
            stored_face_encoding = face_recognition.face_encodings(stored_image)

            if stored_face_encoding:
                matches = face_recognition.compare_faces(stored_face_encoding, captured_face_encoding[0])
                if True in matches:
                    student_name = filename.split('.')[0]
                    return jsonify({"message": f"Attendance marked for {student_name}."}), 200

    return jsonify({"message": "Face not recognized."}), 404


@app.route('/summarize_attendance')
def summarize_attendance():
    attendance_summary = []  # Replace with actual data
    return render_template('summarize_attendance.html', summary=attendance_summary)


if __name__ == '__main__':
    app.run(debug=True)
