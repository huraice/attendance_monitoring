import os
import cv2
import face_recognition


@app.route('/enroll_student', methods=['POST'])
def enroll_student():
    # Assuming you receive a face image as base64 encoded string
    data = request.json
    face_image_data = data['face_image']

    # Decode the image
    img_data = face_image_data.split(",")[1]
    img_data = base64.b64decode(img_data)

    # Load the image with OpenCV
    np_img = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Encode the face for recognition
    face_encoding = face_recognition.face_encodings(image)

    if face_encoding:
        # Save the face image to the 'faces' folder
        student_name = data['name']  # Get student's name from request
        face_file_path = os.path.join('faces', f"{student_name}.jpg")
        cv2.imwrite(face_file_path, image)

        return jsonify({"message": f"{student_name} enrolled successfully."}), 200
    else:
        return jsonify({"message": "No face detected."}), 400
