import numpy as np

from flask import Flask, request, jsonify
from PIL import Image

from face_recog import convert_img_to_array, get_encodings, run_recognition

app = Flask(__name__)


@app.route("/im_size", methods=["POST"])
def process_image():
    file = request.files['image']

    # Read the image via file.stream
    img = Image.open(file.stream)

    return jsonify({'msg': 'success', 'size': [img.width, img.height]})


@app.route("/standalone/recognize", methods=["POST"])
def recognize_from_image():
    # Get the image from the request
    img = request.files['image']

    # Read the image and convert to array
    img_array = convert_img_to_array(img)

    # Get the face encodings from the image array
    face_encodings = get_encodings(img_array)

    # Run the face recognition
    result = run_recognition(face_encodings)

    return jsonify({'msg': 'success', 'response': result})


@app.route("/oblique/recognize", methods=["POST"])
def recognize_from_image_encodings():
    # Get the image encodings from the request and convert the list format to numpy arrays
    face_encodings = [np.array(arr_list) for arr_list in request.json['face_encodings']]

    # Run the face recognition
    result = run_recognition(face_encodings)

    return jsonify({'msg': 'success', 'response': result})


if __name__ == "__main__":
    app.run(debug=True)
