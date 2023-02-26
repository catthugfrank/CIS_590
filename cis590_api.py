import requests
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
def standalone_recognition():
    # Get the image from the request
    img = request.files['image']

    # Read the image and convert to array
    img_array = convert_img_to_array(img)

    # Get the face encodings from the image array
    face_encodings = get_encodings(img_array)

    # Run the face recognition
    result = run_recognition(face_encodings)

    return jsonify({'msg': 'success', 'response': result})


@app.route("/hybrid/edge", methods=["POST"])
def hybrid_recognition_edge():
    # Get the image from the request
    img = request.files['image']

    # Read the image and convert to array
    img_array = convert_img_to_array(img)

    # Get the face encodings from the image array
    face_encodings = get_encodings(img_array)

    # Convert the numpy arrays to python lists
    face_encodings_list = [arr.tolist() for arr in face_encodings]

    # Send a post request to the specific API endpoint with encodings data
    result = requests.post("http://127.0.0.1:5001/hybrid/cloud", json={"face_encodings": face_encodings_list}).json()

    if result['msg'] == 'success':
        result = result['response']

    # Send results to requesting device
    return jsonify({'msg': 'success', 'response': result})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
