import numpy as np

from flask import Flask, request, jsonify

from face_recog import convert_img_to_array, get_encodings, run_recognition

app = Flask(__name__)


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


@app.route("/hybrid/cloud", methods=["POST"])
def hybrid_recognition_cloud():
    # Get the image encodings from the request and convert the list format to numpy arrays
    face_encodings = [np.array(arr_list) for arr_list in request.json['face_encodings']]

    # Run the face recognition
    result = run_recognition(face_encodings)

    return jsonify({'msg': 'success', 'response': result})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
