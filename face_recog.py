import sys
import face_recognition
import numpy as np
from PIL import Image


# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama_small.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []


def convert_img_to_array(img):
    img = Image.open(img)
    img_array = np.array(img)
    return img_array


def get_encodings(img_array):
    face_locations = face_recognition.face_locations(img_array)
    print("Found {} faces in image.".format(len(face_locations)))
    face_encodings = face_recognition.face_encodings(img_array, face_locations)
    # print(sys.getsizeof(face_encodings))
    return face_encodings


def run_recognition(face_encodings):
    # Loop over each face found in the frame to see if it's someone we know.
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
        name = "<Unknown Person>"

        if match[0]:
            name = "Barack Obama"

        print("I see someone named {}!".format(name))

        return name
