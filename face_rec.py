import sys
import face_recognition
import numpy as np
from PIL import Image

img= Image.open("test.jpeg")
output = np.array(img)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama_small.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []

face_locations = face_recognition.face_locations(output)
print("Found {} faces in image.".format(len(face_locations)))
face_encodings = face_recognition.face_encodings(output, face_locations)
print(sys.getsizeof(face_encodings))

# Loop over each face found in the frame to see if it's someone we know.
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
    name = "<Unknown Person>"

    if match[0]:
        name = "Barack Obama"

    print("I see someone named {}!".format(name))
