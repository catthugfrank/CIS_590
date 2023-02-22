import requests

from face_recog import convert_img_to_array, get_encodings

# constant API endpoints
standalone_api_endpoint = 'http://127.0.0.1:5000/standalone/recognize'
oblique_api_endpoint = 'http://127.0.0.1:5000/oblique/recognize'


def test_standalone_recognition():
    filename ='President_Barack_Obama.jpg'
    my_img = {'image': open(filename, 'rb')}
    r = requests.post(standalone_api_endpoint, files=my_img)

    # convert server response into JSON format.
    print(r.json())
    return r.json()


def test_oblique_recognition():
    # Static image file(s)
    img_file = 'President_Barack_Obama.jpg'

    # Read the image and convert to numpy array
    img_array = convert_img_to_array(img_file)

    # Get the face encodings from the image array
    face_encodings_array = get_encodings(img_array)

    # Convert the numpy arrays to python lists
    face_encodings_list = [arr.tolist() for arr in face_encodings_array]

    # Send a post request to the specific API endpoint with encodings data
    res = requests.post(oblique_api_endpoint, json={"face_encodings": face_encodings_list})

    # Returns server response as JSON format
    print(res.json())
    return res.json()


if __name__ == "__main__":
    # test_standalone_recognition
    test_oblique_recognition()
