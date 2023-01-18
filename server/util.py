import os
import numpy as np
import base64
import joblib
import cv2
import json
import matplotlib
from matplotlib import pyplot as plt 
from wavelet import w2d


__face_cascade = None
__eye_cascade = None
__class_number_to_name = None
__class_name_to_number = None
__model = None


# this method will just load all the variables and keep it as global for later use
def initialize_variables():
    print("loading saved model ... start")
    global __face_cascade
    global __eye_cascade
    global __class_name_to_number
    global __class_number_to_name
    global __model

    __face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    __eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')    

    with open('./server/model/class_dictionary.json',"r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {k:v for v, k in __class_name_to_number.items()}
    
    if __model is None:
        with open('./server/model/svm_model.pkl', "rb") as f:
            __model = joblib.load(f)
    print("loading saved model ... done")


# classify image will get the cropped faces from the image and then convert it to wavelet, 
# resize it and make a hot a vector from it to pass to the model
def classify_image(img_base64, img_path=None):
    cropped_faces = get_cropped_image_if_2_eyes_exist(img_path, img_base64)
    result = []
    for face in cropped_faces:
        scalled_img = cv2.resize(face, (64,64))
        img_wav = w2d(face, 'db1', 5)
        scalled_img_wav = cv2.resize(img_wav, (64,64))
        combined_img = np.vstack((scalled_img.reshape(64 * 64 *3, 1), scalled_img_wav.reshape(64 * 64, 1)))
        len_img_array = 64 * 64 * 3 + 64*64
        input = combined_img.reshape(1, len_img_array).astype(float)
        model_result = __model.predict(input)
        result.append({
            'class': class_number_to_name(model_result[0]),
            'class_probability': np.around(__model.predict_proba(input) * 100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })
    return result

# just a simple utitly function to ocnver the class number to class name(string)
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

# get the cropped faces from an image
def get_cropped_image_if_2_eyes_exist(img_path, image_base64):
    if img_path:
        img = cv2.imread(img_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64)
    if img is None:
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = __face_cascade.detectMultiScale(img_gray, 1.3,5)
    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = img_gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = __eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        if len(eyes) >=2:
            cropped_faces.append(roi_color)
    return cropped_faces 


def get_b64_test_image():
    with open("lionelmessi38.txt") as f:
        return f.read()

# convert a string image base64 to a cv2 image 
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


if __name__ == '__main__':
    initialize_variables()
    # img_string = get_b64_test_image()
    prediction = classify_image(None, './testimages/lionelmessi38.jpeg')
    print(prediction)
    # for img_path in os.scandir("./testimages"):
    #     prediction = classify_image(None, img_path)
    #     print(img_path, ' prediction is ', prediction)
    