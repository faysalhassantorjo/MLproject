import joblib
import cv2
import pywt
import json
import numpy as np
import sklearn

face_cascade = cv2.CascadeClassifier('D:/DM_ML/my_ml_project/ml_app/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('D:/DM_ML/my_ml_project/ml_app/opencv/haarcascades/haarcascade_eye.xml')

prediction_class = {}
model = None

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load Haar cascade files.")
else:
    print("Cascade files loaded successfully.")

def save_model():
    global model, prediction_class  
    
    print("Loading saved artifacts...start")

    try:
        with open("D:/DM_ML/my_ml_project/ml_app\predictionModel/class_dictionary.json", "r") as f:
            class_name_to_number = json.load(f)
            prediction_class = {v: k for k, v in class_name_to_number.items()}
            print('Class Name Mapping:', prediction_class)
    except FileNotFoundError:
        print("Error: class_dictionary.json not found.")
        return
    
    if model is None:
        try:
            with open('D:\DM_ML\my_ml_project\ml_app\predictionModel\prediction_model.pkl', 'rb') as f:
                model = joblib.load(f)
                print(model)
                print('Model loaded successfully.')
        except FileNotFoundError:
            print("Error: prediction_model.pkl not found.")

# Call save_model once to load the model and class dictionary at import
save_model()

def get_cropped_image(image_array):
    # Check if image is loaded correctly
    if image_array is None:
        print("Error: Could not read the image.")
        return None

    # Ensure image is in uint8 format for OpenCV processing
    img = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)

    # Resize large images to optimize face detection performance
    if img.shape[0] > 800:  # For example, resize if height > 800px
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect faces with tuned parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Detected {len(faces)} faces.")  # Debug: Output number of faces detected

    # Loop through detected faces to find eyes
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # Region of interest for grayscale face
        roi_color = img[y:y+h, x:x+w]  # Region of interest for color face

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
        print(f"Detected {len(eyes)} eyes in face region.")  # Debug: Output number of eyes detected
        
        if len(eyes) >= 1:
            return roi_color  # Return the color region of the face if two eyes are detected

    print("Face or eyes not detected in the image.")  # Debug message if detection fails
    return None



def make_prediction(input_data):
    if model is None:
        print("Error: Model not loaded.")
        return "Model not loaded."
    
    processed_image = get_cropped_image(input_data)
    if processed_image is not None:
        scalled_raw_img = cv2.resize(processed_image, (32, 32))
        img_har = w2d(processed_image, 'haar', 1)
        scalled_img_har = cv2.resize(img_har, (32, 32))  # Ensure img_har is single-channel

        scalled_raw_img_flat = scalled_raw_img.reshape(32 * 32 * 3, 1)
        scalled_img_har_flat = scalled_img_har.reshape(32 * 32, 1)

        combined_img = np.vstack((scalled_raw_img_flat, scalled_img_har_flat))

        len_image_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_image_array).astype(float)
        
        # Make prediction
        prediction = model.predict(final)
        return prediction_class.get(prediction[0], "Unknown class")
    else:
        return f"Face or eyes not detected in the image.{processed_image}"

#image preprocessing
def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)   
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    coeffs_H = list(coeffs)  
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H
