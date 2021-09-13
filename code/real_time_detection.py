import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2,os
import tensorflow as tf
# from text_to_speech import tts
# from google.cloud import texttospeech
# from pygame import mixer  # Load the popular external library


if os.path.exists('gesture_regconition_model_4.h5'):
    print("Trained model already exists....\n"
        "Continue executing the program.============>")
else:
    try:
        from mega import Mega
        mega = Mega()
        m = mega.login()  # login using a temporary anonymous account
        m.download_url('https://mega.nz/file/AJgXEKLY#R6AWCd_eKM-EoctOQVqnDkuiB3Ba1MRBwXectSJXaW0')
    except PermissionError:
        print("Ignore permission error and download the model")


datagen = ImageDataGenerator(rescale=1/255.,
                                  samplewise_center=True,
                                  samplewise_std_normalization=True,
                                  brightness_range=[0.8, 2.5],
                                  zoom_range=[0.8, 1.5])
### LOADING MODEL

model = tf.keras.models.load_model('gesture_regconition_model_4.h5')
labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
cap = cv2.VideoCapture(0) # agrument 0 meaning internal camera, when using Rasberry Pi Cam, the argument should be different
x1,y1 = 5,5
x2,y2 = 330,330
IMAGE_SIZE = 64
CROP_SIZE = 330
while True:
    """
    x1,y1---------------
    |                   |
    |                   |
    |                   |
    |_________________x2,y2
    """
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 2)
    # Target area where the hand gestures should be.
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 6, 6), 1)

    # Preprocessing the frame before input to the model.
    cropped_image = frame[5:CROP_SIZE, 5:CROP_SIZE]
    resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    frame_for_model = datagen.standardize(np.float64(reshaped_frame))
    
    cv2.putText(frame, "Place your left hand here",
                (5, 400), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.FILLED)
    
    # Create a blackboard to display predited text
    blackboard = np.zeros(frame.shape, dtype=np.uint8)
    cv2.putText(blackboard, "Hand Gestures to Text", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    
    # Predicting the frame.
    prediction = np.array(model.predict(frame_for_model))
    predicted_class = labels[prediction.argmax()]  # Selecting the max confidence index.

    # Preparing output based on the model's confidence.
    prediction_probability = prediction[0, prediction.argmax()]
    if prediction_probability > 0.5:
        # High pred prob.
        cv2.putText(blackboard, '{} - {:.2f}%'.format(predicted_class, prediction_probability * 100),
                    (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    elif prediction_probability > 0.2 and prediction_probability <= 0.5:
        # Low pred prob.
        cv2.putText(blackboard, 'Maybe {}... - {:.2f}%'.format(predicted_class, prediction_probability * 100),
                    (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    res = np.hstack((frame, blackboard))
    cv2.imshow("Camera", res)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
cap.release()
cv2.destroyAllWindows()
