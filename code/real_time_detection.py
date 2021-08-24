import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2,os
import tensorflow as tf

if os.path.exists('gesture_regconition_model_3.h5'):
    print("Trained model already exists.")
else:
    try:
        from mega import Mega
        mega = Mega()
        m = mega.login()  # login using a temporary anonymous account
        m.download_url('https://mega.nz/file/gUpRwQDb#xumwfAZcMNd8Bt9gIHg515-R14KwTia4NoAFz9Y4zgM')
    except PermissionError:
        print("Ignore permission error and download the model")


datagen = ImageDataGenerator(rescale=1/255.,
                                  samplewise_center=True,
                                  samplewise_std_normalization=True,
                                  brightness_range=[0.8, 2.5],
                                  zoom_range=[0.4, 0.8])
### LOADING MODEL
#model = tf.keras.models.load_model('gesture_regconition_model_2_100.h5')
model = tf.keras.models.load_model('gesture_regconition_model_3.h5')
labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
cap = cv2.VideoCapture(0)
x1,y1 = 5,5
x2,y2 = 320,320
IMAGE_SIZE = 64
CROP_SIZE = 320
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
    cropped_image = frame[0:CROP_SIZE, 0:CROP_SIZE]
    resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    frame_for_model = datagen.standardize(np.float64(reshaped_frame))

    # Predicting the frame.
    prediction = np.array(model.predict(frame_for_model))
    predicted_class = labels[prediction.argmax()]  # Selecting the max confidence index.

    # Preparing output based on the model's confidence.
    prediction_probability = prediction[0, prediction.argmax()]
    if prediction_probability > 0.5:
        # High confidence.
        cv2.putText(frame, '{} - {:.2f}%'.format(predicted_class, prediction_probability * 100),
                    (5, 400), cv2.FONT_ITALIC, 2, (0, 0, 0), 2, cv2.FILLED)
    elif prediction_probability > 0.2 and prediction_probability <= 0.5:
        # Low confidence.
        cv2.putText(frame, 'Maybe {}... - {:.2f}%'.format(predicted_class, prediction_probability * 100),
                    (5, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.FILLED)
    else:
        # No confidence.
        cv2.putText(frame, labels[-2], (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
cap.release()
cv2.destroyAllWindows()