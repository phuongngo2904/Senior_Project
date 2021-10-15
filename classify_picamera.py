# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to classify objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import argparse
import io
import time
import numpy as np
import picamera
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tflite_runtime.interpreter import Interpreter


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  datagen = ImageDataGenerator(rescale=1 / 255.,
                               samplewise_center=True,
                               samplewise_std_normalization=True,
                               brightness_range=[0.8, 2.5],
                               zoom_range=[0.8, 1.5])
  x1, y1 = 5, 5
  x2, y2 = 330, 330
  IMAGE_SIZE = 64
  CROP_SIZE = 330
  timer = 0
  previous_text = ""
  predicted_string = ""
  key_pressed = False
  predicted_text = ""
  activate_text_to_speech = False
  reset_predicted_string = False
  cap = cv2.VideoCapture(0)
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
    results = classify_image(interpreter, frame_for_model)
    label_id, prob = results[0]
    if prob > 0.5:
      # High pred prob.
      cv2.putText(blackboard, '{} - {:.2f}%'.format(labels[label_id], prob * 100),
                  (30, 300), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    elif prob > 0.2 and prob <= 0.5:
      # Low pred prob.
      cv2.putText(blackboard,
                  'Maybe {}... - {:.2f}%'.format(labels[label_id], prob * 100),
                  (30, 300), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    res = np.hstack((frame, blackboard))
    cv2.imshow("Camera", res)

    interrupt = cv2.waitKey(1)
    if interrupt & 0xFF == 27:  # esc key: abort the program
      break
  cap.release()
  cv2.destroyAllWindows()

  """
  with picamera.PiCamera(resolution=(330, 140), framerate=30) as camera:
    camera.start_preview()
    camera.hflip = True
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize((width, height),
                                                         Image.ANTIALIAS)
        start_time = time.time()
        results = classify_image(interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000
        label_id, prob = results[0]
        stream.seek(0)
        stream.truncate()
        camera.annotate_text = '%s %.2f\n%.1fms' % (labels[label_id], prob,
                                                    elapsed_ms)
    finally:
      camera.stop_preview()
    """
if __name__ == '__main__':
  main()