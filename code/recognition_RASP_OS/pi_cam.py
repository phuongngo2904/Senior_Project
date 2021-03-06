from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import RPi.GPIO as GPIO
import cv2, pyttsx3, argparse, io, os, picamera
import numpy as np
import speech_recognition as sr
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tflite_runtime.interpreter import Interpreter

def set_up_buttons():
  GPIO.setmode(GPIO.BCM)
  GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP) # acitvate_prediction
  GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP) # activate_text_to_speech
  GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP) # delete the latest character recenlty added to the string
  GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP) # activate speech_to_text  
def speech_to_text(r, mic):
    
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
    except:
        return "There was an error! Please try again."
    return text

def text_to_speech(text):
  engine = pyttsx3.init()
  voices = engine.getProperty('voices')
  engine.setProperty('voice',voices[0].id)
  engine.setProperty('rate',150)
  engine.say(text)
  engine.runAndWait()
def load_labels():
  labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
  return labels


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


  labels = load_labels()

  interpreter = Interpreter(model_path="model.tflite")
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  datagen = ImageDataGenerator(rescale=1 / 255.,
                               samplewise_center=True,
                               samplewise_std_normalization=True,
                               brightness_range=[0.8, 3.5],
                               zoom_range=[0.8, 2.5])
  set_up_buttons()
  previous_text = ""
  predicted_string = ""
  predicted_text = ""
  timer = 0
  #sets up microphone
  r = sr.Recognizer()
  mic = sr.Microphone(device_index=2)
  activate_prediction = False # pin17
  activate_text_to_speech = False # pin 16
  activate_speech_to_text = False # pin 23
  with picamera.PiCamera(resolution=(640, 480), framerate=40) as camera:
    camera.start_preview(fullscreen=False,window=(110,20,640,480))
    camera.vflip =True
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
          
        #button  
        if(GPIO.input(17) == False): activate_prediction = True
        #joystick select
        if(GPIO.input(16) == False): activate_text_to_speech = True
        #joystick select
        if(GPIO.input(23) == False): activate_speech_to_text = True
        #joystick down
        if(GPIO.input(27) == False): #stop the prediction
            activate_prediction = False
            previous_text = ""
            predicted_string = ""
            predicted_text = ""
            timer = 0
            camera.annotate_text = '%s\n\n\n%s %.2f' % (predicted_string,labels[label_id], prob*100)
        stream.seek(0)       
        image = Image.open(stream).convert('RGB').resize((width, height),
                                                   Image.ANTIALIAS)
        image = datagen.standardize(np.float64(image))
        if activate_text_to_speech:
            text_to_speech(predicted_string)
            activate_text_to_speech=False
            previous_text = ""
            predicted_string = ""
            predicted_text = ""
            timer = 0
            camera.annotate_text = '%s\n\n\n%s %.2f' % (predicted_string,labels[label_id], prob*100)
        if activate_speech_to_text:
            activate_speech_to_text = False
            camera.annotate_text = '%s' % ("SPEECH TO TEXT MODE")
            camera.annotate_text = '%s' % (speech_to_text(r, mic))
        if timer > 10 and predicted_text not in labels[26:-1]:
            if predicted_text== "space":
                predicted_string += " "
            else:
                predicted_string += predicted_text
            timer = 0
        if activate_prediction:
            previous_text = predicted_text
            results = classify_image(interpreter, image)
            label_id, prob = results[0]
            predicted_text = labels[label_id]
            stream.seek(0)
            stream.truncate()
            
            if previous_text == predicted_text: timer += 1
            else: timer = 0
            print(f"TIMER IS {timer}")
            camera.annotate_text = '%s\n\n\n%s %.2f' % (predicted_string,labels[label_id], prob*100)
       
    finally:
        camera.stop_preview()
        
              
if __name__ == '__main__':
  if os.path.exists('model.tflite'):
       print("Trained model already exists....\n"
            "Continue executing the program.============>")
  else:
      try:
          from mega import Mega
          mega = Mega()
          m = mega.login()  # login using a temporary anonymous account
          m.download_url('https://mega.nz/file/WQdAjZZL#IhPKLt4sYcpWtQXMIVQQrDNhF7UvPOF0lv2e5wXAjlc')
      except PermissionError:
          print("Ignore permission error and download the model")
  main()
  #text_to_speech("Hello")
