import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random,os,cv2,np

def view_random_image(target_dir, target_class):
  # Set up the target directory.
  target_folder = target_dir+ "/" +target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder),1)
  print(random_image)
  # Read in the image using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");
  print(f"Image Shape: {img.shape}") # show the shape of the image
  return img

def test_model_w_images(model, TEST_DIR):
  IMAGE_SIZE = 64
  for test_image in (os.listdir(TEST_DIR)):
    #random_img = random.sample(os.listdir(TEST_DIR+ "/" + test_image),1)
    path = TEST_DIR + "/" + test_image
    #print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    plt.figure()
    plt.axis('Off')
    plt.imshow(img) 
    img = np.array(img) / 255.
    img = img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    print(img.shape)
    img = datagen.standardize(img) 
    prediction = np.array(model.predict(img))
    actual = test_image.split('_')[0]
    predicted = labels[prediction.argmax()]
    print('Actual class: {} \n Predicted class: {}'.format(actual, predicted)) 
    plt.show()
    
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();
