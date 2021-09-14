import os
import tensorflow as tf
from gesture_reg import gesture_regcognize

if __name__ == "__main__":
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
    ### LOADING MODEL
    g_model = tf.keras.models.load_model('gesture_regconition_model_4.h5')
    gesture_reg = gesture_regcognize()
    gesture_reg.run_function(g_model)
