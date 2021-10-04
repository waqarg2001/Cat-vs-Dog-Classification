import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

model = tf.keras.models.load_model("cat-vs-dog.h5")
st.title('Simple Cat Dog Classifier')
uploaded_file = st.file_uploader("Choose a image file", type=["jpg",'jpeg','png'])
map_dict={0:'Cat',1:'Dog'}



if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(150,150))
    st.image(opencv_image, channels="RGB")

    resized = vgg16_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("This is a {}".format(map_dict [prediction]))