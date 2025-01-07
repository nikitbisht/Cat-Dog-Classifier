import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np

def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image)
    image_array = image_array / 255.0
    test_input = np.expand_dims(image_array, axis=0)
    return test_input



model = load_model('Model/Cat_dog_Classifier_Model.h5')
st.title("🐾 Cat vs Dog Classifier")

uploaded_image = st.file_uploader("Upload your image here:", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image",use_column_width=False)

    if st.button("Predict"):
        test_input = preprocess_image(image)
        prediction = model.predict(test_input)
        class_name = {0:'Cat',1:'Dog'}
        flag = 1 if prediction[0][0] >= 0.5 else 0
        # st.write(prediction)
        st.markdown(f"<h2 style='text-align: center; color: #FFFF00;'>This is a <span style='font-weight: bold; font-size: 48px;'>{class_name[flag]}</span> 🐾</h2>", unsafe_allow_html=True)
