import streamlit as st
import tensorflow as tf
import gdown
import io
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    # https://drive.google.com/file/d/1mUhkSozH2pfFxz_WHLALnGAo3DkFmCd2/view?usp=sharing
    url='https://drive.google.com/uc?id=1mUhkSozH2pfFxz_WHLALnGAo3DkFmCd2'
    gdown.download(url, 'model_downloaded.tflite')
    interpreter = tf.lite.Interpreter(model_path='model_downloaded.tflite')
    interpreter.allocate_tensors()

    return interpreter

def load_image():
    uploaded_file = st.file_uploader('click here to upload a image', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('The image was loaded')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0

        image = np.expand_dims(image, axis=0)

        return image

def classify(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    st.write(f'probability: {output_data*100}%')

    # return output_data[0]

def main():
    st.set_page_config(
        page_title="Cataract Detector",
        page_icon="👁️"
    )

    st.write("# Detects the presence of cataract based on an eye photo")

    # load model
    interpreter = load_model()

    # load image
    image = load_image()

    # classify
    if image is not None:
        classify(interpreter, image)

if __name__ == "__main__":
    main()