import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your saved .h5 model (adjust path if needed)
model = load_model('vehicle_recognition_model_.h5')

# Replace this with your actual class names in the order your model expects
class_names = ['bus', 'car', 'bike', 'truck']  # update this list to your classes

st.title("Vehicle Recognition App")

# Choose input method
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

def preprocess_image(image: Image.Image):
    img_size = 224
    image_resized = image.resize((img_size, img_size))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_and_display(image: Image.Image):
    st.image(image, caption="Input Image", use_column_width=True)
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = preds[0][pred_index]
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        predict_and_display(image)

else:  # Webcam option
    picture = st.camera_input("Take a picture")
    if picture is not None:
        image = Image.open(picture).convert('RGB')
        predict_and_display(image)
