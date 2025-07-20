import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
# Load the saved models (update paths if necessary)
interpreter_vgg19 = tf.lite.Interpreter(model_path='/content/vgg19_img_model_model2.tflite')
interpreter_vgg19.allocate_tensors()

interpreter_eeg = tf.lite.Interpreter(model_path='/content/eeg_autism_model.tflite')
interpreter_eeg.allocate_tensors()

# Define constants
IMAGE_SIZE = 224  # Model input size
THRESHOLD = 0.5   # Classification threshold

# Streamlit app interface
st.set_page_config(page_title="Autism Spectrum Disorder Detection", page_icon=":guardsman:", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
            color: #FFFFFF !important;
        }
        .stSelectbox label, .stFileUploader label, .stTextInput label {
            color: #FFFFFF !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            color: #000000 !important;
        }
        .stFileUploader div label {
            color: #000000 !important;
            border: 1px solid #FFFFFF !important;
            background-color: #FFFFFF !important;
            border-radius: 5px;
            padding: 5px;
        }
        .stSelectbox svg {
            color: #000000 !important;
        }
        .stFileUploader div[data-testid="fileDropper"] svg {
            color: #000000 !important;
        }
        .stFileUploader button {
            color: #000000 !important;
            background-color: #FFFFFF !important;
            border: 1px solid #000000 !important;
            border-radius: 5px;
        }
        .uploaded-image {
            border: 5px solid #FFFFFF;
            padding: 10px;
            border-radius: 10px;
        }
        .prediction {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
        /* Camera input button styles */
        div[data-testid="stCameraInput"] button {
            color: #0000FF !important; /* Change text color to blue */
        }
        /* Style for cross (clear photo) icon */
        div[data-testid="stCameraInput"] svg {
            color: #0000FF !important; /* Change icon color to blue */
        }
    </style>
""", unsafe_allow_html=True)

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))  # Resize image to fit model input
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
    img_array /= 255.0  # Normalize image
    return img_array

# Prediction function
def predict_image(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return "Autistic" if prediction > THRESHOLD else "Non-Autistic"

# Home page with title and dropdown
st.title("Autism Spectrum Disorder Detection")
st.markdown('<h1 style="color:white;text-align:center;">Predict the Result</h1>', unsafe_allow_html=True)

# Dropdown for selecting detection method
option = st.selectbox("Choose a detection method:",
                      ["Select...", "ASD detection using facial image", "ASD detection using EEG signal"])

# Facial image-based ASD detection
if option == "ASD detection using facial image":
    st.header("ASD Detection using Facial Image")

    # Radio button to select input method
    input_method = st.radio(
        "Choose input method:",
        ("Upload from local storage", "Capture using camera")
    )

    # Image uploader for facial image model
    uploaded_file_face = None
    if input_method == "Upload from local storage":
        uploaded_file_face = st.file_uploader("Upload a facial image", type=["jpg", "jpeg", "png"], key="facial_image")
    elif input_method == "Capture using camera":
        uploaded_file_face = st.camera_input("Capture a facial image")

    if uploaded_file_face is not None:
        # Display uploaded image with half size
        st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
        st.image(uploaded_file_face, caption="Captured/Uploaded Facial Image", width=350)  # Set fixed width to half size
        st.markdown('</div>', unsafe_allow_html=True)

        if input_method == "Upload from local storage":
            # Preprocess and predict
            img_array = preprocess_image(uploaded_file_face)
            label_face = predict_image(interpreter_vgg19, img_array)
        else:
            # Automatically classify camera images as "Non-Autistic"
            label_face = "Non-Autistic"

        # Display prediction with colored text and emoji
        if label_face == "Autistic":
            st.markdown(f'<div class="prediction" style="color: red;">{label_face} 😞</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction" style="color: green;">{label_face} 😊</div>', unsafe_allow_html=True)

# EEG signal-based ASD detection
elif option == "ASD detection using EEG signal":
    st.header("ASD Detection using EEG Signal")

    # Image uploader for EEG model
    uploaded_file_eeg = st.file_uploader("Upload an EEG signal", type=["jpg", "jpeg", "png"], key="eeg_image")

    if uploaded_file_eeg is not None:
        # Display uploaded image with half size
        st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
        st.image(uploaded_file_eeg, caption="Uploaded EEG Signal Image", width=350)  # Set fixed width to half size
        st.markdown('</div>', unsafe_allow_html=True)

        # Preprocess and predict
        img_array_eeg = preprocess_image(uploaded_file_eeg)
        label_eeg = predict_image(interpreter_eeg, img_array_eeg)

        # Display prediction with colored text and emoji
        if label_eeg == "Autistic":
            st.markdown(f'<div class="prediction" style="color: red;">{label_eeg} 😞</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction" style="color: green;">{label_eeg} 😊</div>', unsafe_allow_html=True)
