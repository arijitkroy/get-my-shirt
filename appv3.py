import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from prediction import process_image, load_yolo_ultralytics, TSHIRT_OUT_DIR

# Default page configuration
st.set_page_config(
    page_title="Get my Shirt",
    page_icon="./assets/logo.png",
    initial_sidebar_state="expanded"
)

# Consolidated Custom CSS to override default theme styles
st.markdown(
    """
    <style>
    /* Main container background */
    .stApp {
        background-color: #e0f7fa !important; /* Light cyan background */
    }
    /* Sidebar background and text */
    [data-testid="stSidebar"] {
        background-color: #006064 !important; /* Dark cyan */
        color: white !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Heading styles */
    h1, h2, h3, h4, h5, h6 {
        color: #004d40 !important; /* Dark green */
    }
    /* Button styling */
    div.stButton > button {
        background-color: #00796b !important; /* Teal */
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 4px !important;
    }
    div.stButton > button:hover {
        background-color: #004d40 !important; /* Darker green on hover */
        color: white !important;
    }
    /* Input fields styling */
    input, textarea, .stTextInput>div>div>input {
        background-color: #b2dfdb !important; /* Light teal */
        color: #004d40 !important; /* Dark green */
    }
    /* st.radio customization */
    /* Targeting container labels and spans */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] span {
        color: #004d40 !important;
        font-family: 'Helvetica Neue', sans-serif !important;
        font-size: 1rem !important;
    }
    /* Customize the radio button accent color */
    div[data-testid="stRadio"] input[type="radio"] {
        accent-color: #00796b !important;
    }
    /* st.file_uploader customization */
    div[data-testid="stFileUpload"] {
        background-color: #b2dfdb !important;
        border: 1px solid #00796b !important;
        border-radius: 4px !important;
        padding: 10px !important;
    }
    div[data-testid="stFileUpload"] label {
        color: #004d40 !important;
        font-family: 'Helvetica Neue', sans-serif !important;
        font-size: 1rem !important;
    }
    div[data-testid="stFileUpload"] button {
        background-color: #00796b !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.4rem 0.8rem !important;
    }
    div[data-testid="stFileUpload"] button:hover {
        background-color: #004d40 !important;
    }
    /* Custom header and paragraph styling */
    .custom-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .custom-subtitle {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.5rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 20px;
    }
    .custom-paragraph {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1rem;
        color: #7f8c8d;
        line-height: 1.6;
        text-align: center;
        margin: 0 auto 20px;
        width: 80%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display logo at center
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("./assets/logo.png")

# Subtitle and description
st.markdown('<div class="custom-subtitle">Upload an Image or Capture from Camera</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="custom-paragraph">'
    'This app uses a custom prediction model to detect your T-shirt and predict its size. '
    'Simply choose your input type below, provide an image, and click the analyze button!'
    '</div>',
    unsafe_allow_html=True
)

# Load the YOLO model (adjust the path if needed)
model = load_yolo_ultralytics("best.pt")

# Input selection: image upload or camera capture
input_option = st.radio("Select Input Type", ["Image Upload", "Camera Capture"])

uploaded_image = None
if input_option == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif input_option == "Camera Capture":
    captured_file = st.camera_input("Capture an image")
    if captured_file is not None:
        file_bytes = np.asarray(bytearray(captured_file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Preview the input image if available
if uploaded_image is not None:
    st.image(uploaded_image, channels="BGR", caption="Input Image")

# Center the "Analyze" button using a three-column layout
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    analyze_clicked = st.button("Analyze")

if analyze_clicked:
    if uploaded_image is None:
        st.error("Please provide an image first.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_filename = temp_file.name
            cv2.imwrite(temp_filename, uploaded_image)
        st.write("Processing image, please wait...")
        size_category = process_image(temp_filename, model)
        if size_category is not None:
            st.success(f"Predicted T-Shirt Size: {size_category}")
            detected_image_path = os.path.join(TSHIRT_OUT_DIR, "detected_" + os.path.basename(temp_filename))
            if os.path.exists(detected_image_path):
                detected_image = cv2.imread(detected_image_path)
                # Display input and processed images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_image, channels="BGR", caption="Input Image")
                with col2:
                    st.image(detected_image, channels="BGR", caption="Detected T-Shirt with Bounding Box")
            else:
                st.warning("Detected bounding box image not found.")
        else:
            st.error("T-Shirt detection or analysis failed.")