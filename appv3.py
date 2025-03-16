import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from prediction import process_image, load_yolo_ultralytics, TSHIRT_OUT_DIR
from scrape import getDetails
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(
    page_title="Get my Shirt",
    page_icon="./assets/logo.png",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .custom-subtitle {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 1.5rem;
            color: #ffffff;
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
    """, unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("./assets/logo.png")

st.markdown('<div class="custom-subtitle">Upload an Image or Capture from Camera</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="custom-paragraph">'
    'This app uses a custom prediction model to detect your T-shirt and predict its size. '
    'Simply choose your input type below, provide an image, and click the analyze button!'
    '</div>',
    unsafe_allow_html=True
)

model = load_yolo_ultralytics("best.pt")

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

if uploaded_image is not None:
    st.image(uploaded_image, channels="BGR", caption="Input Image")

col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    analyze_clicked = st.button("Analyze Image")

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
            detected_image_path = os.path.join(TSHIRT_OUT_DIR, "detected_" + os.path.basename(temp_filename))
            if os.path.exists(detected_image_path):
                detected_image = cv2.imread(detected_image_path)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_image, channels="BGR", caption="Input Image")
                with col2:
                    st.image(detected_image, channels="BGR", caption="Detected T-Shirt with Bounding Box")
                st.success(f"Predicted T-Shirt Size: {size_category}")
            else:
                st.warning("Detected bounding box image not found.")

            st.markdown("### Recommended T-Shirts")
            shirts = getDetails()
            cols = st.columns(5)
            for i, shirt in enumerate(shirts):
                with cols[i]:
                    response = requests.get(shirt["Product Image"])
                    img = Image.open(BytesIO(response.content))
                    img = img.resize((150, 150))
                    st.image(img, caption=f"Rating: {shirt["Rating"]}", use_container_width=True)
                    st.write(f"**{shirt['Product Name']}**")
                    st.write(f"Price: {shirt['Price']}")
                    st.markdown(f"[View Product]({shirt['Product Link']})")
        else:
            st.error("T-Shirt detection or analysis failed.")