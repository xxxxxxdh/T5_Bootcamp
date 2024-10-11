import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Set page config
st.set_page_config(page_title="🚗 Self-Driving Car Object Detector", page_icon="🚦", layout="wide")

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

st.title("🚗 Self-Driving Car Object Detector 🚦")

# Introduction and Goal
st.write("""
## Introduction
Welcome to the Self-Driving Car Object Detector! This application simulates the vision system of an autonomous vehicle, 
demonstrating how AI can identify and classify objects on the road.

## Goal
Our primary goal is to showcase the object detection capabilities crucial for self-driving cars. By processing uploaded
images from a car's perspective, we aim to illustrate how AI contributes to safe navigation in complex traffic scenarios.

### Key Features:
- 🖼️ Image Upload: Upload your own image from a car's perspective.
- 🔍 Object Detection: Utilize YOLO (You Only Look Once) to identify objects in the image.
- 📊 Visualization: See detected objects highlighted directly on the image.
- 📋 Detailed Results: Explore a list of detected objects with their classifications.

Let's explore how a self-driving car 'sees' the world!
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button("🔍 Detect Objects"):
            with st.spinner("Detecting objects..."):
                # Convert PIL Image to numpy array
                img_array = np.array(image)

                # Perform prediction
                results = model.predict(img_array, conf=0.2, iou=0.5)

                # Plot the results
                for result in results:
                    plot = result.plot()
                    plot_rgb = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
                    st.image(plot_rgb, caption="Detection Result", use_column_width=True)

                # Display detection information
                st.subheader("🔎 Detection Results:")
                for result in results:
                    for box in result.boxes:
                        class_id = result.names[box.cls[0].item()]
                        cords = box.xyxy[0].tolist()
                        cords = [round(x) for x in cords]
                        emoji = "🚗" if class_id == "car" else "🚛" if class_id == "truck" else "🚶" if class_id == "person" else "🚲" if class_id == "bicycle" else "🚦" if class_id == "traffic light" else "🔍"
                        st.write(f"{emoji} {class_id.capitalize()}: Coordinates: {cords}")

st.write("---")
st.write("🚀 Powered by YOLOv8 and Streamlit")