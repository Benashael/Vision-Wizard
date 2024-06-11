import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Set up Streamlit app
st.set_page_config(page_title="Vision Wizard", page_icon="🧙‍♂️", layout="wide")

st.title("Vision Wizard 🧙‍♂️✨: Simplifying Computer Vision Tasks")

page = st.sidebar.radio("**🌐 Select a Page**", ["Home Page 🏠", "Image Resizing 📏🔄", "Image Grayscale Conversion 🌑🔄", "Edge Detection ✂️🔍", "Image Rotation 🔄↪️", "Image Cropping ✂️🖼️", "Image Flipping ↔️🔄", "Color Space Conversion 🌈🔄", "Image Blurring 🌫️🔄", "Histogram Equalization 📊✨", "Face Detection 😊🔍"])

def get_image_input():
    # Check if image is already in session state
    if 'image' not in st.session_state:
        st.session_state.image = None

    # Function to check image complexity
    def is_image_complex(image):
        img_array = np.array(image)
        return img_array.shape[0] * img_array.shape[1] > 10000 * 10000  # Example threshold

    # Function to check image size
    def is_file_size_ok(file):
        file.seek(0, 2)  # Move to end of file
        file_size = file.tell()
        file.seek(0, 0)  # Move back to start of file
        return file_size <= 10 * 1024 * 1024  # 10 MB limit

    # Choose input method
    input_method = st.radio("Select Image Input Method: 📸",
                            ("📁 Upload Image", "📷 Capture Image", "🖼️ Use Example Image"))

    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose an image 📂", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            if not is_file_size_ok(uploaded_file):
                st.error("❌ Uploaded image is too large. Please upload an image smaller than 10MB.")
                return None
            try:
                image = Image.open(uploaded_file)
                if is_image_complex(image):
                    st.error("❌ Uploaded image is too complex to process. Please upload a simpler image.")
                    return None
                st.session_state.image = image
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                return None
        else:
            st.error("❌ Error: Please upload a file.")

    elif input_method == "🖼️ Use Example Image":
        example_image_path = "example.jpg"  # Ensure this file is in the same directory
        try:
            image = Image.open(example_image_path)
            st.session_state.image = image
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            return None

    elif input_method == "📷 Capture Image":
        capture_image = st.camera_input("Capture an image")
        if capture_image is not None:
            try:
                image = Image.open(capture_image)
                if is_image_complex(image):
                    st.error("❌ Captured image is too complex to process. Please capture a simpler image.")
                    return None
                st.session_state.image = image
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                return None
        else:
            st.error("❌ Error: Please upload a file.")

    return st.session_state.image

# List of pages to exclude the common input section
exclude_input_pages = ["Home Page 🏠"]

if page not in exclude_input_pages:
    # Common input section for pages not in the exclude list
    image = get_image_input()

    # Add a button to clear the session state
    if st.button("🗑️ Clear Input"):
        clear_session_state()
        st.experimental_rerun()
      
    st.info("⚠️ Click '🗑️ Clear Input' to reset the text input and file upload fields. This will clear all entered data and allow you to start fresh.")

# Page 2
if page == "Image Resizing 📏🔄":
    if image is not None:
        st.header("Image Resizing 📏🔄")
        width, height = image.size
        st.write(f"Original Dimensions: {width} x {height}")
        new_width = st.number_input("New Width", value=width, min_value=1, max_value=3000)
        new_height = st.number_input("New Height", value=height, min_value=1, max_value=3000)
        resized_image = image.resize((new_width, new_height))
        st.image(resized_image, caption='Resized Image', use_column_width=True)
        img_array = np.array(resized_image)
        resized_img = Image.fromarray(img_array)
        st.download_button(label="Download Resized Image", 
                           data=resized_img.tobytes(), 
                           file_name="resized_image.png", 
                           mime="image/png")
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")
