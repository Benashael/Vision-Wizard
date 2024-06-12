import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Set up Streamlit app
st.set_page_config(page_title="Vision Wizard", page_icon="🧙‍♂️", layout="wide")

st.title("Vision Wizard 🧙‍♂️✨: Simplifying Computer Vision Tasks")

page = st.sidebar.radio("**🌐 Select a Feature**", ["Home Page 🏠", "Image Resizing 📏🔄", "Image Grayscale Conversion 🌑🔄", "Edge Detection ✂️🔍", "Image Rotation 🔄↪️", "Image Cropping ✂️🖼️", "Image Flipping ↔️🔄", "Color Space Conversion 🌈🔄", "Image Blurring 🌫️🔄", "Histogram Equalization 📊✨", "Face Detection 😊🔍"])

def clear_session_state():
    st.session_state.pop("input_method", None)
    st.session_state.pop("uploaded_file", None)
    st.session_state.pop("capture_image", None)
    st.session_state.pop("image", None)
    
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
    input_method = st.radio("**Select Image Input Method:** 📸",
                            ("📁 Upload Image", "📷 Capture Image", "🖼️ Use Example Image"))

    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose an image 📂", type=["jpg", "jpeg", "png"])
        if st.button("📁 Submit Image"):
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
                st.error("❌ Error: Please upload an image.")

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
        if st.button("📷 Submit Image"):
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
                st.error("❌ Error: Please take a photo.")

    return st.session_state.image

# List of pages to exclude the common input section
exclude_input_pages = ["Home Page 🏠"]

if page not in exclude_input_pages:
    # Common input section for pages not in the exclude list
    image = get_image_input()

    # Add a button to clear the session state
    if st.button("🗑️ Clear Input"):
        clear_session_state()
      
    st.info("⚠️ Click '🗑️ Clear Input' to reset the image input fields. This will clear all entered data and allow you to start fresh.")

# Page 2
if page == "Image Resizing 📏🔄":
    st.header("📏🔄 Image Resizing Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        width, height = image.size
        st.subheader(f"**Original Image Dimensions:** {width} x {height}")
        new_width = st.number_input("New Width", value=width, min_value=1, max_value=6000)
        new_height = st.number_input("New Height", value=height, min_value=1, max_value=6000)
        resized_image = image.resize((new_width, new_height))
        if st.button("🔄 Resize Image"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader('Resized Image')
            st.image(resized_image, caption='Resized Image', use_column_width=True)
            img_array = np.array(resized_image)
            resized_img = Image.fromarray(img_array)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 3
elif page == "Image Grayscale Conversion 🌑🔄":
    st.header("🌑🔄 Image Grayscale Conversion Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        if st.button("🌑 Perform Grayscale Conversion"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("⬛ Grayscale Image") 
            img_array = np.array(image)
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            st.image(gray_img, caption='Grayscale Image', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")
        
# Page 4
elif page == "Edge Detection ✂️🔍":
    st.header("✂️🔍 Edge Detection Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        if st.button("🔍 Perform Edge Detection"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("🔍 Image with Detected Edges") 
            img_array = np.array(image)
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_img, 100, 200)
            st.image(edges, caption='Edges Detected', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 5
elif page == "Image Rotation 🔄↪️":
    st.header("🔄↪️ Image Rotation Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        angle = st.slider("Rotate Angle", min_value=0, max_value=360, value=0)
        if st.button("↪️ Rotate Image"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("🔄 Rotated Image") 
            rotated_image = image.rotate(angle)
            st.image(rotated_image, caption=f'Image Rotated by {angle} degrees', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 6
elif page == "Image Cropping ✂️🖼️":
    st.header("✂️🖼️ Image Cropping Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        width, height = image.size
        x = st.number_input("X Coordinate", value=0, min_value=0, max_value=width-1, step=1)
        y = st.number_input("Y Coordinate", value=0, min_value=0, max_value=height-1, step=1)
        max_crop_width = width - x
        max_crop_height = height - y
        new_width = st.number_input("Crop Width", value=max_crop_width, min_value=1, max_value=max_crop_width, step=1)
        new_height = st.number_input("Crop Height", value=max_crop_height, min_value=1, max_value=max_crop_height, step=1)
        if st.button("✂️ Crop Image"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("✂️ Cropped Image")
            cropped_image = image.crop((x, y, x + new_width, y + new_height))
            st.image(cropped_image, caption='Cropped Image', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 7
elif page == "Image Flipping ↔️🔄":
    st.header("↔️🔄 Image Flipping Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        flip_option = st.radio("Flip Option", ["Horizontal Flip", "Vertical Flip"])
        img_array = np.array(image)
        if flip_option == "Horizontal Flip":
            flipped_image = cv2.flip(img_array, 1)
        else:
            flipped_image = cv2.flip(img_array, 0)
        if st.button("↔️ Flip Image"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("↔️ Flipped Image")
            st.image(flipped_image, caption=f'Image with {flip_option}', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")
