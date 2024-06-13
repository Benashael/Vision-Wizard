import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Set up Streamlit app
st.set_page_config(page_title="Vision Wizard", page_icon="🧙‍♂️", layout="wide")

st.title("Vision Wizard 🧙‍♂️✨: Simplifying Computer Vision Tasks")

page = st.sidebar.radio("**🌐 Select a Feature**", ["Home Page 🏠", "Image Resizing 📏🔄", "Image Grayscale Conversion 🌑🔄", "Edge Detection ✂️🔍", "Image Rotation 🔄↪️", "Image Cropping ✂️🖼️", "Image Flipping ↔️🔄", "Color Space Conversion 🌈🔄", "Image Brightness/Contrast Adjustment ☀️🌑", "Image Blurring 🌫️🔄", "Histogram Equalization 📊✨", "Face Detection 😊🔍", "Image Segmentation 🧩📦"])

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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    # Convert PIL image to grayscale OpenCV image
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces
    
# List of pages to exclude the common input section
exclude_input_pages = ["Home Page 🏠"]

if page not in exclude_input_pages:
    # Common input section for pages not in the exclude list
    image = get_image_input()

    # Add a button to clear the session state
    if st.button("🗑️ Clear Input"):
        clear_session_state()
      
    st.info("⚠️ Click '🗑️ Clear Input' to reset the image input fields. This will clear all entered data and allow you to start fresh.")

# Page 1
if page == "Home Page 🏠":
    st.subheader("Discover the power of computer vision with easy-to-use tools and techniques in Vision Wizard.")
    st.markdown("___________")
    st.header("Explore the following features:")
    st.markdown("""
    🔸 **Image Resizing 📏🔄**: Resize your images to different dimensions.
    
    🔸 **Image Grayscale Conversion 🌑🔄**: Convert images to grayscale.
    
    🔸 **Edge Detection ✂️🔍**: Detect edges in images.
    
    🔸 **Image Rotation 🔄↪️**: Rotate images to a desired angle.
    
    🔸 **Image Cropping ✂️🖼️**: Crop images to focus on specific areas.
    
    🔸 **Image Flipping ↔️🔄**: Flip images horizontally or vertically.
    
    🔸 **Color Space Conversion 🌈🔄**: Convert images between different color spaces.
    
    🔸 **Image Brightness/Contrast Adjustment ☀️🌑**: Adjust the brightness and contrast of images.
    
    🔸 **Image Blurring 🌫️🔄**: Apply blurring effects to images.
    
    🔸 **Histogram Equalization 📊✨**: Enhance the contrast of images using histogram equalization.
    
    🔸 **Face Detection 😊🔍**: Detect faces in images.
    
    🔸 **Image Segmentation 🧩📦**: Segment images into distinct regions.
    """)

# Page 2
elif page == "Image Resizing 📏🔄":
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
        method = st.radio("✂️ **Select Edge Detection Method**", ["Canny", "Sobel", "Laplacian"])
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if method == "Canny":
            threshold1 = st.slider("Threshold 1", 0, 255, 100)
            threshold2 = st.slider("Threshold 2", 0, 255, 200)
            edges = cv2.Canny(opencv_image, threshold1, threshold2)
        elif method == "Sobel":
            dx = st.slider("dx", 0, 1, 1)
            dy = st.slider("dy", 0, 1, 1)
            ksize = st.slider("Kernel Size (must be odd)", 1, 31, 3, step=2)
            edges = cv2.Sobel(opencv_image, cv2.CV_64F, dx, dy, ksize=ksize)
            edges = cv2.convertScaleAbs(edges)  # Convert the result to uint8
        elif method == "Laplacian":
            ksize = st.slider("Kernel Size (must be odd)", 1, 31, 3, step=2)
            edges = cv2.Laplacian(opencv_image, cv2.CV_64F, ksize=ksize)
            edges = cv2.convertScaleAbs(edges)  # Convert the result to uint8
        if st.button("🔍 Perform Edge Detection"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("🔍 Image with Detected Edges") 
            st.image(edges, caption=f'Edges Detected using {method}', use_column_width=True)
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
        flip_option = st.radio("🔄 **Flip Option**", ["Horizontal Flip", "Vertical Flip", "Diagonal Flip"])
        img_array = np.array(image)
        if flip_option == "Horizontal Flip":
            flipped_image = cv2.flip(img_array, 1)
        elif flip_option == "Diagonal Flip":
            flipped_image = image.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
        else:
            flipped_image = cv2.flip(img_array, 0)
        if st.button("↔️ Flip Image"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("↔️ Flipped Image")
            st.image(flipped_image, caption=f'Image with {flip_option}', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 8
elif page == "Color Space Conversion 🌈🔄":
    st.header("🌈🔄 Color Space Conversion Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        color_space = st.radio("🔴🟢🔵 **Color Space**", ["RGB", "HSV", "LAB", "HLS", "YCbCr"])
        img_array = np.array(image)
        if color_space == "HSV":
            converted_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        elif color_space == "LAB":
            converted_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        elif color_space == "YCbCr":
            converted_img = image.convert("YCbCr")
        elif color_space == "HLS":
            converted_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HLS)
            converted_img = Image.fromarray(converted_image)
        else:
            converted_img = img_array
        if st.button("📟 Convert Color Space"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("🌟 Converted Image")
            st.image(converted_img, caption=f'Image in {color_space} Color Space', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 9
elif page == "Image Brightness/Contrast Adjustment ☀️🌑":
    st.header("☀️🌑 Image Brightness/Contrast Adjustment Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        brightness = st.slider("Adjust Brightness", -100, 100, 0)
        contrast = st.slider("Adjust Contrast", -100, 100, 0)
        if st.button("☀️ Adjust Brightness/Contrast"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("📷 Adjusted Image")
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            adjusted = cv2.convertScaleAbs(opencv_image, alpha=1 + contrast / 100, beta=brightness)
            st.image(adjusted, caption='Brightness/Contrast Adjusted Image', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 10
elif page == "Image Blurring 🌫️🔄":
    st.header("🌫️🔄 Image Blurring Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        blur_type = st.radio("🌫️ **Choose Blurring Effect**", ["Gaussian Blur", "Median Blur", "Bilateral Filter"])
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if blur_type == "Gaussian Blur":
            ksize = st.slider("Kernel Size", min_value=1, max_value=50, value=5, step=2)
            blurred_image = cv2.GaussianBlur(opencv_image, (ksize, ksize), 0)
        elif blur_type == "Median Blur":
            ksize = st.slider("Kernel Size", min_value=1, max_value=50, value=5, step=2)
            if ksize % 2 == 0:
                ksize += 1  # Kernel size must be odd for median blur
            blurred_image = cv2.medianBlur(opencv_image, ksize)
        elif blur_type == "Bilateral Filter":
            d = st.slider("Diameter", min_value=1, max_value=50, value=9)
            sigmaColor = st.slider("Sigma Color", min_value=1, max_value=100, value=75)
            sigmaSpace = st.slider("Sigma Space", min_value=1, max_value=100, value=75)
            blurred_image = cv2.bilateralFilter(opencv_image, d, sigmaColor, sigmaSpace)
        if st.button("📟 Blur Image"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("👓 Blurred Image")
            blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
            blurred_image = Image.fromarray(blurred_image)
            st.image(blurred_image, caption=f'{blur_type} Applied', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 11
elif page == "Histogram Equalization 📊✨":
    st.header("📊✨ Histogram Equalization Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        if st.button("📊 Perform Histogram Equalization"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("✨ Histogram Equalized Image")
            img_array = np.array(image)
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            equalized_img = cv2.equalizeHist(gray_img)
            st.image(equalized_img, caption='Histogram Equalized Image', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 12
elif page == "Face Detection 😊🔍":
    st.header("😊🔍 Face Detection Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        if st.button("😊 Detect Faces"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("🔍 Detected Faces")
            faces = detect_faces(image)
            if len(faces) == 0:
                st.error("⚠️ No faces detected in the image. Please try another image.")
            else:
                st.success(f"😊 Detected {len(faces)} face(s).")
                # Draw rectangles around the faces
                draw_image = np.array(image.copy())
                for (x, y, w, h) in faces:
                    cv2.rectangle(draw_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                st.image(draw_image, caption='Detected Faces', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")

# Page 13
elif page == "Image Segmentation 🧩📦":
    st.header("🧩📦 Image Segmentation Feature")
    if "image" in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        if st.button("🧩 Segment Image"):
            st.subheader("🖼️ Original Image") 
            st.image(image, caption='Original Image', use_column_width=True)
            st.subheader("📦 Segmented Image")
            opencv_image = cv2.cvtColor(np.array(st.session_state.image), cv2.COLOR_RGB2BGR)
            mask = np.zeros(opencv_image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64) 
            rect = (50, 50, opencv_image.shape[1] - 50, opencv_image.shape[0] - 50)
            cv2.grabCut(opencv_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            segmented_image = opencv_image * mask2[:, :, np.newaxis]
            st.image(segmented_image, caption='Segmented Image', use_column_width=True)
    else:
        st.info("⚠️ Please upload or capture an image, or use an example image.")
