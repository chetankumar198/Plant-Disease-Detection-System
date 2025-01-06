import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from tensorflow.keras.models import load_model

# Function to load the model and make predictions
def model_predict(image):
    try:
        model = load_model(r"D:\Python\PlantDiseaseDetection\CNN_plantdiseases_model.keras")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    # Convert PIL image to NumPy array
    if isinstance(image, Image.Image):  # Check if input is PIL Image
        img = np.array(image)
    else:
        img = cv2.imread(image)  # If the input is a path, read it using OpenCV
    
    if img is None:
        st.error("Error: Image not found or invalid format.")
        return None

    H, W, C = 224, 224, 3
    try:
        # Resize the image and convert color to RGB
        img = cv2.resize(img, (H, W))
        if len(img.shape) == 3 and img.shape[2] == 3:  # Ensure the image has 3 channels (RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV default) to RGB
        else:
            st.error("Invalid image format: Image must have 3 color channels (RGB).")
            return None
        img = img.astype("float32") / 255.0  # Normalize the image
        img = img.reshape(1, H, W, C)  # Add batch dimension

        prediction = model.predict(img)
        result_index = np.argmax(prediction, axis=-1)[0]
        return result_index
    except Exception as e:
        st.error(f"Error during image processing or prediction: {e}")
        return None

# Apply custom CSS for enhanced styling
st.markdown(
    """
    <style>
    body {
        background-color: #a8d8a8; /* Parrot green */
        color: #000;
    }
    .stSidebar {
        background: linear-gradient(to right, #7cb77c, #a8d8a8);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background: linear-gradient(to bottom, #7cb77c, #a8d8a8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .content-box {
        background: white;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        padding: 20px;
    }
    footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        background-color: #7cb77c;
        color: white;
        padding: 10px;
    }
    .image-shadow {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        border-radius: 10px;
    }
    
    /* Mobile-responsive styles */
    @media (max-width: 600px) {
        .content-box {
            padding: 10px;
        }
        h1, h2 {
            font-size: 18px;
        }
        footer {
            font-size: 12px;
            padding: 5px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar and main interface
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display the image in the sidebar
img = Image.open(r"D:\Python\PlantDiseaseDetection\DALL·E 2025-01-03 09.39.38 - A vibrant and dynamic farm landscape in a 16_9 aspect ratio, with diverse healthy crops under a glowing sunset. Digital overlays display crop health a.webp")
st.sidebar.image(img, use_container_width=True, caption="Healthy Crops")

# Class names
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']

# Main application logic
if app_mode == "HOME":
    st.markdown(
        """
        <div class="content-box">
            <h1 style='text-align: center;'>🌿 Plant Disease Detection System for Sustainable Agriculture</h1>
            <p style='text-align: center;'>Empowering farmers with AI-driven solutions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
elif app_mode == "DISEASE RECOGNITION":
    st.markdown(
        """
        <div class="content-box">
            <h2>Plant Disease Detection</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    option = st.radio("Choose an option:", ["Upload an Image", "Scan via Webcam"])

    
    if option == "Upload an Image":
        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            save_path = os.path.join(os.getcwd(), test_image.name)
            with open(save_path, "wb") as f:
                f.write(test_image.getbuffer())
            if st.button("Show Image"):
                st.image(test_image, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict"):
                result_index = model_predict(save_path)
                if result_index is not None and result_index < len(class_name):
                    st.success(f"Predicted class: {class_name[result_index]}")
                else:
                    st.error("Invalid prediction index or no prediction made.")

    elif option == "Scan via Webcam":
        st.write("Click 'Start Camera' to capture an image using your webcam.")
        camera_image = st.camera_input("Capture Image")
        if camera_image is not None:
            # Convert the captured image to a PIL Image
            img = Image.open(camera_image)
            st.image(img, caption="Captured Image", use_column_width=True)
            if st.button("Predict"):
                result_index = model_predict(img)
                if result_index is not None and result_index < len(class_name):
                    st.success(f"Predicted class: {class_name[result_index]}")
                else:
                    st.error("Invalid prediction index or no prediction made.")

# Footer
st.markdown(
    """
    <footer>
        <p>🌱 Empowering farmers, one crop at a time.</p>
    </footer>
    """,
    unsafe_allow_html=True..
)







