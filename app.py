import streamlit as st
import cv2
import numpy as np
import face_recognition
from PIL import Image
import tempfile

# ---------------------------
# Helper: Load known faces
# ---------------------------
@st.cache_data
def load_known_faces():
    """
    Loads and encodes three known faces from image1.jpg, image2.jpg, image3.jpg.
    Returns a tuple of (known_encodings, known_names).
    """
    known_encodings = []
    # Define the names for the three known persons.
    names = ["Kishan", "Fauzi", "Ming Fatt"]
    known_names = []
    for i in range(1, 4):
        image_path = f"image{i}.jpg"
        try:
            # Load the image file and get face encodings (assumes one face per image)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(names[i - 1])
            else:
                st.error(f"No face found in {image_path}.")
        except Exception as e:
            st.error(f"Error loading {image_path}: {e}")
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Face Recognition App")
st.write("""
Upload an image or video that contains a person.  
The app will detect the face, draw a bounding box with the name placed just outside the bottom-right of the box and also display the recognized person’s image.
""")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

# ---------------------------
# Process Uploaded File
# ---------------------------
if uploaded_file is not None:
    file_type = uploaded_file.type
    st.write(f"**File type:** {file_type}")
    
    # ----- Process Image Files -----
    if file_type.startswith("image"):
        # Open the image with PIL and convert to a NumPy array
        image_pil = Image.open(uploaded_file)
        image = np.array(image_pil)
        
        # If the image has an alpha channel, convert from RGBA to RGB
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Create a copy for annotation
        annotated_image = image.copy()
        
        # Detect faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        recognized_faces = []  # To collect recognized persons (name and corresponding known image)
        
        if not face_encodings:
            st.write("No faces detected in the image.")
        else:
            # Iterate over each detected face and its location
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare the detected face with the known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    # Also add the known person's image to the recognized_faces list.
                    known_image = Image.open(f"image{best_match_index+1}.jpg")
                    recognized_faces.append((name, known_image))
                else:
                    name = "Unknown"
                
                # Draw a rectangle around the face
                cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Set up font for text
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                thickness = 1
                # Get text size to determine placement
                (text_width, text_height), baseline = cv2.getTextSize(name, font, fontScale, thickness)
                # Place text outside the box: aligned to the bottom-right edge, but just below the rectangle.
                text_x = right - text_width  # right-align the text with the box's right edge
                text_y = bottom + text_height + 5  # 5 pixels below the bounding box
                cv2.putText(annotated_image, name, (text_x, text_y), font, fontScale, (0, 255, 0), thickness)
            
            # Display the annotated image
            st.image(annotated_image, caption="Annotated Image", use_container_width=True)
            
            # Display recognized persons side by side using Streamlit columns (if any recognized)
            if recognized_faces:
                st.write("### Recognized Persons")
                cols = st.columns(len(recognized_faces))
                for col, (name, known_image) in zip(cols, recognized_faces):
                    col.image(known_image, caption=name, width=200)
    
    # ----- Process Video Files -----
    elif file_type.startswith("video"):
        # Write the uploaded video to a temporary file so that OpenCV can read it.
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        ret, frame = cap.read()  # Read the first frame
        if not ret:
            st.write("Could not read the video.")
        else:
            # Convert frame from BGR (OpenCV format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            recognized_faces = []  # To collect recognized persons from the video frame
            
            # Detect faces in the frame
            face_locations = face_recognition.face_locations(frame_rgb)
            face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
            
            if not face_encodings:
                st.write("No faces detected in the video frame.")
            else:
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        known_image = Image.open(f"image{best_match_index+1}.jpg")
                        recognized_faces.append((name, known_image))
                    else:
                        name = "Unknown"
                    
                    # Draw bounding box on the frame
                    cv2.rectangle(frame_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(name, font, fontScale, thickness)
                    text_x = right - text_width
                    text_y = bottom + text_height + 5  # place text 5 pixels below the box
                    cv2.putText(frame_rgb, name, (text_x, text_y), font, fontScale, (0, 255, 0), thickness)
                
                st.image(frame_rgb, caption="Annotated First Frame", use_container_width=True)
                
                # Display recognized persons side by side.
                if recognized_faces:
                    st.write("### Recognized Persons")
                    cols = st.columns(len(recognized_faces))
                    for col, (name, known_image) in zip(cols, recognized_faces):
                        col.image(known_image, caption=name, width=200)
        cap.release()
