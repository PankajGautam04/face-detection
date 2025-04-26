import cv2
import mediapipe as mp
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import uvicorn
import time
import base64
import dlib  # Add DLib for secondary face detection

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://faceswapmagic.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,  # Keep limited to reduce false positives
    refine_landmarks=True,
    min_detection_confidence=0.6,  # Slightly increased to reduce false positives
    min_tracking_confidence=0.6
)

# Load Haar Cascade for secondary face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize DLib face detector
dlib_detector = dlib.get_frontal_face_detector()

def validate_face_with_haar(image, x_min, y_min, x_max, y_max):
    """Validate the face region using Haar Cascade face detector with stricter parameters."""
    face_region = image[y_min:y_max, x_min:x_max]
    if face_region.size == 0:
        return False

    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(50, 50), maxSize=(200, 200))
    return len(faces) > 0

def validate_face_with_dlib(image, x_min, y_min, x_max, y_max):
    """Validate the face region using DLib face detector."""
    face_region = image[y_min:y_max, x_min:x_max]
    if face_region.size == 0:
        return False

    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    faces = dlib_detector(gray, 1)
    return len(faces) > 0

def validate_face_geometry(image, landmarks, x_min, y_min, x_max, y_max):
    """Validate face geometry using landmark distribution with stricter checks."""
    if len(landmarks) < 468:  # Mediapipe Face Mesh should return 468 landmarks
        return False

    # Check key facial landmarks (eyes, nose, mouth)
    left_eye = landmarks[33]   # Left eye outer corner
    right_eye = landmarks[263] # Right eye outer corner
    nose_tip = landmarks[1]    # Nose tip
    mouth_center = landmarks[13]  # Mouth center

    # Check vertical alignment (eyes above nose, nose above mouth)
    if not (left_eye[1] < nose_tip[1] and right_eye[1] < nose_tip[1] and nose_tip[1] < mouth_center[1]):
        return False

    # Check horizontal alignment (eyes should be roughly symmetric)
    eye_distance = abs(left_eye[0] - right_eye[0])
    face_width = x_max - x_min
    if eye_distance < 0.3 * face_width or eye_distance > 0.7 * face_width:
        return False

    # Check face region using contour analysis
    face_region = image[y_min:y_max, x_min:x_max]
    if face_region.size == 0:
        return False

    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h > 0 else 0

    min_area = 2000  # Stricter minimum area
    min_aspect_ratio = 0.7  # Stricter aspect ratio
    max_aspect_ratio = 1.7  # Stricter aspect ratio

    if contour_area < min_area:
        return False
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False

    return True

def validate_face_region(image, x_min, y_min, x_max, y_max):
    """Validate the face region with additional checks."""
    # Check region size
    face_width = x_max - x_min
    face_height = y_max - y_min
    if face_width < 50 or face_height < 50 or face_width > 300 or face_height > 300:
        return False

    # Check region aspect ratio
    aspect_ratio = face_width / face_height if face_height > 0 else 0
    if aspect_ratio < 0.7 or aspect_ratio > 1.7:
        return False

    return True

def extract_faces(image):
    """Extracts all faces from the image with enhanced validation to reduce false positives."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # First pass: Detect faces with Mediapipe
    results = face_mesh.process(img_rgb)
    faces = []
    if not results.multi_face_landmarks:
        return faces

    for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
        # Get bounding box for the face
        landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                     for landmark in face_landmarks.landmark]
        x_coords = [p[0] for p in landmarks]
        y_coords = [p[1] for p in landmarks]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add padding to the bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)

        # Validate the face region
        if not validate_face_region(image, x_min, y_min, x_max, y_max):
            logging.warning(f"Face at index {face_idx} failed region validation")
            continue

        # Validate the face using multiple checks
        # 1. Haar Cascade validation
        if not validate_face_with_haar(image, x_min, y_min, x_max, y_max):
            logging.warning(f"Face at index {face_idx} failed Haar Cascade validation")
            continue

        # 2. DLib validation
        if not validate_face_with_dlib(image, x_min, y_min, x_max, y_max):
            logging.warning(f"Face at index {face_idx} failed DLib validation")
            continue

        # 3. Geometry validation
        if not validate_face_geometry(image, landmarks, x_min, y_min, x_max, y_max):
            logging.warning(f"Face at index {face_idx} failed geometry validation")
            continue

        # Extract the face
        face_img = image[y_min:y_max, x_min:x_max]
        if face_img.size == 0:
            continue

        # Encode the face image as JPEG
        _, buffer = cv2.imencode(".jpg", face_img)
        face_data = buffer.tobytes()
        faces.append({
            "index": face_idx,
            "image": face_data,
            "bounding_box": {"x": x_min / image.shape[1], "y": y_min / image.shape[0], 
                             "width": (x_max - x_min) / image.shape[1], "height": (y_max - y_min) / image.shape[0]}
        })

    # Second pass: Re-detect with stricter parameters to confirm
    if faces:
        stricter_faces = []
        for face in faces:
            # Re-crop the face region and re-detect
            x_min = int(face["bounding_box"]["x"] * image.shape[1])
            y_min = int(face["bounding_box"]["y"] * image.shape[0])
            x_max = x_min + int(face["bounding_box"]["width"] * image.shape[1])
            y_max = y_min + int(face["bounding_box"]["height"] * image.shape[0])
            face_region = image[y_min:y_max, x_min:x_max]
            if face_region.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(face_rgb)
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                stricter_faces.append(face)
            else:
                logging.warning(f"Face at index {face['index']} failed stricter validation")

        faces = stricter_faces

    return faces

@app.post("/detect-faces/")
async def detect_faces(target: UploadFile = File(...)):
    """Detects and extracts all faces from the target image with improved accuracy."""
    try:
        logging.info("Received request to detect faces")
        if not target.content_type.startswith('image/'):
            logging.error("Invalid file type for target image")
            raise HTTPException(status_code=400, detail="Target must be an image file (e.g., JPG, PNG).")
        if target.size > 5_000_000:
            logging.error("Target image too large")
            raise HTTPException(status_code=400, detail="Target image too large (max 5MB).")

        target_data = await target.read()
        target_img = cv2.imdecode(np.frombuffer(target_data, np.uint8), cv2.IMREAD_COLOR)
        if target_img is None:
            logging.error("Invalid target image data")
            raise HTTPException(status_code=400, detail="Invalid target image! Please upload a valid image file.")

        logging.info(f"Target image decoded: {target_img.shape}")

        # Add a slight delay to ensure thorough detection
        time.sleep(0.5)  # 0.5 seconds delay to simulate more careful processing

        faces = extract_faces(target_img)
        if not faces:
            logging.info("No faces detected in target image after validation")
            raise HTTPException(status_code=400, detail="No faces detected in the target image after validation. Please try a different image.")

        faces_response = []
        for face in faces:
            faces_response.append({
                "index": face["index"],
                "image_base64": base64.b64encode(face["image"]).decode('utf-8'),
                "bounding_box": {
                    "x": face["bounding_box"]["x"],
                    "y": face["bounding_box"]["y"],
                    "width": face["bounding_box"]["width"],
                    "height": face["bounding_box"]["height"]
                }
            })
        logging.info(f"Returning {len(faces_response)} validated faces")
        return JSONResponse(content={"faces": faces_response})
    except Exception as e:
        logging.error(f"Error detecting faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to detect faces: {str(e)}. Please try again or contact support.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
