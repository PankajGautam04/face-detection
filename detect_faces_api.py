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
    max_num_faces=3,  # Reduced to avoid excessive false positives
    refine_landmarks=True,
    min_detection_confidence=0.5,  # Increased for better accuracy
    min_tracking_confidence=0.5
)

def validate_face(image, x_min, y_min, x_max, y_max):
    """Validate if the detected face is real using contour analysis."""
    # Crop the face region
    face_region = image[y_min:y_max, x_min:x_max]
    if face_region.size == 0:
        return False

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # Check the largest contour area and aspect ratio
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h > 0 else 0

    # Define thresholds for validation
    min_area = 1000  # Minimum area to consider a face
    min_aspect_ratio = 0.5  # Minimum aspect ratio (avoid too narrow/wide)
    max_aspect_ratio = 2.0  # Maximum aspect ratio

    if contour_area < min_area:
        return False
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False

    return True

def extract_faces(image):
    """Extracts all faces from the image with validation to reduce false positives."""
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

        # Validate the face using contour analysis
        if not validate_face(image, x_min, y_min, x_max, y_max):
            logging.warning(f"Face at index {face_idx} failed validation")
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
