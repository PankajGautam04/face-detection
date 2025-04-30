import cv2
import mediapipe as mp
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import uvicorn

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
    max_num_faces=5,  # Increased to detect multiple faces
    refine_landmarks=True,
    min_detection_confidence=0.1,  # Lowered to improve detection
    min_tracking_confidence=0.1
)

def extract_faces(image):
    """Extracts all faces from the image and returns them as cropped images with bounding boxes."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            "bounding_box": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
        })
    return faces

@app.post("/detect-faces/")
async def detect_faces(target: UploadFile = File(...)):
    """Detects and extracts all faces from the target image."""
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
            raise HTTPException(status_code=400, detail="Invalid target image!")
        logging.info(f"Target image decoded: {target_img.shape}")

        faces = extract_faces(target_img)
        if not faces:
            logging.info("No faces detected in target image")
            raise HTTPException(status_code=400, detail="No faces detected in the target image.")

        import base64
        faces_response = []
        for face in faces:
            faces_response.append({
                "index": face["index"],
                "image_base64": base64.b64encode(face["image"]).decode('utf-8'),
                "bounding_box": face["bounding_box"]
            })
        logging.info(f"Returning {len(faces_response)} faces")
        return JSONResponse(content={"faces": faces_response})
    except Exception as e:
        logging.error(f"Error detecting faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to detect faces: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
