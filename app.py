from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import json
import mysql.connector
import faiss
import cv2
import os
from insightface.app import FaceAnalysis
from pix2pix_model import Generator  

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained Pix2Pix Generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
model.load_state_dict(torch.load(r"C:\Users\Mit\Desktop\saraswati hackathon\backend\sketch_to_image_07.pth", map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Initialize Face Recognition Model (IR-SE50)
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# MySQL Connection Setup
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='project@123@',
        database='TEST'
    )

# Path to store the generated image
GENERATED_IMAGE_PATH = "last_generated.png"
target_face_feature_vector = None 

# Load Face Database and Build FAISS Index
def load_face_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, location, feature_vector FROM IMAGES')
    records = cursor.fetchall()
    cursor.close()
    conn.close()

    if not records:
        return None, []
    
    feature_vectors = [json.loads(record[3]) for record in records]  
    feature_vectors = np.array(feature_vectors).astype('float32')
    faiss.normalize_L2(feature_vectors)
    index = faiss.IndexFlatIP(feature_vectors.shape[1])  
    index.add(feature_vectors)
    
    return index, records

# Function to Extract Face Features
def extract_face_features(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = face_app.get(image)
    
    if not faces:
        return None

    # Extract the feature vector from the largest face
    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    feature_vector = largest_face.embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(feature_vector)
    
    return feature_vector, largest_face.bbox 

# Function to Find Similar Faces in the Database
def find_similar_face(feature_vector, threshold=0.3):
    index, records = load_face_database()
    
    if index is None:
        return None
    
    distances, indices = index.search(feature_vector, 1) 
    
    if distances[0][0] >= threshold:
        db_index = indices[0][0]
        if 0 <= db_index < len(records):
            person_id, person_name, person_location, _ = records[db_index]
            return {
                "id": person_id,
                "name": person_name,
                "location": person_location,
                "similarity": round(float(distances[0][0]), 2)
            }
    
    return None

@app.post("/generate/")
async def generate_image(file: UploadFile = File(...)):
    global target_face_feature_vector
    try:
        # Read and preprocess the input sketch
        sketch_image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = transform(sketch_image).unsqueeze(0).to(device)

        # Generate the colorized image
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Convert tensor to image
        output_tensor = (output_tensor.squeeze().cpu().detach() + 1) / 2
        generated_image = transforms.ToPILImage()(output_tensor)

        # Save generated image to disk
        generated_image.save(GENERATED_IMAGE_PATH)

        # Extract the feature vector from the generated image
        target_face_feature_vector, _ = extract_face_features(generated_image)

        # Convert to PNG for response
        output_io = io.BytesIO()
        generated_image.save(output_io, format="PNG")
        output_io.seek(0)

        return StreamingResponse(output_io, media_type="image/png")
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get_matched_info/")
def get_matched_info():
    if target_face_feature_vector is None:
        return JSONResponse(content={"error": "No face detected in generated image"}, status_code=404)

    # Find similar face
    matched_person = find_similar_face(target_face_feature_vector)
    if matched_person is None:
        return JSONResponse(content={"error": "No matching person found"}, status_code=404)

    return JSONResponse(content=matched_person, status_code=200)

camera = None 

def generate_frames():
    global camera, target_face_feature_vector
    if camera is None:
        camera = cv2.VideoCapture(0) 
    
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_app.get(rgb_frame)

        for face in faces:
            detected_feature_vector = face.embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(detected_feature_vector)

            if target_face_feature_vector is not None:
                similarity = np.dot(target_face_feature_vector, detected_feature_vector.T)[0, 0]

                if similarity >= 0.3:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) 
                    cv2.putText(frame, "Target Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Stream frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/live_camera")
def live_camera():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("shutdown")
def shutdown_event():
    global camera
    if camera is not None:
        camera.release()
        camera = None