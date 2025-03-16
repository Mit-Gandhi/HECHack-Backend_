import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import json
import mysql.connector
import faiss
from insightface.app import FaceAnalysis
import os

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='project@123@',
    database='TEST'
)
cursor = conn.cursor()

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def load_face_database():
    cursor.execute('SELECT id, name, location, feature_vector FROM IMAGES')
    records = cursor.fetchall()
    
    if not records:
        return None, []
    
    feature_vectors = [json.loads(record[3]) for record in records]  
    feature_vectors = np.array(feature_vectors).astype('float32')
    faiss.normalize_L2(feature_vectors)
    index = faiss.IndexFlatIP(feature_vectors.shape[1])  
    index.add(feature_vectors)
    
    return index, records

class DownSample(nn.Module):
    def __init__(self, Input_Channels, Output_Channels):
        super(DownSample, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(Input_Channels, Output_Channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.model(x)

class Upsample(nn.Module):
    def __init__(self, Input_Channels, Output_Channels):
        super(Upsample, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(Input_Channels, Output_Channels, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)
        self.down5 = DownSample(512, 512)
        self.down6 = DownSample(512, 512)
        self.down7 = DownSample(512, 512)
        self.down8 = DownSample(512, 512)
        
        self.up1 = Upsample(512, 512)
        self.up2 = Upsample(1024, 512)
        self.up3 = Upsample(1024, 512)
        self.up4 = Upsample(1024, 512)
        self.up5 = Upsample(1024, 256)
        self.up6 = Upsample(512, 128)
        self.up7 = Upsample(256, 64)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(128, 3, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.final(u7)
        
        return u8


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)


model_path = r"C:\Users\Mit\Desktop\saraswati hackathon\backend\sketch_to_image_07.pth"

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def process_input_image(image_path, skip_generation=False):
    if skip_generation:
        print(f"Using input image directly: {image_path}")
        return image_path
    
    input_image = Image.open(image_path).convert("RGB")  
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_tensor = (output_tensor.squeeze().cpu().detach() + 1) / 2  
    output_image = transforms.ToPILImage()(output_tensor)
    output_path = "generated_output.jpg"
    output_image.save(output_path)
    print("Generated image saved at:", output_path)
    
    return output_path

def extract_face_features(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
        
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb_img)
    
    if not faces:
        print("No face detected with standard settings. Trying enhanced detection...")
        
        resized_img = cv2.resize(img, (640, 640))
        rgb_resized = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb_resized)
        
        if not faces:
            print("Enhanced detection failed. Using input image directly...")

            return None
    
    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])) if faces else None
    
    if largest_face:
        feature = largest_face.embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(feature)
        print(f"Successfully extracted features from face")
        return feature
    
    return None

def direct_database_search(image_path, threshold=0.3):
    print("Attempting direct database search...")
    index, records = load_face_database()
    if index is None:
        print("No faces in database")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb_img)
    
    if not faces:
        print("No faces detected in input image for direct search")
        return
        
    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    feature_vector = largest_face.embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(feature_vector)
    
    # Search database for closest match
    distances, indices = index.search(feature_vector, 1)
    if distances[0][0] >= threshold:
        db_index = indices[0][0]
        if 0 <= db_index < len(records):
            person_id = records[db_index][0]
            person_name = records[db_index][1]
            person_location = records[db_index][2]
            print(f"Found direct match: {person_name} from {person_location} with similarity {distances[0][0]:.2f}")
            return True
            
    print("No direct match found above threshold")
    return False

# Function to Detect Only the Person with Closest Match in Crowd
def detect_in_crowd(ref_feature=None, input_image_path=None, threshold=0.3):

    index, records = load_face_database()
    if index is None:
        print("No faces in database")
        return

    if ref_feature is None and input_image_path is not None:

        ref_feature = extract_face_features(input_image_path)
        
        if ref_feature is None:
            print("Could not extract features from input image.")
            print("Will attempt to match anyone in the database with faces in the camera.")
    
    cap = cv2.VideoCapture(0) 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_app.get(frame_rgb)
        
        best_match = None
        best_similarity = 0
        best_db_record = None
        
        for face in faces:
            feature_vector = face.embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(feature_vector)
            
            if ref_feature is not None:
                ref_similarity = np.dot(ref_feature, feature_vector.T)[0][0]
                similarity = ref_similarity
            else:

                distances, indices = index.search(feature_vector, 1)
                similarity = distances[0][0]
                
            distances, indices = index.search(feature_vector, 1)
            db_index = indices[0][0]
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = face
                
                if 0 <= db_index < len(records):
                    best_db_record = records[db_index]
        
        if best_match and best_db_record is not None:
            x1, y1, x2, y2 = map(int, best_match.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get person info from database record
            person_id = best_db_record[0]
            person_name = best_db_record[1]
            person_location = best_db_record[2]
            
            # Display person info on the bounding box
            info_text = f"{person_name} - {person_location}"
            similarity_text = f"Match: {best_similarity:.2f}"
            
            # Background for text
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1-40), (x1 + text_size[0], y1-10), (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y1-70), (x1 + text_size[0], y1-40), (0, 255, 0), -1)
            
            # Display text
            cv2.putText(frame, info_text, (x1, y1-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, similarity_text, (x1, y1-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.imshow("Face Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            threshold += 0.05
            print(f"Threshold increased to: {threshold:.2f}")
        elif key == ord('-'):
            threshold -= 0.05
            threshold = max(0.0, threshold)
            print(f"Threshold decreased to: {threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()

def process_and_find_in_crowd(input_path, threshold=0.3):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist")
        return
        
    ref_feature = extract_face_features(input_path)
    
    if ref_feature is not None:
        print("Successfully extracted features from input image directly")

        detect_in_crowd(ref_feature, input_path, threshold)
    else:

        print("Trying to generate image from sketch...")
        generated_path = process_input_image(input_path, skip_generation=False)
        ref_feature = extract_face_features(generated_path)
        
        if ref_feature is not None:

            detect_in_crowd(ref_feature, generated_path, threshold)
        else:

            print("Falling back to camera detection with database matching...")
            detect_in_crowd(None, input_path, threshold)

if __name__ == "__main__":
    input_path = r"C:\Users\Mit\Desktop\saraswati hackathon\backend\Images\rohit2.png"  
    process_and_find_in_crowd(input_path)