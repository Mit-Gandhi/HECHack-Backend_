import cv2
import numpy as np
import json
import mysql.connector
import faiss
from insightface.app import FaceAnalysis

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

def recognize_live():
    index, records = load_face_database()
    if index is None:
        print("No faces in database")
        return

    cap = cv2.VideoCapture(1) 
    threshold = 0.3 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_app.get(frame_rgb)

        for face in faces:
            feature_vector = face.embedding.reshape(1, -1).astype('float32')

            faiss.normalize_L2(feature_vector)

            similarity, index_match = index.search(feature_vector, 1)  
            best_similarity = similarity[0][0] 
            print(f"Similarity: {best_similarity:.2f}")

            x1, y1, x2, y2 = map(int, face.bbox)

            if best_similarity < threshold: 
                matched_name = "Unknown"
                matched_location = "Unknown"
                print("Recognized: Unknown")
            else:
                matched_id, matched_name, matched_location = records[index_match[0][0]][:3]
                print(f"Recognized: {matched_name} from {matched_location} (Similarity: {best_similarity:.2f})")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Prepare text to display (Name + Location + Similarity Score)
            display_text = f"{matched_name} - {matched_location} ({best_similarity:.2f})"

            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display_width = 800 
        aspect_ratio = display_width / frame.shape[1]
        display_height = int(frame.shape[0] * aspect_ratio)
        resized_frame = cv2.resize(frame, (display_width, display_height))

        cv2.imshow("Live Face Recognition", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()


recognize_live()

cursor.close()
conn.close()