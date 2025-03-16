# FRAME BY FRAME FACE RECOGNITION FROM VIDEO FILE

import cv2
import numpy as np
import json
import mysql.connector
import faiss
from insightface.app import FaceAnalysis

# MySQL Connection Setup
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='project@123@',
    database='TEST'
)
cursor = conn.cursor()

# Load InsightFace Model
face_app = FaceAnalysis(name='buffalo_l')  # Uses IR-SE50
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# Load Face Database and Build FAISS Index
def load_face_database():
    cursor.execute('SELECT id, name, location, feature_vector FROM IMAGES')
    records = cursor.fetchall()

    if not records:
        return None, []

    # Convert JSON string to NumPy array
    feature_vectors = [json.loads(record[3]) for record in records]  
    feature_vectors = np.array(feature_vectors).astype('float32')

    # Normalize embeddings to unit length (for cosine similarity)
    faiss.normalize_L2(feature_vectors)

    # Use FAISS index with inner product (cosine similarity)
    index = faiss.IndexFlatIP(feature_vectors.shape[1])  
    index.add(feature_vectors)

    return index, records

# Function to recognize faces in a video file
def recognize_from_video(video_path):
    index, records = load_face_database()
    if index is None:
        print("No faces in database")
        return

    cap = cv2.VideoCapture(video_path)  # Load the video file
    threshold = 0.4  # Cosine similarity threshold (adjust as needed)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        # Convert frame to RGB for InsightFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = face_app.get(frame_rgb)

        for face in faces:
            feature_vector = face.embedding.reshape(1, -1).astype('float32')

            # Normalize extracted feature vector
            faiss.normalize_L2(feature_vector)

            # Search in FAISS for the closest match
            similarity, index_match = index.search(feature_vector, 1)  
            best_similarity = similarity[0][0]  # Cosine similarity score
            print(f"Similarity: {best_similarity:.2f}")

            x1, y1, x2, y2 = map(int, face.bbox)

            # Determine if match is valid
            if best_similarity < threshold:  # Lower similarity means unknown
                matched_name = "Unknown"
                matched_location = "Unknown"
                print("Recognized: Unknown")
            else:
                matched_id, matched_name, matched_location = records[index_match[0][0]][:3]
                print(f"Recognized: {matched_name} from {matched_location} (Similarity: {best_similarity:.2f})")

            # Draw Bounding Box and Display Closest Match Name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Prepare text to display (Name + Location + Similarity Score)
            display_text = f"{matched_name} - {matched_location} ({best_similarity:.2f})"

            # Display Name, Location, and Similarity Score
            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 20)

        # Resize frame for better display
        display_width = 800  # Set a fixed width
        aspect_ratio = display_width / frame.shape[1]
        display_height = int(frame.shape[0] * aspect_ratio)
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Show the output frame
        cv2.imshow("Face Recognition from Video", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Start face recognition on the given video file
video_path = r"C:\Users\Mit\Desktop\saraswati hackathon\backend\mit2.mp4"  # Replace with the actual video file path
recognize_from_video(video_path)

# Close Database Connection
cursor.close()
conn.close()


# SECOND BY SECOND FACE

# import cv2
# import numpy as np
# import json
# import mysql.connector
# import faiss
# from insightface.app import FaceAnalysis

# # MySQL Connection Setup
# conn = mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='project@123@',
#     database='TEST'
# )
# cursor = conn.cursor()

# # Load InsightFace Model
# face_app = FaceAnalysis(name='buffalo_l')  # Uses IR-SE50
# face_app.prepare(ctx_id=-1, det_size=(640, 640))

# # Load Face Database and Build FAISS Index
# def load_face_database():
#     cursor.execute('SELECT id, name, location, feature_vector FROM IMAGES')
#     records = cursor.fetchall()

#     if not records:
#         return None, []

#     # Convert JSON string to NumPy array
#     feature_vectors = [json.loads(record[3]) for record in records]  
#     feature_vectors = np.array(feature_vectors).astype('float32')

#     # Normalize embeddings to unit length (for cosine similarity)
#     faiss.normalize_L2(feature_vectors)

#     # Use FAISS index with inner product (cosine similarity)
#     index = faiss.IndexFlatIP(feature_vectors.shape[1])  
#     index.add(feature_vectors)

#     return index, records

# # Function to recognize faces every second in a video file
# def recognize_from_video(video_path):
#     index, records = load_face_database()
#     if index is None:
#         print("No faces in database")
#         return

#     cap = cv2.VideoCapture(video_path)  # Load the video file
#     fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
#     duration = frame_count // fps  # Total video duration in seconds

#     threshold = 0.2  # Cosine similarity threshold (adjust as needed)

#     for sec in range(duration):
#         target_frame = sec * fps  # Select frame corresponding to each second
#         cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)  # Jump to the specific frame

#         ret, frame = cap.read()
#         if not ret:
#             break  # Stop if video ends

#         # Convert frame to RGB for InsightFace
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Detect faces in the frame
#         faces = face_app.get(frame_rgb)

#         for face in faces:
#             feature_vector = face.embedding.reshape(1, -1).astype('float32')

#             # Normalize extracted feature vector
#             faiss.normalize_L2(feature_vector)

#             # Search in FAISS for the closest match
#             similarity, index_match = index.search(feature_vector, 1)  
#             best_similarity = similarity[0][0]  # Cosine similarity score
#             print(f"Time: {sec}s - Similarity: {best_similarity:.2f}")

#             x1, y1, x2, y2 = map(int, face.bbox)

#             # Determine if match is valid
#             if best_similarity < threshold:  # Lower similarity means unknown
#                 matched_name = "Unknown"
#                 matched_location = "Unknown"
#                 print(f"Time: {sec}s - Recognized: Unknown")
#             else:
#                 matched_id, matched_name, matched_location = records[index_match[0][0]][:3]
#                 print(f"Time: {sec}s - Recognized: {matched_name} from {matched_location} (Similarity: {best_similarity:.2f})")

#             # Draw Bounding Box and Display Closest Match Name
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             # Prepare text to display (Name + Location + Similarity Score)
#             display_text = f"{matched_name} - {matched_location} ({best_similarity:.2f})"

#             # Display Name, Location, and Similarity Score
#             cv2.putText(frame, display_text, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Resize frame for better display
#         display_width = 800  # Set a fixed width
#         aspect_ratio = display_width / frame.shape[1]
#         display_height = int(frame.shape[0] * aspect_ratio)
#         resized_frame = cv2.resize(frame, (display_width, display_height))

#         # Show the output frame
#         cv2.imshow("Face Recognition from Video", resized_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Start face recognition on the given video file
# video_path = r"C:\Users\Mit\Desktop\saraswati hackathon\backend\mit2.mp4"  # Replace with the actual video file path
# recognize_from_video(video_path)

# # Close Database Connection
# cursor.close()
# conn.close()