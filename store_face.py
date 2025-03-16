import os
import json
import mysql.connector
import re

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="project@123@",
    database="TEST"
)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS IMAGES(
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    location VARCHAR(100),
    image_path VARCHAR(255),
    feature_vector JSON
)
''')
conn.commit()

image_folder = r"C:\Users\Mit\Desktop\saraswati hackathon\backend\cropped_faces-h"
json_file = r"C:\Users\Mit\Desktop\saraswati hackathon\backend\features-h.json"
names_file = r"C:\Users\Mit\Desktop\saraswati hackathon\backend\names.txt"
locations_file = r"C:\Users\Mit\Desktop\saraswati hackathon\backend\location.txt"

with open(names_file, "r") as f:
    names = [line.strip() for line in f.readlines()]

with open(locations_file, "r") as f:
    locations = [line.strip() for line in f.readlines()]

with open(json_file, "r") as f:
    feature_vectors = json.load(f)

def natural_sort_key(filename):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

# Get sorted image files
image_files = sorted(os.listdir(image_folder), key=natural_sort_key)

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    
    if idx < len(names) and idx < len(locations):
        name = names[idx]
        location = locations[idx]
        feature_vector = feature_vectors.get(image_file, [])
        
        if not feature_vector:
            print(f"No feature vector found for {image_file}, skipping...")
            continue

        # Convert feature vector to JSON format for MySQL
        feature_vector_json = json.dumps(feature_vector)

        sql = "INSERT INTO IMAGES (name, location, image_path, feature_vector) VALUES (%s, %s, %s, %s)"
        values = (name, location, image_path, feature_vector_json)
        
        cursor.execute(sql, values)
        conn.commit()

        print(f"Uploaded: {image_file} -> Name: {name}, Location: {location}, Features: {len(feature_vector)} values")
    else:
        print(f"Skipping {image_file}, no matching name or location found!")

cursor.close()
conn.close()

print("All data uploaded sequentially into MySQL!")