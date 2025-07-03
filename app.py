import os
import time
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import hashlib

app = Flask(__name__)
CORS(app)  

# Load the pre-trained face detection model
# Here we're using OpenCV's pre-trained Haar cascade for simplicity
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Uncomment these lines if you want to use a TensorFlow CNN model instead
# MODEL_PATH = 'path_to_your_model/face_detection_model.h5'
# face_detection_model = load_model(MODEL_PATH)

# Dataset configuration
DATASET_PATH = r'D:\face\dataset_celeb_faces'  # Path to your dataset of known images
# Create dataset directory if it doesn't exist
os.makedirs(DATASET_PATH, exist_ok=True)

# Initialize dataset index
image_hashes = set()

def load_dataset_index():
    """Load hashes of all images in the dataset for quick comparison"""
    global image_hashes
    if os.path.exists(os.path.join(DATASET_PATH, 'index.txt')):
        with open(os.path.join(DATASET_PATH, 'index.txt'), 'r') as f:
            image_hashes = set(line.strip() for line in f.readlines())
    else:
        # If no index exists, create one by scanning the dataset directory
        for filename in os.listdir(DATASET_PATH):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(DATASET_PATH, filename)
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    image_hashes.add(file_hash)
        
        # Save the index for future use
        with open(os.path.join(DATASET_PATH, 'index.txt'), 'w') as f:
            for hash_value in image_hashes:
                f.write(f"{hash_value}\n")

def is_image_in_dataset(image_data):
    """Check if the image exists in our dataset using MD5 hash"""
    # Remove header of base64 string if it exists
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64 string
    image_bytes = base64.b64decode(image_data)
    
    # Calculate hash of the image
    image_hash = hashlib.md5(image_bytes).hexdigest()
    
    # Check if hash exists in our dataset index
    return image_hash in image_hashes

def add_image_to_dataset(image_data, filename=None):
    """Add a new image to the dataset"""
    # Remove header of base64 string if it exists
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64 string
    image_bytes = base64.b64decode(image_data)
    
    # Calculate hash
    image_hash = hashlib.md5(image_bytes).hexdigest()
    
    # Generate filename if not provided
    if not filename:
        filename = f"{image_hash}.jpg"
    
    # Save the image to dataset
    file_path = os.path.join(DATASET_PATH, filename)
    with open(file_path, 'wb') as f:
        f.write(image_bytes)
    
    # Add hash to our index
    image_hashes.add(image_hash)
    
    # Update index file
    with open(os.path.join(DATASET_PATH, 'index.txt'), 'a') as f:
        f.write(f"{image_hash}\n")
    
    return image_hash

def preprocess_image_for_cv2(image_data):
    """Convert base64 image to format suitable for OpenCV processing"""
    # Remove header of base64 string
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(image_data)
    
    # Convert to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode to image
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def preprocess_image_for_cnn(image_data):
    """Convert base64 image to format suitable for CNN model"""
    # Remove header of base64 string if it exists
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64 string
    image_bytes = base64.b64decode(image_data)
    
    # Open as PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize to expected input dimensions for the model
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, image.size

def detect_faces_with_opencv(image):
    """Detect faces using OpenCV's Haar Cascade but return only one face"""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Format results - only return the first face (highest priority)
    results = []
    if len(faces) > 0:
        # Sort faces by area (width * height) to get the largest face
        sorted_faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)
        x, y, w, h = sorted_faces[0]  # Take only the largest face
        
        confidence = 0.95  # Placeholder confidence value
        results.append({
            'box': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'confidence': float(confidence)
        })
    
    return results

def detect_faces_with_cnn(img_array, original_size):
    """Detect faces using CNN model (configured to detect only one face)"""
    # NOTE: This is a placeholder. In a real application, you would:
    # 1. Run inference with your model
    # 2. Process predictions to get bounding boxes and confidence scores
    # 3. Filter to keep only the most confident face
    
    # This is dummy data for illustration - only returning one face
    width, height = original_size
    results = [
        {
            'box': {
                'x': int(width * 0.3),
                'y': int(height * 0.2),
                'width': int(width * 0.4),
                'height': int(height * 0.5)
            },
            'confidence': 0.95
        }
    ]
    
    return results

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    """API endpoint for face detection"""
    try:
        # Start timing
        start_time = time.time()
        
        # Get image data from request
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Check if image exists in our dataset
        image_in_dataset = is_image_in_dataset(image_data)
        
        # If the image is not in our dataset, we can either:
        # 1. Reject the request
        # 2. Add it to the dataset
        # 3. Process it without adding to dataset
        # Based on your requirements:
        
        if not image_in_dataset:
            # Option 1: Reject the request
            if data.get('require_dataset_match', False):
                return jsonify({
                    'error': 'Image not found in dataset',
                    'in_dataset': False
                }), 400
            
            # Option 2: Add to dataset if requested
            if data.get('add_to_dataset', False):
                image_hash = add_image_to_dataset(image_data)
                image_in_dataset = True
        
        # Uncomment to use CNN model
        # img_array, original_size = preprocess_image_for_cnn(image_data)
        # faces = detect_faces_with_cnn(img_array, original_size)
        
        # Using OpenCV for now (modified to return only one face)
        cv_image = preprocess_image_for_cv2(image_data)
        faces = detect_faces_with_opencv(cv_image)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
        
        # Return results
        return jsonify({
            'faces': faces,
            'processingTime': processing_time,
            'count': len(faces),
            'in_dataset': image_in_dataset
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/add', methods=['POST'])
def add_to_dataset():
    """API endpoint to add an image to the dataset"""
    try:
        # Get image data from request
        data = request.json
        image_data = data.get('image')
        filename = data.get('filename')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Add image to dataset
        image_hash = add_image_to_dataset(image_data, filename)
        
        return jsonify({
            'success': True,
            'hash': image_hash
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/check', methods=['POST'])
def check_in_dataset():
    """API endpoint to check if an image exists in the dataset"""
    try:
        # Get image data from request
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Check if image exists in dataset
        in_dataset = is_image_in_dataset(image_data)
        
        return jsonify({
            'in_dataset': in_dataset
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize dataset when app starts (using proper Flask 2.0+ approach)
# This function runs before the first request
with app.app_context():
    load_dataset_index()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)