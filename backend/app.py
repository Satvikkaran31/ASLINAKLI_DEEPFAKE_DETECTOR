import os
import logging
from flask import Flask, request, jsonify
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer 
from model import MultimodalFakeNewsDetectionModel, JointTextImageModel
from flask_cors import CORS  # Import CORS at the top

app = Flask(__name__)
CORS(app)
# Load model and preprocessing tools
CHECKPOINT_PATH = "checkpoint/model.ckpt"  
MODALITY = "text-image"

model = MultimodalFakeNewsDetectionModel.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

text_embedder = SentenceTransformer("all-mpnet-base-v2")
image_transform = JointTextImageModel.build_image_transform()

def preprocess_image(image):
    return image_transform(image).unsqueeze(0) 

def preprocess_text():
    return torch.tensor(text_embedder.encode(["sample image"])).view(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Missing image"}), 400

        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")

        text_embedding = preprocess_text()
        image_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(text_embedding, image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()
            deepfake_prob = probabilities[0]  # Assuming index 1 corresponds to deepfake class

        # Assign Labels Based on Probability
        if deepfake_prob >= 0.9:
            result = "Highly Likely Real News"
        elif deepfake_prob >= 0.75:
            result = "Likely Real News"
        elif deepfake_prob >= 0.6:
            result = "Uncertain Mixed"
        elif deepfake_prob >= 0.35:
            result = "Likely Fake News" 
        else:
            result = "Highly Likely Fake News"

        return jsonify({"prediction": result, "confidence": deepfake_prob})

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/')
def home():
    return "Flask app is running!"    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=4000, debug=True)
