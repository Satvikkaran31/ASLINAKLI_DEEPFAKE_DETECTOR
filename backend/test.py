import os
import logging
import argparse
import torch
import pytorch_lightning as pl
from PIL import Image
from sentence_transformers import SentenceTransformer
from model import MultimodalFakeNewsDetectionModel, MultimodalFakeNewsDetectionModelWithDialogue,JointTextImageModel

def load_model(checkpoint_path, modality):
    if modality == "text-image-dialogue":
        return MultimodalFakeNewsDetectionModelWithDialogue.load_from_checkpoint(checkpoint_path)
    else:
        return MultimodalFakeNewsDetectionModel.load_from_checkpoint(checkpoint_path)

def preprocess_image(image_path, image_transform):
    image = Image.open(image_path).convert("RGB")
    return image_transform(image).unsqueeze(0)  # Add batch dimension

def preprocess_text(text, text_embedder):
    return torch.tensor(text_embedder.encode([text])).view(1, -1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--text", type=str, required=True, help="Text input for evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--modality", type=str, default="text-image", help="Modality type: text-image or text-image-dialogue")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load model
    model = load_model(args.checkpoint, args.modality)
    model.eval()

    # Load preprocessing tools
    text_embedder = SentenceTransformer("all-mpnet-base-v2")
    image_transform = JointTextImageModel.build_image_transform()

    # Preprocess inputs
    text_embedding = preprocess_text(args.text, text_embedder)
    image_tensor = preprocess_image(args.image, image_transform)

    # Run inference
    with torch.no_grad():
        output = model(text_embedding, image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Print result
    result = "Fake News" if prediction == 1 else "Real News"
    print(f"Prediction: {result}")
    logging.info(f"Prediction: {result}")

if __name__ == "__main__":
    main()
