# 📰 Cross-Modality Gated Fusion for Fake News Detection  

## 📌 Overview  
This project implements a **novel multimodal deep learning architecture** that combines **textual and visual data** for detecting fake news.  
The model leverages a **Gated Fusion mechanism** to efficiently integrate features across modalities, making it more robust against noisy and adversarial social media inputs.  

## ✨ Key Features  
- 🔗 **Cross-Modality Gated Fusion** for combining text & image features  
- 🧠 **Multimodal Deep Learning Model** built with PyTorch & PyTorch Lightning  
- 📊 Achieved **97.6% accuracy** on the **Fakeddit benchmark dataset**  
- 🛡️ Improved robustness against **noisy/adversarial inputs**  
- ⚡ Supports **transfer learning** with ResNet & ViT backbones  

## 🏗️ Architecture  
1. **Text Encoder**: Processes textual embeddings from news articles/posts  
2. **Image Encoder**: Uses ResNet-152 or Vision Transformer (ViT) for image features  
3. **Gated Fusion Module**: Learns to dynamically weight contributions from each modality  
4. **Classifier**: Fully connected layers predict fake vs. real news  

## 📂 Project Structure  
