# Cattle Disease Detection Using Deep Learning & AI Advisory System

## Project Description

This project implements a deep learning–based web application for detecting Lumpy Skin Disease (LSD) in cattle using skin images. The system leverages transfer learning with MobileNetV2, a Convolutional Neural Network (CNN), to classify images as Healthy or Lumpy Skin Disease.

Once a disease is detected, the system is designed to integrate a Generative AI (Gemini API) to provide preventive measures, treatment options, veterinary advice, and farm management recommendations. The AI advisory module is currently under testing due to API model availability and compatibility constraints, and will be enhanced in future iterations.

The solution is deployed as a Flask-based web application with a modern, user-friendly HTML/CSS interface that allows users to upload images and receive real-time predictions.

## Key Features

- Image-based disease detection using CNNs
- Transfer learning with MobileNetV2 (ImageNet weights)
- Binary classification: Healthy vs Lumpy Skin Disease
- Class imbalance handling using class weighting
- Data augmentation for improved generalization
- Performance evaluation with accuracy, precision, recall, F1-score
- Flask web deployment (MVP)
- AI-powered veterinary recommendations (Gemini API – under testing)

## Dataset Description

- 324 Lumpy Skin Disease images
- 700 Healthy cattle skin images
- Binary classification problem
- 80/20 Train–Test split

### Dataset Structure (After Processing)

```
cattle_lsd_dataset/
│
├── train/
│   ├── Healthy/
│   └── LSD/
│
└── test/
    ├── Healthy/
    └── LSD/
```

## Data Engineering & Visualization

- Manual dataset splitting using Python
- Visualization of class distribution using bar charts
- Image normalization and augmentation:
  - Rotation
  - Zoom
  - Horizontal flipping
- Class imbalance addressed using class weights during training
- Confusion matrix visualization
- Classification report (Precision, Recall, F1-score)

## Model Architecture

The model uses MobileNetV2, a pretrained CNN, as a feature extractor.

### Architecture Overview:

- MobileNetV2 (pretrained on ImageNet, frozen)
- Global Average Pooling
- Dense layer (128 units, ReLU)
- Dropout (0.5)
- Output layer (1 unit, Sigmoid)

### Training Configuration:

- Optimizer: Adam
- Learning Rate: 0.0001
- Loss Function: Binary Crossentropy
- Metrics: Accuracy

## Model Performance Evaluation

- Accuracy evaluated on unseen test data
- Precision, Recall, and F1-score calculated
- Confusion matrix plotted to analyze misclassifications

These metrics demonstrate the model's ability to distinguish between healthy cattle and those affected by Lumpy Skin Disease.

## Web Deployment (MVP)

The model is deployed using Flask with a responsive web interface built using HTML, CSS, and JavaScript.

### Web App Features:

- Image upload with preview
- Real-time prediction
- Confidence score display
- Disease detection alert
- AI-generated veterinary recommendations (when available)
- Clean and intuitive UI

## Gemini AI Advisory System (Status)

The system integrates Google Gemini AI to generate:

- Prevention methods
- Treatment options
- Veterinary recommendations
- Farm management advice

**Current Status:**
The Gemini API integration is implemented at code level but is currently under testing due to API model availability and compatibility issues. Once stabilized, it will dynamically generate disease-specific recommendations based on the detected condition.

## Development Environment Setup

### Requirements

- Python 3.9+
- TensorFlow
- Flask
- NumPy
- Pillow
- Google Generative AI SDK

### Setup Steps

```bash
git clone https://github.com/your-username/cattle-disease-detection.git
cd cattle-disease-detection
pip install -r requirements.txt
```

Place the trained model file (`lsd_model.h5`) in the project root directory.

Run the application:

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

## Code Files

- `model_notebook.ipynb` – Data preprocessing, training, evaluation
- `app.py` – Flask backend and AI inference logic
- `templates/index.html` – Frontend user interface
- `lsd_model.h5` – Trained CNN model
- `README.md` – Project documentation

## Deployment Plan (Future Enhancements)

- Extend model to support multiple cattle diseases using multi-class classification
- Upgrade output layer to Softmax for multi-disease detection
- Fully stabilize Gemini AI integration for dynamic disease-specific advice
- Improve UI/UX with additional explanations
- Deploy application to cloud platforms 

## GitHub Repository

**Repository Link:** https://github.com/NickMuhigi/ML-Capstone.git

## Video Demo

**Link:** https://drive.google.com/file/d/1jwSzc5vbvKDeHZQdXKkY-CeTgxDGCg0Q/view?usp=sharing

## Design Screenshots
<img width="1918" height="923" alt="1" src="https://github.com/user-attachments/assets/0ff1f3e4-e2fd-4a94-bc60-4847f8339657" />
<img width="1918" height="922" alt="2" src="https://github.com/user-attachments/assets/af0c30ba-6429-428e-bedf-041418368af3" />
<img width="1896" height="922" alt="3" src="https://github.com/user-attachments/assets/f3291ff9-1fbe-4bed-8f70-59377882c24b" />


## Contributors

n.muhigi@alustudent.com
