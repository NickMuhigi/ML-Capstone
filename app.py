from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import google.generativeai as genai
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

app = Flask(__name__)

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyBsDrmvDPcJrYVUUwbCB6Kc1Esw6650ZxI"
genai.configure(api_key=GEMINI_API_KEY)

# Use a valid model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# CNN Model Setup

base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Load trained weights
model.load_weights("lsd_model.h5")

# Routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # Convert image to base64 to keep it on screen
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_src = f"data:image/png;base64,{img_data}"

    # Preprocess image for CNN
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # CNN prediction
    prediction = model.predict(img_array)[0][0]
    advice = None

    if prediction > 0.5:
        result = "⚠️ Lumpy Skin Disease Detected"
        confidence = float(prediction)

        # Gemini AI Advisory
        try:
            prompt = """
            Lumpy Skin Disease has been detected.

            Provide:
            - Prevention methods
            - Treatment options
            - Veterinary recommendations
            - Farm management advice

            Format in clear bullet points, concise and practical.
            """
            response = gemini_model.generate_content(prompt)

            if response and hasattr(response, "candidates") and response.candidates:
                advice = response.candidates[0].content.parts[0].text
            else:
                advice = "No advisory content returned from Gemini."

        except Exception as e:
            advice = f"Gemini API Error: {str(e)}"

    else:
        result = "Your Cow Is Healthy"
        confidence = float(1 - prediction)

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=f"{confidence:.2f}",
        advice=advice,
        img_src=img_src
    )

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
