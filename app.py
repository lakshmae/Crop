from flask import Flask, render_template, request, jsonify
import json
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load crop details
with open('crop_data.json', 'r') as f:
    crop_data = json.load(f)

# Load trained model, scaler, and label encoder
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    crops = list(crop_data.keys())
    return render_template('about.html', crops=crops)

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/contact')
def contact():
    experts = [
        {"name": "Dr. Ramesh Kumar", "phone": "+91 9876543210"},
        {"name": "Ms. Priya Sharma", "phone": "+91 8765432109"},
        {"name": "Dr. Sandeep Verma", "phone": "+91 7654321098"},
        {"name": "Mr. Arun Patel", "phone": "+91 6543210987"},
    ]
    return render_template('contact.html', experts=experts)

@app.route('/get_crop_details', methods=['POST'])
def get_crop_details():
    crop_name = request.json.get("crop", "").lower()
    details = crop_data.get(crop_name, "Crop not found")
    return jsonify(details)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        transformed_features = scaler.transform(features)

        prediction_encoded = model.predict(transformed_features)
        predicted_crop = label_encoder.inverse_transform(prediction_encoded)[0]

        image_path = f"static/images/{predicted_crop.lower()}.jpg"
        if not os.path.exists(image_path):
            image_path = "static/images/default.jpg"

        crop_info = crop_data.get(predicted_crop.lower(), {})

        return jsonify({
            'crop': predicted_crop,
            'image': image_path,
            'growth_stages': crop_info.get('growthStages', []),
            'fertilizer': crop_info.get('fertilizer', "Not Available"),
            'soil_type': crop_info.get('idealConditions', {}).get('soilType', "Unknown")
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
