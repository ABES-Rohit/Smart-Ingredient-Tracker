from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pytesseract
import re
import pandas as pd

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ingredient data from CSV
def load_ingredient_data():
    try:
        df = pd.read_csv("ingredient_color.csv")  # Ensure this file exists with 'ingredient', 'category', 'effect'
        ingredient_dict = {}

        for _, row in df.iterrows():
            name = row["ingredient"].strip().lower()
            category = row["category"].strip().title()
            effect = row["effect"].strip()
            alternative = row["alternative"].strip() if "alternative" in row and pd.notna(row["alternative"]) else "No Need for Alternative"

            ingredient_dict[name] = {"category": category, "effect": effect,"alternative": alternative}
        
        return ingredient_dict
    except Exception as e:
        print(f"Error loading ingredients CSV: {e}")
        return {}

# Global dictionary for ingredient categorization
INGREDIENT_DATA = load_ingredient_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the image
    try:
        image = cv2.imread(file_path)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise and threshold
        denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform OCR
        text = pytesseract.image_to_string(binary_image).lower()  # Convert text to lowercase
        
        # Extract and categorize ingredients
        ingredients_info = extract_ingredients(text)
        
        return jsonify({"text": text, "ingredients": ingredients_info})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_ingredients(text):
    extracted_info = []
    words = text.lower()

    # Check for multi-word ingredients first
    for phrase in sorted(INGREDIENT_DATA.keys(), key=lambda x: -len(x)):  # Sort by length (longest first)
        if phrase in words:
            extracted_info.append({
                "name": phrase,
                "category": INGREDIENT_DATA[phrase]["category"],
                "effect": INGREDIENT_DATA[phrase]["effect"],
                "alternative": INGREDIENT_DATA[phrase]["alternative"]
            })
            words = words.replace(phrase, '')  # Remove matched phrase to prevent duplicates

    # Check for single-word ingredients
    for word in words.split():
        if word in INGREDIENT_DATA and word not in [item["name"] for item in extracted_info]:
            extracted_info.append({
                "name": word,
                "category": INGREDIENT_DATA[word]["category"],
                "effect": INGREDIENT_DATA[word]["effect"],
                "alternative": INGREDIENT_DATA[word]["alternative"]
            })

    return extracted_info

if __name__ == '__main__':
    app.run(debug=True)
