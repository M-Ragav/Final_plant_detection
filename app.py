import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image


medical_uses = {
'Neem':"""
Antimicrobial: Treats infections with antibacterial and antifungal properties.
Anti-Inflammatory: Reduces inflammation, aiding conditions like arthritis.
Blood Sugar Regulation: Helps lower blood sugar levels for diabetes management.
Skin Health: Treats acne and skin conditions due to its soothing effects.
Immune Support: Boosts the immune system and aids detoxification.""",

'Aloe Vera':
"""
Skin Healing: Promotes wound healing and soothes burns and cuts.
Moisturizer: Hydrates skin and treats dry skin conditions.
Anti-Inflammatory: Reduces inflammation and discomfort from conditions like arthritis.
Digestive Aid: Supports digestive health and alleviates constipation.
Immune Booster: Enhances immune function and helps detoxify the body.""",

'Tulsi':
"""
Stress Relief: Reduces stress and anxiety by balancing cortisol levels.
Anti-Inflammatory: Helps alleviate inflammation and pain in the body.
Respiratory Health: Eases respiratory issues like asthma and bronchitis.
Antimicrobial: Fights infections with its antibacterial and antiviral properties.
Blood Sugar Control: Supports healthy blood sugar levels and aids in diabetes management.
""",

'Jasmine':
"""
Aromatherapy: Used for its calming scent to reduce stress and anxiety.
Skin Care: Helps improve skin elasticity and treats minor skin irritations.
Menstrual Relief: Alleviates menstrual cramps and promotes regular cycles.
Antidepressant: Acts as a natural mood enhancer, combating feelings of depression.
Sleep Aid: Promotes restful sleep and helps manage insomnia.""",

'Mango':
"""Rich in Nutrients: Packed with vitamins A, C, and E, supporting overall health and immunity.
Digestive Health: Aids digestion and alleviates constipation due to its fiber content.
Skin Care: Promotes healthy skin and may help treat acne and blemishes.
Eye Health: Supports eye health and vision due to high levels of beta-carotene.
Antioxidant Properties: Contains antioxidants that combat free radicals and reduce inflammation.""",

'Drumstick':
"""Nutrient-Rich: High in vitamins, minerals, and amino acids, supporting overall health.
Anti-Inflammatory: Reduces inflammation and may help with conditions like arthritis.
Blood Sugar Control: Helps lower blood sugar levels, beneficial for diabetes management.
Antioxidant Properties: Contains antioxidants that protect against oxidative stress.
Digestive Health: Supports digestive health and may alleviate constipation.""",

'Gauva':
"""
High in Vitamin C: Boosts the immune system and helps fight infections.
Digestive Aid: Rich in fiber, it promotes healthy digestion and relieves constipation.
Antioxidant Properties: Contains antioxidants that protect against cellular damage.
Skin Health: May improve skin texture and treat acne due to its antibacterial properties.
Blood Sugar Regulation: Supports healthy blood sugar levels, beneficial for diabetes management""",

'Unknown':
""" Try with a clear image!!"""
}


# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
try:
    model = load_model('model/plant_model.h5')  # Replace with your actual model path
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for identifying the plant
@app.route('/identify', methods=['POST'])
def identify_plant():
    data = request.get_json()

    # Extract base64 image string
    image_data = data['image'].split(',')[1]  # Removing the image type info (data:image/png;base64,...)
    img_bytes = base64.b64decode(image_data)

    # Open the image using PIL, then convert to a NumPy array
    img = Image.open(BytesIO(img_bytes))
    img = img.convert("RGB")  # Ensure 3 channels (RGB)

    # Preprocess the image for the model
    img = img.resize((150, 150))  # Resize to the input size expected by the model (update this if your model requires a different size)
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, height, width, channels)

    try:
        # Perform prediction
        prediction = model.predict(img)
        class_index = np.argmax(prediction)

        # Get the predicted plant name
        class_name = get_class_name(class_index)

        # Retrieve medical use from the database
        medical_use = medical_use[class_name]

        return jsonify({'class_name': class_name, 'medical_use': medical_use})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'})

# Function to map class indices to plant names
def get_class_name(index):
    class_names = ['Aloe Vera', 'Neem', 'Tulsi', 'Jasmine', 'Mango', 'Drumstick', 'Guava']  # Update with your actual class names
    return class_names[index] if index < len(class_names) else "Unknown"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
