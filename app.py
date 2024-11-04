import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image


medical_uses = {
    'Aloe Vera': """Used for skin healing, digestion, and moisturizing properties.""",
    'Amla': """Rich in vitamin C, it boosts immunity and supports skin health.""",
    'Amruthaballi': """Known for its immune-boosting properties and detox benefits.""",
    'Arali': """Traditionally used for its anti-inflammatory effects.""",
    'Astma_weed': """Used for respiratory health and as an anti-inflammatory.""",
    'Badipala': """Used in traditional remedies for its digestive benefits.""",
    'Balloon_Vine': """Supports digestive health and is used in traditional remedies.""",
    'Bamboo': """Known for its nutritional benefits and structural uses.""",
    'Beans': """High in protein and fiber, supporting overall health.""",
    'Betel': """Used for digestive issues and oral health benefits.""",
    'Bhrami': """Supports cognitive function and mental clarity.""",
    'Bringaraja': """Traditionally used for hair health and liver support.""",
    'Caricature': """Rich in vitamins, it supports overall health and digestion.""",
    'Castor': """Known for its laxative properties and skin benefits.""",
    'Catharanthus': """Used for its benefits in managing diabetes and blood pressure.""",
    'Chakte': """Has various medicinal properties and is used in traditional remedies.""",
    'Chilly': """Boosts metabolism and has antioxidant properties.""",
    'Citron lime': """Supports digestive health and is rich in vitamin C.""",
    'Coffee': """Known for its stimulating effects and antioxidants.""",
    'Common rue': """Used in traditional medicine for digestive issues.""",
    'Coriender': """Rich in nutrients and has antioxidant properties.""",
    'Curry': """Supports digestion and has anti-inflammatory properties.""",
    'Doddpathre': """Used for its medicinal properties in traditional remedies.""",
    'Drumstick': """Nutrient-dense, supports overall health and wellness.""",
    'Ekka': """Known for its health benefits and traditional uses.""",
    'Eucalyptus': """Supports respiratory health and has antiseptic properties.""",
    'Ganigale': """Used for digestive issues and overall wellness.""",
    'Ganike': """Known for its benefits in traditional medicine.""",
    'Gasagase': """Supports digestive health and is used in various remedies.""",
    'Ginger': """Known for its anti-nausea and digestive benefits.""",
    'Globe Amarnath': """Rich in nutrients, supports overall health and wellness.""",
    'Guava': """Rich in vitamin C, supports immune health and digestion.""",
    'Henna': """Used for skin health and has cooling properties.""",
    'Hibiscus': """Supports heart health and is rich in antioxidants.""",
    'Honge': """Used for its health benefits and traditional remedies.""",
    'Insulin': """Helps manage diabetes and blood sugar levels.""",
    'Jackfruit': """Rich in nutrients and supports digestive health.""",
    'Jasmine': """Used for its calming effects and skin benefits.""",
    'Kambajala': """Traditionally used for its medicinal properties.""",
    'Kasambruga': """Known for its benefits in traditional remedies.""",
    'Kohlrabi': """Rich in nutrients and supports overall health.""",
    'Lantana': """Used for its benefits in traditional medicine.""",
    'Lemon': """Rich in vitamin C, supports immune health and digestion.""",
    'Lemongrass': """Supports digestion and has calming effects.""",
    'Malabar_Nut': """Used for its health benefits in traditional remedies.""",
    'Malabar_Spinach': """Rich in nutrients and supports overall health.""",
    'Mango': """Rich in vitamins, supports immune health and digestion.""",
    'Marigold': """Known for its skin benefits and anti-inflammatory properties.""",
    'Mint': """Supports digestion and has calming effects.""",
    'Neem': """Used for its antiseptic and medicinal properties.""",
    'Nelavembu': """Known for its detoxifying properties and traditional uses.""",
    'Nerale': """Used for its benefits in traditional medicine.""",
    'Nooni': """Known for its health benefits in traditional remedies.""",
    'Onion': """Rich in antioxidants, supports heart health and digestion.""",
    'Padri': """Used for its medicinal properties in traditional remedies.""",
    'Palak': """Rich in iron and vitamins, supports overall health.""",
    'Papaya': """Rich in nutrients, supports digestion and skin health.""",
    'Parijatha': """Known for its calming effects and traditional uses.""",
    'Pea': """Rich in protein and nutrients, supports overall health.""",
    'Pepper': """Supports digestion and has antioxidant properties.""",
    'Pomoegranate': """Rich in antioxidants, supports heart health and digestion.""",
    'Pumpkin': """Rich in vitamins and supports immune health.""",
    'Raddish': """Supports digestion and has antioxidant properties.""",
    'Rose': """Known for its skin benefits and calming properties.""",
    'Sampige': """Used for its fragrance and medicinal properties.""",
    'Sapota': """Rich in nutrients, supports overall health and digestion.""",
    'Seethaashoka': """Used in traditional medicine for various health benefits.""",
    'Sesame': """Rich in nutrients, supports heart health, and is good for skin health.""",
    'Sida': """Used for its digestive benefits and is traditionally used in various remedies.""",
    'Soursop': """Rich in vitamins, it supports overall health and has antioxidant properties.""",
    'Spinach': """Rich in vitamins and minerals, it supports overall health and boosts immunity.""",
    'Tulsi': """Known for its immune-boosting properties and is used in traditional remedies.""",
    'Turmeric': """Anti-Inflammatory: Reduces inflammation, helping with conditions like arthritis.""",
    'Vacha': """Used in traditional remedies for digestive issues and cognitive health.""",
    'Vasaka': """Known for its benefits in respiratory health and is traditionally used in treating coughs.""",
    'Vettiver': """Used for its cooling properties and is beneficial for skin health.""",
    'Unknown':""" Try with a clear image!!"""
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
        medical_use = medical_uses[class_name]

        return jsonify({'class_name': class_name, 'medical_use': medical_use})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'})

# Function to map class indices to plant names
def get_class_name(index):
    class_names = ["Aloe Vera",
    "Amla",
    "Amruthaballi",
    "Arali",
    "Astma_weed",
    "Badipala",
    "Balloon_Vine",
    "Bamboo",
    "Beans",
    "Betel",
    "Bhrami",
    "Bringaraja",
    "Caricature",
    "Castor",
    "Catharanthus",
    "Chakte",
    "Chilly",
    "Citron lime",
    "Coffee",
    "Common rue",
    "Coriender",
    "Curry",
    "Doddpathre",
    "Drumstick",
    "Ekka",
    "Eucalyptus",
    "Ganigale",
    "Ganike",
    "Gasagase",
    "Ginger",
    "Globe Amarnath",
    "Guava",
    "Henna",
    "Hibiscus",
    "Honge",
    "Insulin",
    "Jackfruit",
    "Jasmine",
    "Kambajala",
    "Kasambruga",
    "Kohlrabi",
    "Lantana",
    "Lemon",
    "Lemongrass",
    "Malabar_Nut",
    "Malabar_Spinach",
    "Mango",
    "Marigold",
    "Mint",
    "Neem",
    "Nelavembu",
    "Nerale",
    "Nooni",
    "Onion",
    "Padri",
    "Palak",
    "Papaya",
    "Parijatha",
    "Pea",
    "Pepper",
    "Pomoegranate",
    "Pumpkin",
    "Raddish",
    "Rose",
    "Sampige",
    "Sapota",
    "Seethaashoka",
    "Sesame",
    "Sida",
    "Soursop",
    "Spinach",
    "Tulsi",
    "Turmeric",
    "Vacha",
    "Vasaka",
    "Vettiver"]  # Update with your actual class names
    return class_names[index] if index < len(class_names) else "Unknown"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
