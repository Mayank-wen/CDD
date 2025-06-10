from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import tensorflow as tf

# --- Define your custom ECALayer again ---
class ECALayer(tf.keras.layers.Layer):
    def __init__(self, gamma=2, b=1, **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        channels = input_shape[-1]
        t = int(abs((np.log2(channels) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = tf.keras.layers.Reshape((channels, 1))
        self.conv1d = tf.keras.layers.Conv1D(1, kernel_size=k, padding='same', use_bias=False)
        self.activation = tf.keras.layers.Activation('sigmoid')
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.reshape(x)
        x = self.conv1d(x)
        x = self.activation(x)
        x = tf.reshape(x, [-1, 1, 1, x.shape[1]])
        return self.multiply([inputs, x])

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model with custom_objects
model = load_model('mobilenetv2_eca_model_final.keras', custom_objects={'ECALayer': ECALayer})

# Define all 33 class names (in correct order)
class_names = [
    "Apple Black rot", "Apple Healthy", "Apple Scab",
    "Bell pepper Bacterial spot", "Bell pepper Healthy",
    "Cedar apple rust",
    "Citrus Black spot", "Citrus Healthy", "Citrus canker", "Citrus greening",
    "Corn Common rust", "Corn Gray leaf spot", "Corn Healthy", "Corn Northern Leaf Blight",
    "Grape Black Measles", "Grape Black rot", "Grape Healthy", "Grape Isariopsis Leaf Spot",
    "Peach Bacterial spot", "Peach Healthy",
    "Potato Early blight", "Potato Healthy", "Potato Late blight",
    "Tomato Bacterial spot", "Tomato Early blight", "Tomato Healthy",
    "Tomato Late blight", "Tomato Leaf Mold", "Tomato Mosaic virus",
    "Tomato Septoria leaf spot", "Tomato Spider mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus"
]

# Image preprocessing function
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')  # Ensure 3 channels
    img = img.resize((224, 224))               # Resize to model input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index])

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
