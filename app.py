import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", "hasta_mudra_classifier.h5")
LE_PATH = os.environ.get("LE_PATH", "label_encoder.pkl")
IMG_TARGET_SIZE = (48, 48)   # grayscale input size
MAX_CONTENT_LENGTH = 6 * 1024 * 1024  # 6 MB upload limit (adjust as needed)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")

# Load model and label encoder at startup
print("Loading model and label encoder...")
model = load_model(MODEL_PATH)
le = joblib.load(LE_PATH)
print("Model and label encoder loaded.")

def preprocess_image(image_path):
    """Load image in grayscale, resize, normalize and expand dims."""
    img = load_img(image_path, color_mode='grayscale', target_size=IMG_TARGET_SIZE)
    arr = img_to_array(img)               # shape (48,48,1)
    arr = arr.astype('float32') / 255.0   # normalize
    arr = np.expand_dims(arr, axis=0)     # shape (1,48,48,1)
    return arr

def predict_mudra_from_path(image_path):
    arr = preprocess_image(image_path)
    preds = model.predict(arr)
    predicted_index = int(np.argmax(preds, axis=1)[0])
    try:
        predicted_label = le.inverse_transform([predicted_index])[0]
    except Exception:
        # Fallback: if label encoder maps strings to indices in a different way
        predicted_label = str(predicted_index)
    confidence = float(np.max(preds)) * 100.0
    return predicted_label, confidence

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Expect form file field named "file" (from index.html)
    if 'file' not in request.files:
        flash("No file part in the request.")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for('index'))

    # Temporary save to disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        temp_path = tmp.name
        file.save(temp_path)

    try:
        label, confidence = predict_mudra_from_path(temp_path)
        os.remove(temp_path)
        # If request expects JSON, return JSON
        if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
            return jsonify({"label": label, "confidence": round(confidence, 2)})
        # Otherwise render result page
        return render_template("result.html", label=label, confidence=f"{confidence:.2f}%")
    except Exception as e:
        # Clean up and show error
        try:
            os.remove(temp_path)
        except Exception:
            pass
        app.logger.exception("Prediction error")
        flash("Prediction failed: " + str(e))
        return redirect(url_for('index'))

if __name__ == "__main__":
    # For local dev only; Render uses gunicorn in production.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
