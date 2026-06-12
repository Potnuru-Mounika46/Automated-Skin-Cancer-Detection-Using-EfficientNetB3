from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os

# -----------------------------
# Flask app setup
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Folder setup
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = load_model("skin_cancer_prediction_model.h5")


# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        confidence=None
    )


# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return render_template(
                "index.html",
                prediction="No image uploaded",
                confidence=None
            )

        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "index.html",
                prediction="Please select an image",
                confidence=None
            )

        # Save image
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Load image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)

        # ---------- IMAGE VALIDATION ----------
        mean_pixel = np.mean(img_array)
        std_pixel = np.std(img_array)

        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])

        if (
            mean_pixel > 220
            or mean_pixel < 25
            or std_pixel < 30
            or abs(r_mean - g_mean) > 80
            or abs(g_mean - b_mean) > 80
        ):
            return render_template(
                "index.html",
                prediction="INVALID IMAGE / NOT A SKIN LESION",
                confidence=None
            )

        # ---------- MODEL PREDICTION ----------
        img_input = np.expand_dims(img_array, axis=0)
        img_input = preprocess_input(img_input)

        prediction = model.predict(img_input)
        probability = float(prediction[0][0])

        if probability >= 0.5:
            result = "MALIGNANT"
            confidence_score = probability * 100
        else:
            result = "BENIGN"
            confidence_score = (1 - probability) * 100

        return render_template(
            "index.html",
            prediction=result,
            confidence=round(confidence_score, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            confidence=None
        )


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)