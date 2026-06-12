from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

app = Flask(__name__)

model = load_model("skin_cancer_model.h5")


# -----------------------------
# 🧠 Skin detection using HSV
# -----------------------------
def is_skin_image(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Skin color range (works reasonably well)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(img_hsv, lower, upper)

    skin_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])

    print("Skin ratio:", skin_ratio)

    # 🔥 Tune this threshold
    if skin_ratio < 0.1:
        return False
    return True


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("index.html", prediction="Invalid image", confidence=None)

        try:
            img = Image.open(file).convert("RGB")
            img = img.resize((224, 224))

            img_array = np.array(img)

            # 🚫 Reject non-skin images
            if not is_skin_image(img_array):
                return render_template(
                    "index.html",
                    prediction="Invalid image / Not a skin lesion",
                    confidence=None
                )

            # -----------------------------
            # Model prediction
            # -----------------------------
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            prediction = model.predict(img_array)
            prob = float(prediction[0][0])

            confidence = max(prob, 1 - prob) * 100

            if prob >= 0.5:
                result = "MALIGNANT"
            else:
                result = "BENIGN"

        except Exception as e:
            print("ERROR:", e)
            result = "Invalid image / Not a skin lesion"
            confidence = None

        return render_template("index.html", prediction=result, confidence=confidence)

    return render_template("index.html", prediction=None, confidence=None)


if __name__ == "__main__":
    app.run(debug=True)