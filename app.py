from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load trained Random Forest
# --------------------------------------------------

MODEL_FILE = "terraguard_model.joblib"

package = joblib.load(MODEL_FILE)

model = package["model"]
features = package["features"]
label_map = package["label_map"]


# --------------------------------------------------
# Same prototype preprocessing used for Darjeeling
# --------------------------------------------------

def to_category(value, low, high):
    if value <= low:
        return 1
    elif value >= high:
        return 3
    return 2


def convert_features(data):

    curvature_cat = to_category(
        data["curvature"],
        -0.001,
        0.001
    )

    slope_cat = to_category(
        data["slope"],
        10,
        30
    )

    aspect_cat = to_category(
        data["aspect"],
        120,
        240
    )

    elevation_cat = to_category(
        data["elevation"],
        1000,
        2000
    )

    ndvi_cat = (
        2 if data["ndvi"] > 0.2
        else 1
    )

    precipitation_cat = to_category(
        data["precipitation"],
        30,
        100
    )

    # Dynamic World:
    # 6 = Built Area
    if int(data["lulc"]) == 6:
        lulc_cat = 2
    else:
        lulc_cat = 1

    return {
        "Curvature": curvature_cat,
        "Slope": slope_cat,
        "Aspect": aspect_cat,
        "Elevation": elevation_cat,
        "NDVI": ndvi_cat,
        "Precipitation": precipitation_cat,
        "LULC": lulc_cat
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------
from flask import Flask, jsonify, request, send_file
@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "Random Forest"
    })


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/api/predict")
def predict():

    try:

        data = request.get_json()

        required = [
            "curvature",
            "slope",
            "aspect",
            "elevation",
            "ndvi",
            "precipitation",
            "lulc"
        ]

        for field in required:
            if field not in data:
                return jsonify({
                    "error": f"Missing field: {field}"
                }), 400

        # Convert raw environmental values
        encoded = convert_features(data)

        # Create model input
        sample = pd.DataFrame([encoded])
        sample = sample[features]

        # Prediction
        prediction = model.predict(sample)[0]

        # Probabilities
        probabilities = model.predict_proba(sample)[0]

        result_probabilities = {}

        for cls, prob in zip(
            model.classes_,
            probabilities
        ):
            label = label_map[int(cls)]

            result_probabilities[label] = round(
                float(prob * 100),
                2
            )

        return jsonify({
            "prediction": label_map[int(prediction)],
            "probabilities": result_probabilities,
            "encoded_features": encoded,
            "model": "Random Forest",
            "prototype": True
        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)