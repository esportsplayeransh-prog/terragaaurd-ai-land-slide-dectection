import joblib
import pandas as pd


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "terraguard_model.joblib"

package = joblib.load(MODEL_FILE)

model = package["model"]
features = package["features"]
label_map = package["label_map"]


# ============================================================
# REAL DARJEELING VALUES WE COLLECTED
# ============================================================

darjeeling = {
    "Curvature": 0.000292,
    "Slope": 8.84,
    "Aspect": 293.55,
    "Elevation": 2102.10,
    "NDVI": 0.0886536,

    # Put the latest live 24h rainfall here
    "Precipitation": 11.0,

    # Dynamic World: 6 = Built Area
    "LULC": 6
}


# ============================================================
# PROTOTYPE CONVERSION
# ============================================================
#
# IMPORTANT:
# These are prototype normalization rules because the
# training CSV contains encoded ordinal categories.
# They are NOT the original scientific thresholds.
#

def to_category(value, low, high):
    if value <= low:
        return 1
    elif value >= high:
        return 3
    else:
        return 2


# Terrain factors
curvature_cat = to_category(
    darjeeling["Curvature"],
    -0.001,
    0.001
)

slope_cat = to_category(
    darjeeling["Slope"],
    10,
    30
)

aspect_cat = to_category(
    darjeeling["Aspect"],
    120,
    240
)

elevation_cat = to_category(
    darjeeling["Elevation"],
    1000,
    2000
)

# NDVI in our dataset only reaches category 1/2.
# Higher NDVI = category 2.
ndvi_cat = 2 if darjeeling["NDVI"] > 0.2 else 1

# Rainfall prototype category
precip_cat = to_category(
    darjeeling["Precipitation"],
    30,
    100
)

# Dynamic World:
# Built area -> moderate prototype LULC category
if darjeeling["LULC"] == 6:
    lulc_cat = 2
else:
    lulc_cat = 1


# ============================================================
# MODEL INPUT
# ============================================================

sample = pd.DataFrame([{
    "Curvature": curvature_cat,
    "Slope": slope_cat,
    "Aspect": aspect_cat,
    "Elevation": elevation_cat,
    "NDVI": ndvi_cat,
    "Precipitation": precip_cat,
    "LULC": lulc_cat
}])

sample = sample[features]


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

prediction_label = label_map[int(prediction)]


# ============================================================
# OUTPUT
# ============================================================

print("\n==========================================")
print("TERRAGUARD AI — DARJEELING ML PREDICTION")
print("==========================================")

print("\nRAW DARJEELING DATA:")
for key, value in darjeeling.items():
    print(f"{key:<15}: {value}")

print("\nMODEL-READY FEATURES:")
print(sample.iloc[0].to_dict())

print("\n==========================================")
print("ML PREDICTION")
print("==========================================")

print("Predicted Susceptibility:", prediction_label)

print("\nProbability:")

for cls, prob in zip(model.classes_, probabilities):
    label = label_map[int(cls)]
    print(f"{label:<10}: {prob * 100:.2f}%")

print("\n==========================================")
print("NOTE")
print("==========================================")
print(
    "Prototype inference only. "
    "Raw Darjeeling values were converted to the "
    "training dataset's encoded feature scale."
)