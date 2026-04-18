from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import traceback

app = Flask(__name__)
CORS(app)   # ✅ frontend connect ke liye important

# ===============================
# 📦 LOAD MODEL FILES
# ===============================
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Error loading model:", e)

# ===============================
# 🏠 HOME ROUTE
# ===============================
@app.route("/")
def home():
    return "✅ Student Performance API Running"

# ===============================
# 🔮 PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # 🔍 Check empty input
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Debug print
        print("\n📥 Input Data:")
        print(df)

        # One-hot encoding
        df = pd.get_dummies(df)

        # Match training features
        df = df.reindex(columns=features, fill_value=0)

        # Debug print
        print("\n🧾 Processed Data:")
        print(df.head())

        # Scaling
        df_scaled = scaler.transform(df)

        # Prediction
        prediction = model.predict(df_scaled)[0]

        result = round(float(prediction), 2)

        return jsonify({
            "status": "success",
            "predicted_score": result
        })

    except Exception as e:
        print("\n❌ ERROR:")
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ===============================
# 🚀 RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(debug=True)