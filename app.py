from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import shap

app = Flask(__name__)

# Charger modèle + scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# SHAP explainer (compatible tree models)
explainer = shap.TreeExplainer(model)

# -------------------------------
# Page principale
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# Prediction
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("data", None)

        # 🔥 Validation solide
        if not data or len(data) != 13:
            return jsonify({"error": "Invalid input length"}), 400

        # convertir en float
        data = [float(x) for x in data]

        # reshape + scaling
        data_np = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data_np)

        # prediction
        pred = int(model.predict(data_scaled)[0])
        proba = float(model.predict_proba(data_scaled)[0][1])

        return jsonify({
            "prediction": pred,
            "risk": round(proba, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# SHAP explanation
# -------------------------------
@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.json.get("data", None)

        if not data or len(data) != 13:
            return jsonify({"error": "Invalid input"}), 400

        data = [float(x) for x in data]
        data_np = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data_np)

        # 🔥 Correction SHAP (important)
        shap_values = explainer.shap_values(data_scaled)

        # gérer multi-class ou binaire
        if isinstance(shap_values, list):
            explanation = shap_values[1][0].tolist()
        else:
            explanation = shap_values[0].tolist()

        return jsonify({
            "shap_values": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)