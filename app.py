from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diamonds_price_bundle.pkl"

app = Flask(__name__)


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró 'diamonds_price_bundle.pkl'. "
            "Ejecuta primero 'exportar_modelo_diamonds.py'."
        )
    return joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("formulario.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        bundle = load_bundle()
        model = bundle["model"]
        encoders = bundle["encoders"]
        feature_columns = bundle["feature_columns"]

        valores = {
            "carat": float(request.form["carat"]),
            "color": encoders["color"].transform([request.form["color"]])[0],
            "clarity": encoders["clarity"].transform([request.form["clarity"]])[0],
            "y": float(request.form["y"]),
            "z": float(request.form["z"]),
        }

        data_df = pd.DataFrame([valores])[feature_columns]
        prediction = model.predict(data_df)
        precio_estimado = round(float(prediction[0]), 2)

        return jsonify({"price": precio_estimado})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)