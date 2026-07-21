import os
from pathlib import Path

from flask import Flask, jsonify, request

try:
    from .train_model import MODEL_PATH, load_model, predict_price, train_and_save_model
except ImportError:  # pragma: no cover - fallback for direct execution
    from train_model import MODEL_PATH, load_model, predict_price, train_and_save_model


def ensure_model_bundle():
    if not MODEL_PATH.exists():
        train_and_save_model()
    return load_model(MODEL_PATH)


app = Flask(__name__)
app.config["MODEL_BUNDLE"] = ensure_model_bundle()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    if not payload:
        payload = request.args.to_dict()

    try:
        price = predict_price(app.config["MODEL_BUNDLE"], payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"price": price})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=False)
