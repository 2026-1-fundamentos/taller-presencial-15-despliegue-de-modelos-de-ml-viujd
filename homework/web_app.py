from flask import Flask, render_template_string, request

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

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\">
    <title>Predicción de precio de casa</title>
  </head>
  <body>
    <h1>Predicción de precio de casa</h1>
    <form method=\"post\">
      <label>bedrooms <input name=\"bedrooms\" type=\"number\" value=\"3\" /></label><br>
      <label>bathrooms <input name=\"bathrooms\" type=\"number\" step=\"0.25\" value=\"2\" /></label><br>
      <label>sqft_living <input name=\"sqft_living\" type=\"number\" value=\"1800\" /></label><br>
      <label>sqft_lot <input name=\"sqft_lot\" type=\"number\" value=\"7000\" /></label><br>
      <label>floors <input name=\"floors\" type=\"number\" step=\"0.5\" value=\"1\" /></label><br>
      <label>waterfront <input name=\"waterfront\" type=\"number\" value=\"0\" /></label><br>
      <label>condition <input name=\"condition\" type=\"number\" value=\"3\" /></label><br>
      <button type=\"submit\">Predecir</button>
    </form>
    {% if prediction is not none %}
      <h2>Precio estimado: ${{ prediction }}</h2>
    {% endif %}
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        payload = {
            key: request.form.get(key, type=float)
            for key in [
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "sqft_lot",
                "floors",
                "waterfront",
                "condition",
            ]
        }
        prediction = predict_price(app.config["MODEL_BUNDLE"], payload)

    return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
