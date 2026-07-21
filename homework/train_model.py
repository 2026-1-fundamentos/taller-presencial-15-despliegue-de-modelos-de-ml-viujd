import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]
DATASET_PATH = Path(__file__).resolve().parent.parent / "files" / "input" / "house_data.csv"
MODEL_PATH = Path(__file__).resolve().parent / "house_predictor.pkl"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["price", *FEATURE_COLUMNS]].dropna().copy()


def train_and_save_model(path: Path = DATASET_PATH, model_path: Path = MODEL_PATH):
    df = load_dataset(path)
    X = df[FEATURE_COLUMNS]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    bundle = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "target": "price",
        "r2_score": float(r2),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    return bundle


def load_model(model_path: Path = MODEL_PATH):
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def predict_price(bundle, payload):
    values = {}
    for feature in bundle["features"]:
        raw_value = payload.get(feature)
        if raw_value is None:
            raise ValueError(f"Falta el valor para '{feature}'.")
        values[feature] = float(raw_value)

    frame = pd.DataFrame([values], columns=bundle["features"])
    prediction = bundle["model"].predict(frame)[0]
    return float(prediction)


if __name__ == "__main__":
    bundle = train_and_save_model()
    print(f"Modelo entrenado y guardado en {MODEL_PATH}")
    print(f"R2 en conjunto de prueba: {bundle['r2_score']:.3f}")
