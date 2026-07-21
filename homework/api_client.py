import argparse

import requests


DEFAULT_URL = "http://127.0.0.1:5001/predict"


def build_payload():
    return {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 7000,
        "floors": 1.0,
        "waterfront": 0,
        "condition": 3,
    }


def main():
    parser = argparse.ArgumentParser(description="Consultar el modelo desde una API")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    payload = build_payload()
    response = requests.post(args.url, json=payload, timeout=10)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
