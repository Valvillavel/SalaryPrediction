"""Script para probar el endpoint de predicción."""
import urllib.request
import json

# Payload con un registro de ejemplo
payload = {
    "data": [
        {
            "age": 39,
            "workclass": " State-gov",
            "fnlwgt": 77516,
            "education": " Bachelors",
            "education-num": 13,
            "marital-status": " Never-married",
            "occupation": " Adm-clerical",
            "relationship": " Not-in-family",
            "race": " White",
            "sex": " Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": " United-States"
        }
    ]
}

# Hacer la petición POST
url = "http://127.0.0.1:8000/predict"
headers = {"Content-Type": "application/json"}
data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers=headers, method='POST')

try:
    response = urllib.request.urlopen(req)
    result = json.loads(response.read().decode())
    print("✅ Respuesta del API:")
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"❌ Error: {e}")
