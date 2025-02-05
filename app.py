from flask import Flask, request, jsonify
import pickle
import numpy as np

# Carregando o modelo
model = pickle.load(open("modelo.pkl", "rb"))

# Inicializando o Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recebendo os dados JSON
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Fazendo a predição
        prediction = modelo.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
