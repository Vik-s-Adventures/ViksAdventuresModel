from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import itertools

app = Flask(__name__)

# Cargar el modelo entrenado
model = tf.keras.models.load_model("model.h5")

# PERMUTACIONES
desempenos = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]
all_permutations = list(itertools.permutations(desempenos))

@app.route('/predict', methods=['POST'])
def predict_learning_path():
    data = request.get_json()

    sexo = 0 if data.get("SEXO") == "F" else 1
    edad = (data.get("EDAD", 13) - 12) / 1.15
    respuestas = [data.get(f"PREG{i+1}", 0) for i in range(10)]
    num_aciertos = sum(respuestas)

    # N_LOGRO
    if num_aciertos >= 8:
        nivel_logro = 3
    elif num_aciertos >= 5:
        nivel_logro = 2
    elif num_aciertos >= 3:
        nivel_logro = 1
    else:
        nivel_logro = 0 
    
    # Agregar PERMUTACION_ID como el último valor en el input
    permutacion_id = data.get("PERMUTACION_ID", 0)  # Default a 0 si no está presente

    # PREDICCION
    input_data = np.array([sexo, edad, nivel_logro] + respuestas + [permutacion_id]).reshape(1, -1)
    prediction = model.predict(input_data)
    
    # PERMUTACION_ID
    pred_perm_id = int(np.argmax(prediction))

    # RUTA
    predicted_permutation = ', '.join(all_permutations[pred_perm_id])

    # RESPONSE
    response = {
        "STUDENT_ID": data.get("STUDENT_ID"),
        "PRED_PERMUTACION_ID": pred_perm_id,
        "PRED_PERMUTACION_RUTA": predicted_permutation
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
