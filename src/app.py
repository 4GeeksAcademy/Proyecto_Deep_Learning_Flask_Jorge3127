from flask import Flask, request, render_template
from pickle import load
import os
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

app = Flask(__name__)

# Ruta al archivo del modelo
model_path = "/workspaces/Proyecto_Deep_Learning_Flask_Jorge3127/models/modelo_adaboost_optimizado_corrido.pkl"

# Verificar la existencia del archivo del modelo
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo {model_path} no existe")

# Cargar el modelo
with open(model_path, "rb") as f:
    model = load(f)

# Verificar el tipo de modelo
if not isinstance(model, (AdaBoostClassifier, AdaBoostRegressor)):
    raise TypeError("El archivo cargado no es un modelo AdaBoost válido.")

# Diccionario de clases
class_dict = {
    "0": "Diabético",
    "1": "No Diabético",
}

@app.route("/", methods=["GET", "POST"])
def index():
    pred_class = None

    if request.method == "POST":
        try:
            # Obtener valores del formulario
            val1 = float(request.form.get("embarazos", 0))
            val2 = float(request.form.get("glucosa", 0))
            val3 = float(request.form.get("insulina", 0))
            val4 = float(request.form.get("bmi", 0))
            val5 = float(request.form.get("diabetes", 0))
            val6 = float(request.form.get("edad", 0))

            # Validar los datos ingresados
            if any(v < 0 for v in [val1, val2, val3, val4, val5, val6]):
                pred_class = "Error: Todos los valores deben ser positivos."
            else:
                # Crear un array para la predicción
                data = [[val1, val2, val3, val4, val5, val6]]

                # Realizar la predicción
                prediction = str(model.predict(data)[0])
                pred_class = class_dict.get(prediction, "Clase desconocida")

        except ValueError:
            pred_class = "Error: Ingrese valores numéricos válidos."
        except Exception as e:
            pred_class = f"Error en la predicción: {e}"

    return render_template("formulario.html", prediction=pred_class)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
