from flask import Flask, request, render_template
from pickle import load
import os
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
import logging
import sys

# Configurar logs para Render
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

app = Flask(__name__, template_folder='templates')

# Ruta dinámica para el modelo
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_adaboost_optimizado_corrido.pkl")

# Agregar logs para verificar la ruta y la existencia del archivo
app.logger.debug(f"Ruta absoluta calculada del modelo: {os.path.abspath(model_path)}")
app.logger.debug(f"¿Existe el archivo?: {os.path.exists(model_path)}")

# Verificar la existencia del archivo del modelo
if not os.path.exists(model_path):
    app.logger.error(f"El archivo {model_path} no existe")
    raise FileNotFoundError(f"El archivo {model_path} no existe")

# Cargar el modelo
try:
    with open(model_path, "rb") as f:
        model = load(f)
    app.logger.info("Modelo cargado correctamente.")
except Exception as e:
    app.logger.error(f"Error al cargar el modelo: {e}")
    raise

# Verificar el tipo de modelo
if not isinstance(model, (AdaBoostClassifier, AdaBoostRegressor)):
    app.logger.error("El archivo cargado no es un modelo AdaBoost válido.")
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
            # Obtener valores del formulario con manejo de comas
            val1 = float(request.form.get("embarazos", "0").replace(",", "."))
            val2 = float(request.form.get("glucosa", "0").replace(",", "."))
            val3 = float(request.form.get("insulina", "0").replace(",", "."))
            val4 = float(request.form.get("bmi", "0").replace(",", "."))
            val5 = float(request.form.get("diabetes", "0").replace(",", "."))
            val6 = float(request.form.get("edad", "0").replace(",", "."))

            # Validar los datos ingresados
            if any(v < 0 for v in [val1, val2, val3, val4, val5, val6]):
                pred_class = "Error: Todos los valores deben ser positivos."
            else:
                # Crear un array para la predicción
                data = [[val1, val2, val3, val4, val5, val6]]

                # Realizar la predicción
                prediction = str(model.predict(data)[0])
                pred_class = class_dict.get(prediction, "Clase desconocida")

        except ValueError as e:
            app.logger.error(f"Error de conversión: {e}. Datos recibidos: {request.form}")
            pred_class = "Error: Ingrese valores numéricos válidos."
        except Exception as e:
            app.logger.error(f"Error en la predicción: {e}")
            pred_class = f"Error en la predicción: {e}"

    return render_template("formulario.html", prediction=pred_class)


if __name__ == "__main__":
    app.logger.info("Iniciando la aplicación Flask...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
