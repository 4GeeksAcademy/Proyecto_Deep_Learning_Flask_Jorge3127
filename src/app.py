from flask import Flask, request, render_template
from pickle import load
import os
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

app = Flask(__name__)

# Ruta al archivo del modelo
model = "/opt/render/project/src/models/modelo_adaboost_optimizado_corrido.pkl"

# Verificar la existencia del archivo del modelo
if not os.path.exists(model):
    raise FileNotFoundError(f"El archivo {model} no existe")

# Cargar el modelo
with open(model, "rb") as f:
    model = load(f)

# Diccionario de clases
class_dict = {
    "0": "Diabètico",
    "1": "No Diabètico",
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Obtener valores del formulario
        val1 = float(request.form["embarazos"])
        val2 = float(request.form["glucosa"])
        val3 = float(request.form["insulina"])
        val4 = float(request.form["bmi"])
        val5 = float(request.form["diabetes"])
        val6 = float(request.form["edad"])

        data = [[val1, val2, val3, val4, val5, val6]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None

    return render_template("formulario.html", prediction=pred_class)

if __name__ == "__main__":
    app.run(debug=True)
