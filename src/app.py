# coloco el còdigo
from flask import Flask, request, render_template
from pickle import load
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

app = Flask(__name__)

# Cargar el modelo
model = load(open("/workspaces/Proyecto_Deep_Learning_Flask_Jorge3127/models/modelo_adaboost_optimizado.pkl", "rb"))

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
