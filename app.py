from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

try:
    model = joblib.load('model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

def calculate_zscore(value):
    try:
        return (value - np.mean([value])) / np.std([value]) if np.std([value]) != 0 else 0
    except Exception as e:
        print(f"Error calculating z-score: {e}")
        return 0


def calculate_log_age(age):
    return np.log(age) if age > 0 else np.nan

def validate_input(value, field_name):
    if not value:
        raise ValueError(f'{field_name} cannot be empty')
    try:
        value = float(value)
        return value
    except (ValueError, TypeError):
        raise ValueError(f'Invalid input for {field_name}, please enter a valid number.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Age = validate_input(request.form['Age'], 'Age')
        PlasmaGlucose = validate_input(request.form['PlasmaGlucose'], 'Plasma Glucose')
        DiastolicBloodPressure = validate_input(request.form['DiastolicBloodPressure'], 'Diastolic Blood Pressure')
        TricepsThickness = validate_input(request.form['TricepsThickness'], 'Triceps Thickness')
        SerumInsulin = validate_input(request.form['SerumInsulin'], 'Serum Insulin')
        BMI = validate_input(request.form['BMI'], 'BMI')
        DiabetesPedigree = validate_input(request.form['DiabetesPedigree'], 'Diabetes Pedigree')
        Pregnancies = validate_input(request.form['Pregnancies'], 'Pregnancies')

        log_Age = calculate_log_age(Age)
        zscore_glucose = calculate_zscore(PlasmaGlucose)
        zscore_pressure = calculate_zscore(DiastolicBloodPressure)
        zscore_thick = calculate_zscore(TricepsThickness)
        zscore_insulin = calculate_zscore(SerumInsulin)
        zscore_bmi = calculate_zscore(BMI)

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform([[Pregnancies, DiabetesPedigree]])
        minMaxPreg, minMaxPedigree = scaled_features[0]

        features = pd.DataFrame([[log_Age, zscore_glucose, zscore_pressure, zscore_thick, 
                                zscore_insulin, zscore_bmi, minMaxPreg, minMaxPedigree]],
                                columns=["log_Age", "zscore_glucose", "zscore_pressure", "zscore_thick", 
                                        "zscore_insulin", "zscore_bmi", "minMaxPreg", "minMaxPedigree"])

        print("Fitur yang akan diprediksi:")
        print(features)
        print("Shape:", features.shape)

        imputer = SimpleImputer(strategy='median')
        features = imputer.fit_transform(features)

        prediction = model.predict(features)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')


if __name__ == "__main__":
    app.run(debug=True)
