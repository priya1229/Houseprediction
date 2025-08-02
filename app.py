from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Neighborhood list as in sample CSV
neighborhoods = [
    "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
    "Gilbert", "NridgHt", "Sawyer", "NWAmes", "BrkSide"
]

# We must encode neighborhoods same way as in training
# Fit LabelEncoder once here:
le = LabelEncoder()
le.fit(neighborhoods)

@app.route('/')
def home():
    return render_template("index.html", neighborhoods=neighborhoods)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form inputs
        GrLivArea = float(request.form['GrLivArea'])
        BedroomAbvGr = int(request.form['BedroomAbvGr'])
        Neighborhood = request.form['Neighborhood']
        OverallQual = int(request.form['OverallQual'])
        YearBuilt = int(request.form['YearBuilt'])
        FullBath = int(request.form['FullBath'])
        GarageCars = int(request.form['GarageCars'])
        TotRmsAbvGrd = int(request.form['TotRmsAbvGrd'])

        # Encode Neighborhood
        Neighborhood_encoded = le.transform([Neighborhood])[0]

        # Prepare features array
        features = np.array([[GrLivArea, BedroomAbvGr, Neighborhood_encoded, OverallQual,
                              YearBuilt, FullBath, GarageCars, TotRmsAbvGrd]])

        # Predict
        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)

        return render_template("index.html", prediction_text=f"Estimated House Price: ₹ {prediction:,.2f}",
                               neighborhoods=neighborhoods)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}", neighborhoods=neighborhoods)

if __name__ == '__main__':
    app.run(debug=True)
