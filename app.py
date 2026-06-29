from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('floods.save')
scaler = joblib.load('transform.save')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # 1. Collect inputs
        try:
            data = [
                float(request.form['Cloud Cover']),
                float(request.form['ANNUAL']),
                float(request.form['Jan-Feb']),
                float(request.form['Mar-May']),
                float(request.form['Jun-Sep'])
            ]
            
            # 2. DataFrame and Scaling
            features = pd.DataFrame([data], columns=['Cloud Cover', 'ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep'])
            scaled_input = scaler.transform(features.values)
            
            # 3. Predict
            prediction = model.predict(scaled_input)[0]
            
            # 4. Redirect based on model output
            # If 1 is Flood, you see chance.html. If 0 is Safe, you see no_chance.html.
            if int(prediction) == 1:
                return render_template('chance.html')
            else:
                return render_template('no_chance.html')
                
        except Exception as e:
            return f"Error in prediction: {e}"
            
    return render_template('index.html')

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)