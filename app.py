from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib

# Load the trained model and scaler
model_path = 'model.pkl'
scaler_path = r"C:\Users\Asus\Downloads\scaler.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

scaler = joblib.load(scaler_path)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        int_features = [float(x) for x in request.form.values()]
        final_features = scaler.transform([np.array(int_features)])
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'
    
    except Exception as e:
        output = f'Error: {str(e)}'
    
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
