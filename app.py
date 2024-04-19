from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
print("Model loaded successfully:", model)  # Add this line for debugging
print("Model type:", type(model))  # Add this line for debugging

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_Fraud():
    Product_Age = float(request.form.get('Product_Age'))
    Service_Centre = int(request.form.get('Service_Centre'))
    Claim_Value = int(request.form.get('Claim_Value'))

    #prediction
    print("Input data:", [Product_Age, Service_Centre, Claim_Value])  # Add this line for debugging
    try:
        result = model.predict(np.array([Product_Age, Service_Centre, Claim_Value]).reshape(1, 3))
        if result[0] == 1 and Claim_Value < 100000:
            result = 'Fraud'
        else:
            result = 'not Fraud'
    except Exception as e:
        print("Error:", e)  # Add this line for debugging
        result = "Error occurred during prediction"

    return str(result)

if __name__ == '__main__':
    app.run(debug=True)
