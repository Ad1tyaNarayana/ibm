import pickle
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS  # Import CORS from flask_cors module

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# Define your predict function here.
def predict(data):
    # Load the pickled model.
    model = pickle.load(open('model.pk1', 'rb'))

    # Make a prediction on the input data.
    prediction = model.predict(data)

    return prediction

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # Get data from the POST request.
    request_data = request.json  # Access the entire JSON object
    input_data = request_data.get('data')  # Access the 'data' key

    if input_data is None:
        return jsonify({'error': 'Data not provided in the request.'}), 400

    # Call the predict function.
    result = predict(np.array([input_data]))  # Wrap the input data in a NumPy array

    # Send the result to the front end.
    return jsonify({'result': result.tolist()})  # Convert NumPy array to a list for JSON serialization

if __name__ == '__main__':
    app.run(debug=True)
