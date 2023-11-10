import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Initialize the flask App
app = Flask(__name__, static_folder='static')
model = pickle.load(open('models\model.pkl', 'rb'))

# default page of our web-app


@app.route('/')
def home():
    return render_template('index.html')

# To use the predict button in our web-app


@app.route('/predict1', methods=['POST'])
def predict1():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 1:
        prediction = 'Congratulations , you have survived the shipwreck :)'
    else:
        prediction = 'Sorry , you are no more !'

    return render_template('index.html', prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)
