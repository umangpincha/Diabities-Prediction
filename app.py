from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('diabetes_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    int_features = [x for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    output = prediction[0]
    if output == 0:
        return render_template('home.html', prediction_text= 'Woo... You are not diabitic!')
    else:
        return render_template('home.html', prediction_text= 'You are diabitic!')


if __name__ == "__main__":
    app.run(debug=True)
