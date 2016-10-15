import cPickle as pickle
import pandas as pd
from get_data_featurize import get_data, featurizing

from flask import Flask, request, render_template

app = Flask(__name__)

with open('../data/randomforest.pkl') as f:
    model = pickle.load(f)


# Homepage with form on it.
# ================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def show_form():
    return render_template('form.html')

# Once submit is hit, pass info into model, return results.
# ================================================
@app.route('/predict', methods=['POST'])
def predict():
    # get data from request form
    url = request.form['user_input']

    if request.form['field_type'] == 'url':
        pass

    elif request.form['field_type'] == 'story':
        pass

    # convert data from unicode to string
    data = get_data(url)

    # vectorize our new data
    X, y, df = featurizing(data)
    y_true = y
    # make prediction based on new data
    pred = model.predict(X)[0]

    # save the image to a file.png

    # return string format of that prediction to the html page
    return '' + str(pred) + ' ' + str(y_true)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
