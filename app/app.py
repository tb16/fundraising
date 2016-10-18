import pandas as pd
import cPickle as pickle
from get_data_featurize import get_data, featurizing
from campaign_rec import similar_campaign
# from data import model, sparse_mat, vectorizer

from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap



with open('../data/randomforest.pkl') as f:
    model = pickle.load(f)
with open('../data/sparse_mat.pkl') as f:
    sparse_mat = pickle.load(f)
with open('../data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)


app = Flask(__name__)

app.config.update(TEMPLATES_AUTO_RELOAD=True)

Bootstrap(app)


# Homepage with form on it.
# ================================================
@app.route('/', methods = ['GET','POST'])
def index():
    ''' Home page'''
    return render_template('index.html')


@app.route('/welcome')
def about():
    ''' About page '''
    return render_template('welcome.html')


def image():
    return render_template('roc_plot.png')

@app.route('/contact')
def about_me():
    ''' contact page '''
    return render_template('contact.html')



@app.route('/tools')
def tools():
    ''' prediction method and tools page '''
    return render_template('tool.html')


# Once submit is hit, pass info into model, return results.
# ================================================
@app.route('/predict', methods=['GET','POST'])
def predict():
    '''predictions pagecalculations and predictions'''
    url = request.form['user_input']


    # convert data from unicode to string
    data = get_data(url)
    # print 'data', data
    # print 'title: ', data['title']
    # vectorize our new data
    X, y, _ = featurizing(data)
    y_true = y
    raised = data['raised'][0]
    # make prediction based on new data
    pred = model.predict(X)[0]
    prob = round(model.predict_proba(X)[0][1]*100)

    # Similar campaigns, title, keywords, urls
    vector = vectorizer.transform([data['story'][0]])
    # print 'vectorizing'
    similar = similar_campaign(vector, vectorizer, sparse_mat)

    similar = similar.T.to_dict().values()
    # print 'dict', similar[0]['story']
    # print 'dict', similar[1]['story']
    # print 'dict', similar[2]['story']

    data = data.T.to_dict().values()

    # print 'data', data



    return render_template('predictions.html', similar=similar, prob=prob, data = data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
