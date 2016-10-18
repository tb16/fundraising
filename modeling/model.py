from roc_confusion import plot_roc, plot_confusion_matrix
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import pandas as pd
from campaign_rec import similar_campaign
from model_comparison import feature_mat
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    roc_auc_score


'''
For the given model, printing/analysing different features and model output.
'''

with open('../data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)
with open('../data/sparse_mat.pkl') as f:
    sparse_mat = pickle.load(f)
with open('../data/randomforest.pkl') as f:
    rf = pickle.load(f)
with open('../data/gradboost.pkl') as f:
    gbc = pickle.load(f)


def plot_importance(clf, max_features=30):
    '''
    Plot and save feature importance.
    Input: clf: classifier model, max_features: int
    Output: None
    '''
    model_name = clf.__repr__().split('(')[0]
    feature_names = df.columns
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    # Show only top features
    pos = pos[-max_features:]
    feature_importance = (feature_importance[sorted_idx])[-max_features:]
    # feature_names = (.columns[sorted_idx])[-max_features:]
    plt.figure(figsize = (12, 10))
    plt.barh(pos, feature_importance, align='center')
    plt.yticks(pos, (feature_names[sorted_idx])[-max_features:])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance {}'.format(model_name))
    plt.savefig('../images/feat_im_{}.png'.format(model_name), bbox_inches = 'tight')
    plt.show()



def show_scores(model, X_test, y_test):
    '''
    calculate evaluation scores of a model
    Input: model, X_test: feature matrix or array, y_test: true value of response
    Output: float, float, float, float
    '''
    y_predict = model.predict(X_test)
    return {'accuracy ': accuracy_score(y_test, y_predict), \
        'precision ': precision_score(y_test, y_predict), \
        'recall ': recall_score(y_test, y_predict),\
        'auc ': roc_auc_score(y_test, y_predict)}


def plot_roc_confusion(model, X_test, y_test):
    '''
    plots the roc curve and confusion matric from the given model and the data
    print and save roc plot and confusion
    Input: model: classifier, X_test : input array or matrix of predictor,
        y_test: array of true value of prediction
    Output : None
    '''
    y_pred = rf.predict(X_test)
    probs = rf.predict_proba(X_test)[:,1]
    plot_roc(probs, y_test, "ROC plot",\
         "False Positive Rate", "True Positive Rate")
    plot_confusion_matrix(model, X_test, y_test)


if __name__ == '__main__':
    # df = pd.read_csv('../data/featured_data_final.csv')
    # plot_importance(gbc)
    # plot_importance(rf)

    X, y, df = feature_mat()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    # print 'rf: ',show_scores(rf, X_test, y_test)
    # print 'gbc: ',show_scores(gbc, X_test, y_test)
    plot_roc_confusion(gbc, X_test, y_test)
    # vector = sparse_mat[0].toarray()
    # print similar_campaign(vector, vectorizer, sparse_mat)
