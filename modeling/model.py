from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB as mnb
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import cPickle as pickle



def rf_model(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(max_depth= 4,max_features= 57,min_samples_leaf= 1,\
 n_estimators= 100, n_jobs = -1).fit(X_train, y_train)
    recall = cross_val_score(rf, X_train, y_train, scoring = 'recall', cv = 5).mean()
    precision = cross_val_score(rf, X_train, y_train, scoring = 'precision', cv = 5).mean()
    accuracy = cross_val_score(rf, X_train, y_train, scoring = 'accuracy', cv = 5).mean()
    print 'validation set'
    print 'recall: ',recall
    print 'precision: ', precision
    print 'accuracy: ', accuracy
    y_pred = rf.predict(X_test)
    print

    accuracy  = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall =    recall_score(y_test, y_pred)
    importances = rf.feature_importances_
    important_indices = np.argsort(rf.feature_importances_)[-1::-1]
    print
#     importance_summary = dict()
#     for feature, importance in izip(df_X.columns[important_indices], importances[important_indices]):
#          importance_summary[feature] = importance
    return rf, accuracy, precision, recall


def search_best_params(X, y):
    '''


    '''
    rf = RandomForestClassifier(n_jobs = -1)
    param_grid = {
              'n_estimators' : [100, 200, 400],
              'max_depth' : [1, 2, 4],
              'min_samples_leaf': [1, 2, 4],
              'max_features': [10, 30, 57]
             }
    gs_cv = GridSearchCV(rf, param_grid, scoring = 'accuracy').fit(X,y)
    return gs_cv.best_params_, gs_cv.best_estimator_



def feature_mat():
    '''
    returns the feature matrix and labels
    '''
    with open('../data/vectorizer.pkl') as f:
        vectorizer = pickle.load(f)
    with open('../data/sparse_mat.pkl') as f:
        sparse_mat = pickle.load(f)

    with open('../data/nmf_model.pkl') as f:
        nmf_model = pickle.load(f)
    with open('../data/mnb_model.pkl') as f:
        mnb_model = pickle.load(f)
    df = pd.read_csv('../data/featured_data.csv')
    df2 = pd.get_dummies(df.category)
    df = pd.concat((df, df2), axis =1)

    nmf_topics = nmf_model.transform(sparse_mat.toarray())
    df3 = pd.DataFrame(nmf_topics)
    df = pd.concat((df, df3), axis = 1)
    df['mnb_probs'] = mnb_model.predict_proba(sparse_mat)[:,1]

    y = (df.percentage > 0.5).astype(int)
    drop_list = ['category', 'name','place','story','title','date_created','date_recorded','raised','people','percentage',\
                'average_contribution']

    df.drop(drop_list, axis =1, inplace = True)
    X = df.values

    return X, y


def pickle_model(model_name, X, y):
    '''

    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    X_train1, X_train2,y_train1, y_train2 = train_test_split(X_train, y_train, test_size = 0.8, random_state = 1234)
    with open('../data/{}_model.pkl'.format(model_name), 'w') as f:
        pickle.dump(rf, f)

if __name__ == '__main__':
    X, y = feature_mat()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    X_train1, X_train2,y_train1, y_train2 = train_test_split(X_train, y_train, test_size = 0.8, random_state = 1234)
    rf, accuracy, precision, recall = rf_model(X_train2, X_test, y_train2, y_test)
    print accuracy, precision, recall
