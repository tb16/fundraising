import pandas as pd
from sklearn.ensemble import RandomForestClassifier, \
GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
 confusion_matrix, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
import cPickle as pickle


def run_model(Model, X_train, X_test, y_train, y_test):
    '''
    Run the model and returns the output scores
    Input : Model, eg, RandomForestClassifier, and data: train and test with the labels
    Output: scores computed from the test data
    '''
    m = Model()
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    return accuracy_score(y_test, y_predict), \
        f1_score(y_test, y_predict), \
        precision_score(y_test, y_predict), \
        recall_score(y_test, y_predict),\
        roc_auc_score(y_test, y_predict)


def compare_models(X, y, models):
    '''
    Summarizes the models and print their scores.
    Input: X_data, y_data and models
    Output: None
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    X_train1, X_train2,y_train1, y_train2 = train_test_split(X_train, y_train, test_size = 0.8, random_state = 1234)


    print '='*80
    print "acc\tf1\tprec\trecall\tauc score"
    for Model in models:
        name = Model.__name__
        acc, f1, prec, rec, auc = run_model(Model, X_train2, X_test, y_train2, y_test)
        print "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, auc, name)
    print '='*80


def baseline_mat():
    '''
    returns the X and y data from the dataframe, selecting only few features.
    Input: None
    Output: X and y data
    '''
    df = pd.read_csv('../data/featured_data.csv')
    y = (df.percentage > 0.5).astype(int)
    use_cols = ['days','goal','shares']
    df = df[use_cols]
    X = df.values
    return X, y


def feature_mat():
    '''
    returns the feature matrix and labels
    Input : None
    Output: X: feature_matrix, y: array of true predictions, df: dataframe of feature_matrix
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
    col_name = ['nmf_topic_' + str(i) for i in xrange(0,10)]
    df3 = pd.DataFrame(nmf_topics, columns = col_name)
    df = pd.concat((df, df3), axis = 1)
    df['mnb_probs'] = mnb_model.predict_proba(sparse_mat)[:,1]

    y = (df.percentage > 0.5).astype(int)
    drop_list = ['category', 'name','place','story','title','date_created','date_recorded','raised','people','percentage',\
                'average_contribution']

    df.drop(drop_list, axis =1, inplace = True)

    X = df.values

    return X, y, df


def save_df():
    '''
    takes the output df from feature_mat and saves it.
    '''
    X, y, df = feature_mat()
    df.to_csv('../data/featured_data_final.csv', index = False)


if __name__ == '__main__':

    models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,\
    GradientBoostingClassifier, AdaBoostClassifier, KNeighborsClassifier, SVC]
    print '='*80
    print 'baseline scores....'
    X, y = baseline_mat()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    compare_models(X, y, models)

    print 'scores after featurizing....'

    X, y, df = feature_mat()
    compare_models(X, y, models)
    # save_df()
