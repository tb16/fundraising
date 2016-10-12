import numpy as np
import pandas as pd
from itertools import izip
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from roc_confusion import plot_roc, plot_confusion_matrix


def random_forest(X_train, y_train, X_test,  y_test):
    '''

    '''
    rf = RandomForestClassifier(n_estimators=100, max_depth=4,min_samples_leaf=2,\
    max_features = 46, n_jobs = -1).fit(X_train, y_train)
    recall = cross_val_score(rf, X_train, y_train, scoring = 'recall', cv = 5).mean()
    precision = cross_val_score(rf, X_train, y_train, scoring = 'precision', cv = 5).mean()
    accuracy = cross_val_score(rf, X_train, y_train, scoring = 'accuracy', cv = 5).mean()
    print 'recall: ',recall
    print 'precision: ', precision
    print 'accuracy: ', accuracy
    y_pred = rf.predict(X_test)
    accuracy  = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall =    recall_score(y_test, y_pred)
    importances = rf.feature_importances_
    important_indices = np.argsort(rf.feature_importances_)[-1::-1]
    print
    importance_summary = dict()
    for feature, importance in izip(df_X.columns[important_indices], importances[important_indices]):
         importance_summary[feature] = importance
    # accuracy, precision, recall, importance_summary
    return rf.score(X_test, y_test), rf.predict_proba(X_test[:,1])

def search_best_params():
    param_grid = {
              'n_estimators' : [100, 200, 400],
              'max_depth' : [1, 2, 4],
              'min_samples_leaf': [1, 2],
              'max_features': [5, 15, 28]
             }
    gs_cv = GridSearchCV(rf, param_grid, scoring = 'recall').fit(X,y)
    return gs_cv.best_params_, gs_cv.best_estimator_


def logistic(X_train, y_train, X_test, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr.score(X_test, y_test)


def decisiontree(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt.score(X_test, y_test)


def gradboost(X_train, y_train, X_test, y_test):
    gbc = GradientBoostingClassifier(n_estimators=500, max_depth=8, subsample=0.5,
                                 max_features='auto', learning_rate=0.05)
    gbc.fit(X_train, y_train)
    return gbc.score(X_test,y_test)


def knn(X_train, y_train, X_test, y_test):
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)
    return kn.score(X_test, y_test)


def svm(X_train, y_train, X_test, y_test):
    sv = SVC()
    sv.fit(X_train, y_train)
    return sv.score(X_test, y_test)


def plot_importance(clf, X, max_features=10):
    '''Plot feature importance'''
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    # Show only top features
    pos = pos[-max_features:]
    feature_importance = (feature_importance[sorted_idx])[-max_features:]
    feature_names = (X.columns[sorted_idx])[-max_features:]

    plt.barh(pos, feature_importance, align='center')
    plt.yticks(pos, feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')

if __name__ == '__main__':

    df = pd.read_csv('../data/featured_data.csv')
    y = (df.percentage > 0.5).astype(int)
    dummies_df = pd.get_dummies(df.category)
    df = pd.concat((df, dummies_df), axis =1)
    # use_cols = ['category','friends','days','goal','shares','title_length','story_length','word_count_title',\
    #        'word_count_story','sentence_count_story','prob_mnb']
    drop_cols = ['name','place','story','title','date_created','date_recorded','raised','people','percentage',\
            'average_contribution', 'category']
    df.drop(drop_cols, axis =1, inplace = True)
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    score, probs = random_forest(X_train, y_train, X_test, y_test)
    print 'random forest', score
    plot_roc(probs, y_test, "ROC plot",
         "False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity, Recall)")
    # print 'logistic', logistic(X_train, y_train, X_test, y_test)
    # print 'decision tree', decisiontree(X_train, y_train, X_test, y_test)
    # print 'grad boost', gradboost(X_train, y_train, X_test, y_test)
    # print 'knn', knn(X_train, y_train, X_test, y_test)
    # print 'svm', svm(X_train, y_train, X_test, y_test)
