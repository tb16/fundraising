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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from roc_confusion import plot_roc, plot_confusion_matrix
from model import feature_mat


def random_forest():
    '''

    '''
    rf = RandomForestClassifier(n_jobs = -1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    accuracy = rf.score(X_test, y_test)
    return accuracy, auc_score


def logistic():
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    accuracy = lr.score(X_test, y_test)
    return accuracy, auc_score


def decisiontree():
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    accuracy = dt.score(X_test, y_test)
    return accuracy, auc_score

def gradboost():
    gbc = GradientBoostingClassifier(n_estimators=500, max_depth=8, subsample=0.5,
                                 max_features='auto', learning_rate=0.05)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    accuracy = gbc.score(X_test, y_test)
    return accuracy, auc_score


def knn():
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)
    y_pred = kn.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    accuracy = kn.score(X_test, y_test)
    return accuracy, auc_score


def svm():
    sv = SVC()
    sv.fit(X_train, y_train)
    y_pred = sv.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    accuracy = sv.score(X_test, y_test)
    return accuracy, auc_score


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



def run_model():



    print 'random forest: ', random_forest()
    print 'logistic: ', logistic()
    print 'decision tree: ', decisiontree()
    print 'grad boost: ', gradboost()
    print 'knn: ', knn()
    print 'svm: ', svm()

def baseline_mat():
    df = pd.read_csv('../data/featured_data.csv')
    y = (df.percentage > 0.5).astype(int)
    use_cols = ['days','goal','shares']
    df = df[use_cols]
    X = df.values
    return X, y


if __name__ == '__main__':
    print 'baseline scores....'
    print '================================'
    X, y = baseline_mat()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    run_model()
    print 'scores after featurizing....'
    print '================================='
    X, y = feature_mat()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    run_model()
