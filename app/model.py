from roc_confusion import plot_roc, plot_confusion_matrix
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np

with open('../data/vectorizer.pkl') as f:
    vectorizer = pickle.load(f)
with open('../data/sparse_mat.pkl') as f:
    sparse_mat = pickle.load(f)
with open('../data/randomforest.pkl') as f:
    rf = pickle.load(f)
with open('../data/gradboost.pkl') as f:
    gbc = pickle.load(f)


def plot_importance(clf, max_features=10):
    '''Plot feature importance'''
    
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    # Show only top features
    pos = pos[-max_features:]
    feature_importance = (feature_importance[sorted_idx])[-max_features:]
    feature_names = (.columns[sorted_idx])[-max_features:]

    plt.barh(pos, feature_importance, align='center')
    plt.yticks(pos, feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')


if __name__ == '__main__':
    plot_importance(rf)
