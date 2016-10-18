
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
import cPickle as pickle
from model_comparison import feature_mat


def gridsearch_randomforest(X, y):
    '''
    Gridsearching for random forest.
    Input : X, y data
    Returns the best parameters
    '''
    rf = RandomForestClassifier(n_jobs = -1)
    param_grid = {
              'n_estimators' : [50, 100, 150],
              'max_depth' : [2, 4, 5, 6],
              'min_samples_leaf': [2, 4, 5, 6],
              'max_features': [20, 30, 40, 57]
             }
    gs_cv = GridSearchCV(rf, param_grid, scoring = 'accuracy').fit(X,y)
    return gs_cv.best_params_


def gridsearch_gradboost(X, y):
    '''
    Gridsearching for Gradient Boost.
    Input : X, y data
    Returns the best parameters

    '''
    gbc = GradientBoostingClassifier()
    param_grid = {
              'learning_rate': [0.05, 0.03, 0.02],
           'max_depth': [2, 4, 6],
           'max_features': [1, 0.5, 0.3],
           'n_estimators': [100, 150]
             }
    print 'gridsearch gradboost started.....'
    gs_cv = GridSearchCV(gbc, param_grid, scoring = 'accuracy').fit(X,y)
    print gs_cv.best_params_
    return gs_cv.best_params_


def gridsearch_adaboost(X, y):
    '''
    Gridsearching for AdaBoost.
    Input : X, y data
    Returns the best parameters

    '''
    abc = AdaBoostClassifier()
    param_grid = {
              'n_estimators' : [150, 500, 1000],
              'learning_rate': [0.02, 0.1, 0.2]
             }
    gs_cv = GridSearchCV(abc, param_grid, scoring = 'accuracy').fit(X,y)
    return gs_cv.best_params_


def gridsearch(X, y):
    '''
    grid search for best params in the model.
    Output: saves the file with best params
    '''


    randomforest = gridsearch_randomforest(X, y)
    gradboost = gridsearch_gradboost(X, y)
    adaboost = gridsearch_adaboost(X, y)
    with open('params.txt', 'w') as f:
        f.write('\nrandomforest: ')
        f.write(str(randomforest))
        f.write('\ngradboost: ')
        f.write(str(gradboost))
        f.write('\nadaboost: ')
        f.write(str(adaboost))


def pickle_model(X, y):
    '''
    saving pickle file for models RandomForest and GradientBoosting
    Input: X: feature matrix, y: array of output value
    '''
    rf = RandomForestClassifier(max_features= 40, n_estimators= 100, max_depth= 5, \
    min_samples_leaf= 5, n_jobs = -1).fit(X, y)
    with open('../data/randomforest.pkl', 'w') as f:
        pickle.dump(rf, f)

    gbc = GradientBoostingClassifier(max_features = 0.3,n_estimators=500,\
    learning_rate = 0.02, max_depth = 6).fit(X_train, y_train)
    with open('../data/gradboost.pkl', 'w') as f:
        pickle.dump(gbc, f)


if __name__ == '__main__':
    X, y, _ = feature_mat()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    X_train1, X_train2,y_train1, y_train2 = train_test_split(X_train, y_train, \
    test_size = 0.8, random_state = 1234)
    # gridsearch_gradboost(X_train2, y_train2)
    pickle_model(X_train2, y_train2)
