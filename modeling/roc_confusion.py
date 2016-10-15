from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def roc_curve(probabilities, labels):

    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []
    # print sum(labels)
    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = (probabilities >= threshold)
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()


def plot_roc(probs, y_test, title, xlabel, ylabel):
    '''
    plot and save the roc.
    Input: probs: probabilities, array, y_tets: true response, array
        title, xlabel and ylabels: string
    Output: None
    '''
    # ROC curve
    tpr, fpr, thresholds = roc_curve(probs, y_test)

    plt.hold(True)
    plt.plot(fpr, tpr)

    # 45 degree line
    xx = np.linspace(0, 1.0, 20)
    plt.plot(xx, xx, color='red')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('../images/roc_plot.png')
    plt.show()


def plot_confusion_matrix(model, X_test, y_test):
    '''
    plot and save confusion matrix.
    '''

    name = model.__repr__().split('(')[0]

    cm = confusion_matrix(y_test, model.predict(X_test))

    print(cm)

    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../images/confusion_matrix_{}.png'.format(name))
    plt.show()


if __name__=='__main__':
    plot_roc(probs, y_test, "ROC plot",
         "False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity)")
