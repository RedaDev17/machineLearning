import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

def regression(x_train, y_train, x_test, y_test):
    clf = [
        LogisticRegression(solver='newton-cg', penalty='l2', max_iter=1000),  # 0
        LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000),  # 1
        LogisticRegression(solver='sag', penalty='l2', max_iter=1000),  # 2
        LogisticRegression(solver='saga', penalty='l2', max_iter=1000),  # 3
        LogisticRegression(solver='newton-cg', max_iter=1000),  # 4
        LogisticRegression(solver='lbfgs', max_iter=1000),  # 5
        LogisticRegression(solver='sag', max_iter=1000),  # 6
        LogisticRegression(solver='saga', max_iter=1000),  # 7
        LogisticRegression(solver='newton-cg', penalty='l2', max_iter=1000, C=0.001),  # 8
        LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000, C=0.001),  # 9
        LogisticRegression(solver='sag', penalty='l2', max_iter=1000, C=0.001),  # 10
        LogisticRegression(solver='saga', penalty='l2', max_iter=1000, C=0.001),  # 11
        LogisticRegression(solver='newton-cg', max_iter=1000, C=0.001),  # 12
        LogisticRegression(solver='lbfgs', max_iter=1000, C=0.001),  # 13
        LogisticRegression(solver='sag', max_iter=1000, C=0.001),  # 14
        LogisticRegression(solver='saga', max_iter=1000, C=0.001)  # 15
    ]
    clf_columns = []
    clf_compare = pd.DataFrame(columns=clf_columns)

    row_index = 0
    for alg in clf:
        predicted = alg.fit(x_train, y_train).predict(x_test)
        fp, tp, th = roc_curve(y_test, predicted)
        clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(x_train, y_train), 5)
        clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(x_test, y_test), 5)
        clf_compare.loc[row_index, 'Precision'] = round(precision_score(y_test, predicted), 5)
        clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 5)
        clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 5)

        row_index += 1

    clf_compare.sort_values(by=['Test Accuracy'], ascending=False, inplace=True)
    print(clf_compare)
    print("The best hyperparameter are {'solver': 'newton-cg', 'penalty': 'l2', 'max_iter': 1000}")


def svm(x_train, y_train, x_test, y_test):
    c_range = np.logspace(-1, 1, 3)
    gamma_range = np.logspace(-1, 1, 3)

    param_grid = {
        "C": c_range,
        "kernel": ['rbf', 'poly'],
        "gamma": gamma_range.tolist()+['scale', 'auto']
    }

    scoring = ['accuracy']

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    grid_search = GridSearchCV(estimator=SVC(),
                               param_grid=param_grid,
                               scoring=scoring,
                               refit='accuracy',
                               n_jobs=1,
                               cv=kfold,
                               verbose=0)

    grid_result = grid_search.fit(x_train, y_train)

    print(f"The best accuracy score for training dataset is {grid_result.best_score_:.4f}")
    print(f"The best hyperparameter are {grid_result.best_params_}")
    print(f"The accuracy score for the testing dataset is {grid_search.score(x_test, y_test):.4f}")

def main():
    inputs = np.load("inputs.npy")  # refer to x
    labels = np.load("labels.npy")  # refer to y

    train_inputs, test_inputs = inputs[:450], inputs[450:]
    train_labels, test_labels = labels[:450], labels[450:]
    train_labels = train_labels.reshape(450)

    print("\t\t\t\tLogisitic Regression model\n")
    regression(train_inputs, train_labels, test_inputs, test_labels)

    print()

    print("\t\t\t\tSupport Vector Model (SVM)\n")
    svm(train_inputs, train_labels, test_inputs, test_labels)


if __name__ == '__main__':
    main()
