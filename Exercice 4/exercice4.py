import numpy as np
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

GENERATE_BEST_PARAM = False

# model:
# régression linéaire
# decision tree
# random forest
# neural network (lib: pytorch, tensorflow keras)


def lin_regression(x_train, y_train, x_test, y_test):
    clf = {
        "Linear": LinearRegression().fit(x_train, y_train),
        "Ridge": Ridge(10).fit(x_train, y_train),
        "Lasso": Lasso(5.).fit(x_train, y_train)
    }
    for alg in clf:
        y_pred = clf[alg].predict(x_test)
        print(f'{alg} model accuracy :')
        print(f'\tTrain accuracy {round(clf[alg].score(x_train, y_train), 5)}')
        print(f'\tTest accuracy {round(clf[alg].score(x_test, y_test), 5)}')
        print(f'\tR2 score {round(r2_score(y_test, y_pred), 5)}')


def decision_tree(x_train, y_train, x_test, y_test):
    filterwarnings('ignore')
    x_train = x_train.reshape(x_train.size)
    x_test = x_test.reshape(x_test.size)

    lab = LabelEncoder()
    x_train = lab.fit_transform(x_train)
    x_test = lab.fit_transform(x_test)
    y_train = lab.fit_transform(y_train)
    y_test = lab.fit_transform(y_test)

    x_train = x_train.reshape(180, 100)
    x_test = x_test.reshape(20, 100)

    clf = {
        "Gini": DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5),
        "Entropy": DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
    }
    for algo in clf:
        clf[algo].fit(x_train, y_train)
        y_pred = clf[algo].predict(x_test)
        print(f'Criterion parameter used: {algo}')
#        print(f"Report: {classification_report(y_test, y_pred)}")
        print(f"F1_score: {round(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)), 5)}")
        print(f'R2 score {round(r2_score(y_test, y_pred), 5)}')
    filterwarnings('always')


def rand_forest(x_train, y_train, x_test, y_test):
    filterwarnings('ignore')
    y_train = y_train.reshape(y_train.size)

    if GENERATE_BEST_PARAM:
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }
        clf = RandomForestRegressor()
        clf_opti = RandomizedSearchCV(estimator=clf,
                                      param_distributions=random_grid,
                                      n_iter=100,
                                      cv=3,
                                      verbose=2,
                                      random_state=42,
                                      n_jobs=-1).fit(x_train, y_train)

        print(f'Best parameter for this dataset : {clf_opti.best_params_}')

    else:
        best_param = {'n_estimators': 1000,
                      'min_samples_split': 5,
                      'min_samples_leaf': 4,
                      'max_features': 'auto',
                      'max_depth': 100,
                      'bootstrap': True}
        clf_opti = RandomForestRegressor(n_estimators=best_param['n_estimators'],
                                         min_samples_split=best_param['min_samples_split'],
                                         min_samples_leaf=best_param['min_samples_leaf'],
                                         max_features=best_param['max_features'],
                                         max_depth=best_param['max_depth'],
                                         bootstrap=best_param['bootstrap']).fit(x_train, y_train)

    y_pred = clf_opti.predict(x_test)
    print(f'Train accuracy {round(clf_opti.score(x_train, y_train), 5)}')
    print(f'Test accuracy {round(clf_opti.score(x_test, y_test), 5)}')
    print(f'R2 score {round(r2_score(y_test, y_pred), 5)}')

    filterwarnings('always')


def main():
    inputs = np.load("inputs.npy")
    labels = np.load("labels.npy")

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.1, random_state=25)

#     print("===== Linear Regression model =====\n")
#     lin_regression(train_inputs, train_labels, test_inputs, test_labels)

#     print("===== Decision Tree model =====\n")
#     decision_tree(train_inputs, train_labels, test_inputs, test_labels) #decision tree pas adapté pour le model ?

    print("===== Random Forest model =====\n")
    rand_forest(train_inputs, train_labels, test_inputs, test_labels)


if __name__ == '__main__':
    main()
