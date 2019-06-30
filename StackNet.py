from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pystacknet.pystacknet import StackNetClassifier
from models_final import get_report

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def stacknet_train_test(X, y, text=False):
    models = [
            # First level
            [RandomForestClassifier(n_estimators=500, max_depth=3, random_state=0),
             ExtraTreesClassifier (n_estimators=100, max_depth=5, random_state=0),
             SGDClassifier(loss="log", penalty="l2", max_iter=5),
             KNeighborsClassifier(n_neighbors=5),
             LogisticRegression(random_state=0),
             MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0, learning_rate='invscaling'),
             AdaBoostClassifier(n_estimators=500, learning_rate=1e-3, random_state=0),
             ],
            # Second level
            [RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)]
    ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0, shuffle=True)

    model = StackNetClassifier(models, metric="f1", folds=4, restacking=True, use_retraining=True, use_proba=True, random_state=0, verbose=1)

    model.fit(X_train, y_train)
    y_init = model.predict_proba(X_test)
    y_pred = [0 if i[0] > i[1] else 1 for i in y_init]
    y_score = [i[0] for i in y_init]

    y_pred = np.array(y_pred)

    files = {0: "stackNet", 1: "stackNet_text"}

    if text:
        name = files[1]
    else:
        name = files[0]

    # this function is imported from models_final.py file
    report_card = get_report(y_test, y_pred, y_score, name)

    with open("final_results/final_report_stackNet.txt", "a") as f:
        f.write(f"Classification report for {name}: \n")
        f.write(report_card)
        f.write("\n")
        f.write("-----------------------------------------------------------------")
        f.write("\n")


def main():
    X = pd.read_csv("X.csv")
    X.drop("Unnamed: 0", axis=1, inplace=True)
    X_text = pd.read_csv("X_text.csv")
    X_text.drop("Unnamed: 0", axis=1, inplace=True)
    y = pd.read_csv("y.csv")
    y.drop("Unnamed: 0", axis=1, inplace=True)
    y_text = pd.read_csv("y_text.csv")
    y_text.drop("Unnamed: 0", axis=1, inplace=True)

    stacknet_train_test(X, y)
    stacknet_train_test(X_text, y_text, text=True)


if __name__ == "__main__":
    main()
