from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from sklearn.utils.fixes import signature
import scikitplot as skplt
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def get_report(ytest, ypred, yscore, file):
    """ getting the complete classification report """
    target_names = ['class 0', 'class 1']
    report = classification_report(ytest, ypred, target_names=target_names)

    """ Generating and plotting the confusion matrix """
    ext = "png"
    skplt.metrics.plot_confusion_matrix(ytest, ypred, normalize=True)
    plt.savefig(f"final_results/{file}_confusion.{ext}")
    plt.close()

    """ Generating and plotting the precision recall curve"""
    average_precision = average_precision_score(ytest, yscore, pos_label=0)

    precision, recall, _ = precision_recall_curve(ytest, yscore, pos_label=0)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'2-class Precision-Recall curve: AP={average_precision:0.2f}')
    plt.savefig(f"final_results/{file}_precrec.{ext}")
    plt.close()

    return report


def model_train_and_test(X, y, text=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0, shuffle=True)
    if text:
        files = {0: "random_forest_text", 1: "sgd_text", 2: "knn_text", 3: "logistic_text", 4: "mlp_text",
                 5: "ada_boost_text"}
    else:
        files = {0: "random_forest", 1: "sgd", 2: "knn", 3: "logistic", 4: "mlp", 5: "ada_boost"}
    models = [
        RandomForestClassifier(n_estimators=500, max_depth=3, random_state=0, n_jobs=-1, oob_score=True, warm_start=True),
        SGDClassifier(loss="log", penalty="l2", max_iter=5, n_jobs=-1),
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        LogisticRegression(random_state=0, n_jobs=-1),
        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0,
                      learning_rate='invscaling'),
        AdaBoostClassifier(n_estimators=500, learning_rate=1e-3, random_state=0),

    ]
    metric_reports = []
    for i, n in enumerate(models):
        clf = n
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)
        y_score = [x[0] for x in y_score]
        report_card = get_report(y_test, y_pred, y_score, files[i])
        metric_reports.append(report_card)

    with open("final_results/final_report.txt", "a") as f:
        for i, n in enumerate(metric_reports):
            f.write(f"Classification report for {files[i]}:\n")
            f.write(n)
            f.write("\n")
            f.write("-----------------------------------------------------------------")
            f.write("\n")


def main():
    X = pd.read_csv("data/X.csv")
    X.drop("Unnamed: 0", axis=1, inplace=True)
    X_text = pd.read_csv("data/X_text.csv")
    X_text.drop("Unnamed: 0", axis=1, inplace=True)
    y = pd.read_csv("data/y.csv")
    y.drop("Unnamed: 0", axis=1, inplace=True)
    y_text = pd.read_csv("data/y_text.csv")
    y_text.drop("Unnamed: 0", axis=1, inplace=True)

    model_train_and_test(X, y)
    model_train_and_test(X_text, y_text, text=True)


if __name__ == "__main__":
    main()
