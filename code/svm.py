#!usr/bin/env python3

from os.path import join
from itertools import product

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from data_process import *


def compute_f1(neighbors_list, target, n_components, train_size,
               numfeatures=1000):
    temp = [(neighbors, t) for neighbors, t in zip(neighbors_list, target)]
    temp = list(filter(lambda x: x[1] not in ["Dataset", "Metric"], temp))
    neighbors_list, target = zip(*temp)
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.5,
                                 ngram_range=(1, 3),
                                 max_features=numfeatures,
                                 stop_words="english",
                                 use_idf=True)
    X = vectorizer.fit_transform(neighbors_list)
    pca = PCA(n_components=n_components)
    X_r = pca.fit(X.A).transform(X.A)
    y = list(map(lambda x: 1 if x == "Method" else 0, target))
    X_train, X_test, y_train, y_test = train_test_split(X_r, y,
                                                        train_size=train_size,
                                                        random_state=0)
    clf = SGDClassifier(loss='log', penalty='l1', alpha=1e-3, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return round(f1_score(y_test, y_pred), 2)


def compute_f1_all(neighbors_list, target):
    components = np.arange(10, 121, 10)
    train_sizes = np.linspace(0.25, 0.95, 15)
    f1_matrix = np.zeros((len(components), len(train_sizes)))
    f1_max = 0
    n_components = 10
    train_size = 0.3
    for i, j in product(range(len(components)), range(len(train_sizes))):
        f1_score = compute_f1(neighbors_list, target,
                              components[i], train_sizes[j])
        f1_matrix[i, j] = f1_score
        if f1_score > f1_max:
            f1_max = f1_score
            n_components = components[i]
            train_size = train_sizes[j]
    print("The optimal F1 score is {:.0f}%".format(100 * f1_max))
    print("We should use {:d} of PCA components "
          "and {:.0f}% of training data".format(n_components,
                                                100 * train_size))
    return f1_max, f1_matrix

def plot_f1(f1_matrix, path2output):
    sns.set_style("ticks", {"xtick.direction": "in",
                            "ytick.direction": "in"})
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(f1_matrix, cmap=plt.cm.viridis,
                            interpolation="nearest", origin="lower",
                            extent=[25, 95, 10, 120], aspect="auto")
    plt.setp(ax, xlabel="Training size (%)", ylabel="# of Principle components")
    plt.colorbar(im)
    plt.savefig(join(path2output, "F1_Scores.pdf"))
    plt.close(fig)


def run_svm(neighbors_list, target, path2output):
    f1_max, f1_matrix = compute_f1_all(neighbors_list, target)
    plot_f1(f1_matrix, path2output)
    return f1_max