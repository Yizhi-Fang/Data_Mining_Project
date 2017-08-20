#!usr/bin/env python3

from os.path import join, exists
import argparse

from data_process import *
from cube_construction import *
from apriori import *
from kmeans import *
from svm import *


def parse_options():
    optparser = argparse.ArgumentParser(description="Data mining project.")
    optparser.add_argument(
        "-b", "--base",
        dest="base",
        help="base folder to the project folder",
        default="/Users/yzhfang/Documents/Course/Intro to Data Mining/Project"
    )
    optparser.add_argument(
        "-c", "--cube",
        dest="data_cube",
        help="indicating if constructing data cube",
        default=0,
        type=int
    )
    return optparser.parse_args()

if __name__ == "__main__":
    options = parse_options()
    base = options.base

    path2data = join(base, "data")
    subfolder = ["kdd15", "kdd16"]
    path2output = join(base, "output")

    # Get original annotated data first.
    annotated = get_annotated(path2output)

    # Annotate all paper (430 in total).
    annotate_file = join(path2output, "annotated_all.pkl")
    if exists(annotate_file):
        with open(annotate_file, "rb") as f:
            annotated_results = pickle.load(f)
        neighbors_list = annotated_results["neighbors_list"]
        target = annotated_results["target"]
        value2paper = annotated_results["value2paper"]
        value2dim = annotated_results["value2dim"]
    else:
        neighbors_list, target, value2paper, value2dim = (
            annotate_all(annotated, base, path2data, subfolder, path2output))

    # Construct cube.
    if options.data_cube:
        print("\nConstructing data cube...")
        construct_cube(value2paper, value2dim, path2output)

    # Get trasactions as (paper: value).
    transactions = get_value2key(value2paper)

    # Run Apriori model to find frequent patterns in Problem and Method.
    min_support = 0.1
    min_confidence = 0.8
    print("\nRunning Apriori...")
    print("min_support = {:.1f} and "
          "min_confidence = {:.1f}".format(min_support, min_confidence))
    run_apriori(transactions, value2paper, min_support, min_confidence,
                path2output)

    # Run LDA model to mine topics in Method.
    K_clusters = 4
    print("\nRunning K-Means...")
    print("K_clusters = {:d}".format(K_clusters))
    run_kmeans(transactions, value2dim, K_clusters, path2output)

    # Run SGD (Logistic SVM) to classify Method or non-Method.
    print("\nRunning Logistic SVM with SGD training...")
    run_svm(neighbors_list, target, path2output)