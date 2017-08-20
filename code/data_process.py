#!usr/bin/env python3

from os import listdir
from os.path import join, isfile
import re

import pandas as pd
import pickle


def get_annotated(path2output, filename="annotated.csv"):
    """Generate a dictionary where each paper has a list of dimension values."""
    data = pd.read_csv(join(path2output, filename))
    papers = data.Paper.unique()
    dimensions = data.Dimension.unique()
    annotated = {}
    for paper in papers:
        transaction = {}
        for dimension in dimensions:
            sel = ((data.Paper == paper) &
                   (data.Dimension == dimension))
            transaction[dimension] = list(data[sel].Value)
        annotated[paper] = transaction
    return annotated


def get_dimension_values(annotated, dimension):
    """Get a dimension values for all annotated paper."""
    values = []
    for paper in annotated:
        values += annotated[paper][dimension]
    return values


def find_all_paper(base, path2data, subfolder, max_num=500):
    """Find a list of filenames and paper id."""
    count = 0
    for sub in subfolder:
        for item in listdir(join(base, path2data, sub)):
            myfile = join(base, path2data, sub, item)
            if isfile(myfile):
                count += 1
                if count <= max_num:
                    yield myfile, item.rstrip(".txt")


def search_neighbor(word, line, n=2):
    """Searches for line, and retrieves n words either side of the word,
     which are retuned separately."""
    text = r"\W*(\w+)"
    groups = []
    match = re.search(r"{}\W*{}{}".format(text * n, word, text * n),
              line, re.IGNORECASE)
    if match:
        groups = [group for group in match.groups() if group.isalpha()]
    return " ".join(groups)


def search_paper(value, base, path2data, subfolder):
    """Search a dimension value in all paper and find its neighbor words in
    each paper."""
    value = value.lower()
    file_list = find_all_paper(base, path2data, subfolder)
    for file, paper in file_list:
        with open(file, "r") as f:
            text = f.readlines()
        text = " ".join(text)
        if text.count(value) >= 1:
            neighbors = search_neighbor(value, text)
            yield paper, neighbors


def get_neighbors_paper(annotated, dimension, base, path2data, subfolder):
    """Return a list of neighbor words corresponds to a list of values and
    a list of dictionary of (paper, value)."""
    values = get_dimension_values(annotated, dimension)
    for value in set(values):
        total = [(paper, neighbor) for paper, neighbor in
                 search_paper(value, base, path2data, subfolder)]
        paper_list, neighbors = zip(*total)
        yield " ".join(neighbors), dict([(value, list(paper_list))])


def annotate_all(annotated, base, path2data, subfolder, path2output):
    """Annotate all paper with original annotated keywords."""
    neighbors_tot = []
    value2dim = {}
    value2paper = {}
    for dimension in ["Problem", "Method", "Dataset", "Metric"]:
        temp = get_neighbors_paper(annotated, dimension, base,
                                   path2data, subfolder)
        for neighbors, value_dict in temp:
            neighbors_tot += [(neighbors, dimension)]
            value2paper.update(value_dict)
            value2dim[list(value_dict.keys())[0]] = dimension
    neighbors_list, target = zip(*neighbors_tot)
    annotated_all = {"neighbors_list": list(neighbors_list),
                     "target": list(target),
                     "value2paper": value2paper,
                     "value2dim": value2dim}
    output_file = join(path2output, "annotated_all.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(annotated_all, f, protocol=pickle.HIGHEST_PROTOCOL)
    return list(neighbors_list), list(target), value2paper, value2dim


def get_value2key(key2value):
    """Get a dictionary with (value: key) given a dictionary with
     (key: value)."""
    value2key = {}
    for key in key2value:
        if type(key2value[key]) != list:
            value_list = [key2value[key]]
        else:
            value_list = key2value[key]
        for value in value_list:
            if value not in value2key:
                value2key[value] = [key]
            else:
                value2key[value].append(key)
    return value2key