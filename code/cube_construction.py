#!usr/bin/env python3

from os.path import join
from itertools import product

from data_process import *


def construct_cube(value2ppaer, value2dim, path2output, min_support=2):
    dimensions = ["Problem", "Method", "Dataset", "Metric"]
    dim2value = get_value2key(value2dim)
    values = [dim2value[dim] for dim in dimensions]
    with open(join(path2output, "Dim_Table.csv"), "w") as f:
        f.write(",".join(dimensions) + ",Count\n")
        for comb in product(*values):
            paper_set = [set(value2ppaer[value]) for value in comb]
            count = len(set.intersection(*paper_set))
            if count >= min_support:
                f.write(",".join(comb) + ",{:d}\n".format(count))