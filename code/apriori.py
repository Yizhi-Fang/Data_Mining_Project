#!usr/bin/env python3

"""Find frequent patterns in Problem or Method."""

from os.path import join
from itertools import combinations, permutations, chain


def initialize(transactions, value2paper,
               values=["neural network", "deep learning"]):
    """Get transaction list, 1-itemset from transactions."""
    paper_list = [paper for value in values for paper in value2paper[value]]
    transaction_list = []
    itemset = set()
    for paper in set(paper_list):
        transaction_list.append(frozenset(transactions[paper]))
        for val in transactions[paper]:
            itemset.add(frozenset([val]))
    return transaction_list, itemset


def freq_itemset(transaction_list, itemset, min_support):
    """Generate frequent itemset from candidate itemset."""
    f_itemset = []
    for item in itemset:
        support = 0
        for transaction in transaction_list:
            if item.issubset(transaction):
                support += 1
        support /= len(transaction_list)
        if support >= min_support:
            f_itemset.append((item, support))
    return dict(f_itemset)


def joinset(f_itemset, k):
    """Generate k-itemset from (k-1)-itemset."""
    itemset = []
    # A set of frequent itemset because f_itemset is a dictionary.
    f_itemsets = []
    for i in f_itemset:
        f_itemsets += list(i)
    f_itemsets = frozenset(f_itemsets)
    for i in f_itemset:
        for j in f_itemset:
            temp = i.union(j)
            if len(temp) == k:
                for item in combinations(temp, k-1):
                    if not frozenset(item).issubset(f_itemsets):
                        break
                else:  # Executed if the for ended normally.
                    itemset.append(temp)
                # itemset.append(temp)
    return set(itemset)


def apriori_model(transaction_list, itemset, min_support):
    freq_pattern = dict()
    k = 1
    while True:
        f_itemset = freq_itemset(transaction_list, itemset, min_support)
        if not f_itemset:
            break
        freq_pattern.update(f_itemset)
        k += 1
        itemset = joinset(f_itemset, k)
    return freq_pattern


def sort_pattern(f_pattern):
    """Sort frequent patterns by support (descending) then pattern (ascending).
    """
    # Change the key in f_pattern (frozenset) to list to be ordered
    # alphabetically.
    f_pattern2 = []
    for pattern, support in f_pattern.items():
        pattern_list = sorted(list(pattern))
        f_pattern2.append((pattern_list, support))
    s_pattern = sorted(f_pattern2, key=lambda x: (-x[1], x[0]))
    return s_pattern


def write_patterns(path2output, sorted_pattern):
    output_text = ["Pattern,Support"]
    for item in sorted_pattern:
        output_text.append("{},{:f}".format(" | ".join(item[0]), item[1]))
    pattern_file = "Freq_Patterns.csv"
    with open(join(path2output, pattern_file), "w") as f:
        f.write("\n".join(output_text))


def compute_confidence(s12, s1):
    """Compute confidence for 1 -> 2."""
    return s12 / s1


def compute_kulc(s12, s1, s2):
    """Compute Kulczynski."""
    return s12 * (1/s1 + 1/s2) / 2


def compute_ir(s12, s1, s2):
    """Compute Imbalance Ratio."""
    return abs(s1 - s2) / (s1 + s2 - s12)


def rule_candidate(pattern):
    """Find all association rules candidates."""
    candidate = []
    comb = chain(*map(lambda x: combinations(pattern, x),
                      range(1, len(pattern))))
    for p1, p2 in permutations(comb, 2):
        p1_set = frozenset(list(p1))
        p2_set = frozenset(list(p2))
        if not p1_set.intersection(p2_set):
            candidate.append((p1_set, p2_set))
    return candidate


def evaluate_apriori(path2output, freq_pattern, min_confidence):
    """Compute Kulc and Imbalance Ratio for 2-itemset."""
    output_text = ["Association Rule,min_support,min_confidence,Kulc,IR"]
    multi_pattern = dict(filter(lambda x: len(x[0]) > 1, freq_pattern.items()))
    for pattern in multi_pattern:
        s12 = freq_pattern[pattern]
        candidate = rule_candidate(pattern)
        for rule in candidate:
            p1, p2 = rule
            s1 = freq_pattern[p1]
            s2 = freq_pattern[p2]
            conf = compute_confidence(s12, s1)
            if conf >= min_confidence:
                p1_str = " | ".join(list(p1))
                p2_str = " | ".join(list(p2))
                kulc = compute_kulc(s12, s1, s2)
                ir = compute_ir(s12, s1, s2)
                output_text.append("{} => {},{:f},{:f},"
                                   "{:f},{:f}".format(p1_str, p2_str, s12,
                                                      conf, kulc, ir))
    eval_file = "Associ_Rules.csv"
    with open(join(path2output, eval_file), "w") as f:
        f.write("\n".join(output_text))


def run_apriori(transactions, value2paper, min_support, min_confidence,
                path2output):
    transaction_list, itemset = initialize(transactions, value2paper)
    freq_pattern = apriori_model(transaction_list, itemset, min_support)
    sorted_pattern = sort_pattern(freq_pattern)
    write_patterns(path2output, sorted_pattern)
    evaluate_apriori(path2output, freq_pattern, min_confidence)