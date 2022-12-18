""" The :mod:`prda.utility` contains ``what you might use`` when handling your data.
"""

import random
import numpy as np
import pandas as pd
from scipy.stats import anderson, kstest

__all__ = ['nlargest_dict', 'get_distribution', 'classify_indices', 'gussian_test', 'choice_weighted', 'classify_indices']

def nlargest_dict(dict_: dict, n: int)-> list:
    """This function returns `n` largest elements from `dict_`

    Parameters
    ----------
    dict_ : dict
        _description_
    n : int
        How many elements to return

    Returns
    -------
    list
        A list containing `n` largest elements of the dict.
    """
    
    dict_sorted = sorted(dict_.items(),key=lambda x:x[1], reverse=True)
    return dict_sorted[:n]


def choice_weighted(data_weighted: dict, n=1, dupliacted=False):
    """
    Random choice `n` unique elements from `data_weighted`'s keys according to weight of data.

    Parameters
    ----------
    data_weighted : dict
        This dict's values have to be intigers
        
    n : int
        number of elements sampled.

    duplicated : bool
        whether allows duplicated elements.
    
    Returns
    -------
        if n equals to 1, then output a single element, else, return list.
    """
    population = []
    for k, weight in data_weighted.items():
        population.extend([k for _ in range(weight)])
    if n == 1:
        return random.sample(population, 1)[0]
    else:
        results = random.sample(population, n)
        if not dupliacted:
            n_duplicated = n - len(set(results))
            if n_duplicated:
                while len(set(results)) != n:
                    results.append(random.choice(population))
                results = set(results)
        return results
        

def gussian_test(data, type='anderson'):
    """
    Test whether 1-D data fit gussian distribution.

    Parameters
    ----------
    data : _type_
        _description_
    """
    if type == 'anderson':
        return anderson(data)
    if type == 'kstest':
        return kstest(data, cdf='norm')


def classify_indices(sequence)-> dict:
    classifications = {gene: [] for gene in set(sequence)}
    for i in range(len(sequence)):
        classifications[sequence[i]].append(i)
    return classifications

def get_distribution(popul, popul_type: str = 'continuous', n_intervals: int=None)-> dict:
    """Calculate distribution of a given population `popul`.

    Parameters
    ----------
    popul : array_like
        population container
    popul_type : str
        population type indicator, whether `continuous` population or `discrete`
    n_intervals : int, optional
        _description_, by default None

    Returns
    -------
    dict
        distribution dict,
        if `popul_type` is `continous`, then keys of dict is the bottom of each intervals.
    """
    if popul_type.lower() == 'continuous':
        min_val = min(popul)
        max_val = max(popul)
        keys = [min_val + i*(max_val - min_val)/n_intervals for i in range(n_intervals)]
        distribution = {k: 0 for k in keys}
        for sample in popul:
            for i in range(len(keys)):
                if sample < keys[i]:
                    distribution[keys[i-1]] += 1
                    break
    else:
        raise KeyError('Sorry, data type: ', popul_type, 'not supported right now.')
    return distribution


