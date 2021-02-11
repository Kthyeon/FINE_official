import numpy as np
import itertools

def dict_product(d):
    keys = d.keys()
    vals = d.values()
    prod_values = list(itertools.product(*vals))
    all_dicts = map(lambda x: dict(zip(keys, x)), prod_values)
    return all_dicts

def iwt(start, end, interval, trials=1):
    return list(np.arange(start, end, interval))*trials