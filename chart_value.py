import numpy as np
import yaml
from os import getcwd
from value import gen_value

path = getcwd()

with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

seed = params['seed']
dim = params['dim']
size = params['size']
target_x = params['target_x']
target_y = params['target_y']
discount = params['discount']
tol = 10 ** params['tol']
n_t = params['n_t']
n_h = params['n_h']

chart_value, steps = gen_value(path, seed, dim, target_x, target_y, size, tol, n_h, n_t, discount)

np.savez(path + '/data/value/value_' + str(seed) + '.npz', value = chart_value, steps = steps, discount = discount)
