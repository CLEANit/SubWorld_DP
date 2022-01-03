import numpy as np
import yaml
from os import getcwd
from value import gen_value

# Get the current working directory to use for file loading and saving
path = getcwd()

# Load the current parameters set from params.yaml
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

# Generate a value function using the gen_value function
chart_value, rel_chart_value = gen_value(path, seed, dim, target_x, target_y, size, tol, n_h, n_t, discount)

# Save the value function previously generated
np.savez(path + '/data/value/value_' + str(seed) + '.npz', value = chart_value, rel_value = rel_chart_value, discount = discount)
