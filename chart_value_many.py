import numpy as np
import multiprocessing
from itertools import repeat
import yaml
from os import getcwd
from value import gen_value

# Get the current working directory to use for file loading and saving
path = getcwd()

# Load the current parameters set from params.yaml
with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

n_cpu = params['n_cpu']
maps_i = params['maps_i']
maps_f = params['maps_f']
dim = params['dim']
size = params['size']
discount = params['discount']
tol = 10 ** params['tol']
n_t = params['n_t']
n_h = params['n_h']

# Function to display initialization of thread
def start():
    print('Starting', multiprocessing.current_process().name)

# Function to generate and save a value function using the gen_value function
def save_value(seed, path, dim, size, tol, n_h, n_t, discount):
    chart_value, rel_chart_value = gen_value(path, seed, dim, None, None, size, tol, n_h, n_t, discount)
    np.savez(path + '/data/value/value_' + str(seed) + '.npz', value = chart_value, rel_value = rel_chart_value, discount = discount)

if __name__ == "__main__":
    # Initialize parallel threads
    pool = multiprocessing.Pool(processes = n_cpu, initializer=start)
    # Create set of seeds to use for loading charts
    seeds = list(np.arange(maps_i, maps_f, 1, dtype=np.int32))
    # Generate and save many value functions using the gen_value function
    pool.starmap(save_value, zip(seeds, repeat(path), repeat(dim), repeat(size), repeat(tol), repeat(n_h), repeat(n_t), repeat(discount)))
    pool.close()
    pool.join()
