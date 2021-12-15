import numpy as np
import multiprocessing
from itertools import repeat
import yaml
from os import getcwd
from islands import save_chart

path = getcwd()

with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

n_cpu = params['n_cpu']
maps_i = params['maps_i']
maps_f = params['maps_f']

dim = params['dim']
max_islands = params['max_islands']
x_size = params['x_size']
y_size = params['y_size']
min_height = params['min_height']
max_height = params['max_height']
x_decay_min = params['x_decay_min']
x_decay_max = params['y_decay_max']
y_decay_min = params['y_decay_min']
y_decay_max = params['y_decay_max']
max_cur = params['max_cur']

def start():
    print('Starting', multiprocessing.current_process().name)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = n_cpu, initializer=start)
    seeds = list(np.arange(maps_i, maps_f, 1, dtype=np.int32))
    n_islands = np.clip(np.random.normal(0.6*max_islands, 0.25*max_islands, maps_f - maps_i), 0, max_islands).astype(np.int32)
    pool.starmap(save_chart, zip(seeds, n_islands, repeat(x_size), repeat(y_size), repeat(x_decay_min), repeat(x_decay_max), repeat(y_decay_min), repeat(y_decay_max), repeat(min_height), repeat(max_height), repeat(dim), repeat(max_cur), repeat(path)))
    pool.close()
    pool.join()
