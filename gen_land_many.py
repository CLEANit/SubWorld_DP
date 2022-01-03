import numpy as np
import multiprocessing
from itertools import repeat
import yaml
from os import getcwd
from islands import save_chart

# Get the current working directory to use for file loading and saving
path = getcwd()

# Load the current parameters set from params.yaml
with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

n_cpu = params['n_cpu']
maps_i = params['maps_i']
maps_f = params['maps_f']

dim = params['dim']
min_islands = params['min_islands']
max_islands = params['max_islands']
size = params['size']
min_height = params['min_height']
max_height = params['max_height']
x_decay_min = params['x_decay_min']
x_decay_max = params['y_decay_max']
y_decay_min = params['y_decay_min']
y_decay_max = params['y_decay_max']
max_cur = params['max_cur']

# Function to display initialization of thread
def start():
    print('Starting', multiprocessing.current_process().name)

if __name__ == "__main__":
    # Initialize parallel threads
    pool = multiprocessing.Pool(processes = n_cpu, initializer=start)
    # Create set of seeds to use for loading charts
    seeds = list(np.arange(maps_i, maps_f, 1, dtype=np.int32))
    # Randomly determine the number of islands to generate for each chart
    n_islands = np.clip(np.random.normal(0.6*max_islands, 0.25*max_islands, maps_f - maps_i), min_islands, max_islands).astype(np.int32)
    # Generate and save many charts using the save_chart function
    pool.starmap(save_chart, zip(seeds, n_islands, repeat(size), repeat(x_decay_min), repeat(x_decay_max), repeat(y_decay_min), repeat(y_decay_max), repeat(min_height), repeat(max_height), repeat(dim), repeat(max_cur), repeat(path)))
    pool.close()
    pool.join()
