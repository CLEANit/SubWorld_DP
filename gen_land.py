import numpy as np
import yaml
from os import getcwd
from islands import save_chart

# Get the current working directory to use for file loading and saving
path = getcwd()

# Load the current parameters set from params.yaml
with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

seed = params['seed']
dim = params['dim']
n_islands = params['n_islands']
if n_islands == 'None':
    # Randomly determine the number of islands to generate
    min_islands = params['min_islands']
    max_islands = params['max_islands']
    n_islands = np.clip(np.random.normal(0.6*max_islands, 0.25*max_islands), min_islands, max_islands).astype(np.int32)

size = params['size']
min_height = params['min_height']
max_height = params['max_height']
x_decay_min = params['x_decay_min']
x_decay_max = params['y_decay_max']
y_decay_min = params['y_decay_min']
y_decay_max = params['y_decay_max']
max_cur = params['max_cur']

# Generate and save a chart using the save_chart function
save_chart(seed, n_islands, size, x_decay_min, x_decay_max, y_decay_min, y_decay_max, min_height, max_height, dim, max_cur, path)