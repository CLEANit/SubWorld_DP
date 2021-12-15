import numpy as np
import yaml
from os import getcwd
from islands import save_chart

path = getcwd()

with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

seed = params['seed']
dim = params['dim']
n_islands = params['seed']
if n_islands == None:
    max_islands = params['max_islands']
    n_islands = np.clip(np.random.normal(0.6*max_islands, 0.25*max_islands), 0, max_islands).astype(np.int32)

x_size = params['x_size']
y_size = params['y_size']
min_height = params['min_height']
max_height = params['max_height']
x_decay_min = params['x_decay_min']
x_decay_max = params['y_decay_max']
y_decay_min = params['y_decay_min']
y_decay_max = params['y_decay_max']
max_cur = params['max_cur']

save_chart(seed, n_islands, x_size, y_size, x_decay_min, x_decay_max, y_decay_min, y_decay_max, min_height, max_height, dim, max_cur, path)