import numpy as np
import yaml
from os import getcwd
from policy import policy_gps

path = getcwd()

with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

seed = params['seed']
size = params['size']
n_t = params['n_t']
n_h = params['n_h']
sub_x = params['sub_x']
sub_y = params['sub_y']
n_steps = params['n_steps']
gps_cost = params['gps_cost']
uncert_i = params['uncert_i']
uncert_inc = params['uncert_inc']

pos, pos_est, i, no_gps = policy_gps(path, seed, sub_x, sub_y, n_steps, n_t, uncert_i, n_h, size, gps_cost, uncert_inc)

np.savez(path + '/data/policy/policy_gps_' + str(seed) + '.npz', pos = pos[:(i+1)*n_t], pos_est = pos_est[:i+1], no_gps = no_gps[1:])
