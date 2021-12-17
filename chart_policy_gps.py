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
cur_cost = params['cur_cost']
uncert_pos = params['uncert_pos']
uncert_cur = params['uncert_cur']
max_cur = params['max_cur']

pos, pos_est, i, no_gps, no_cur = policy_gps(path, seed, sub_x, sub_y, n_steps, n_t, uncert_pos, n_h, size, gps_cost, cur_cost, uncert_cur, max_cur)

np.savez(path + '/data/policy/policy_gps_' + str(seed) + '.npz', pos = pos[:(i+1)*n_t], pos_est = pos_est[:i+1], no_gps = no_gps[1:], no_cur = no_cur[1:])
