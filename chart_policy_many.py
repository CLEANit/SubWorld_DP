import numpy as np
import multiprocessing
from itertools import repeat
import yaml
from os import getcwd
from policy import policy_gps

path = getcwd()

with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

n_cpu = params['n_cpu']
maps_i = params['maps_i']
maps_f = params['maps_f']
size = params['size']
n_t = params['n_t']
n_h = params['n_h']
n_steps = params['n_steps']
gps_cost = params['gps_cost']
cur_cost = params['cur_cost']
uncert_pos = params['uncert_pos']
uncert_cur = params['uncert_cur']
max_cur = params['max_cur']

def start():
    print('Starting', multiprocessing.current_process().name)

def save_policy(seed, path, n_steps, n_t, uncert_pos, n_h, size, gps_cost, cur_cost, uncert_cur, max_cur):
    pos, pos_est, i, no_gps, no_cur, status = policy_gps(path, seed, None, None, n_steps, n_t, uncert_pos, n_h, size, gps_cost, cur_cost, uncert_cur, max_cur)
    np.savez(path + '/data/policy/policy_gps_' + str(seed) + '.npz', pos = pos[:(i+1)*n_t], pos_est = pos_est[:i+1], no_gps = no_gps[1:], no_cur = no_cur[1:], status = status)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = n_cpu, initializer=start)
    seeds = list(np.arange(maps_i, maps_f, 1, dtype=np.int32))
    pool.starmap(save_policy, zip(seeds, repeat(path), repeat(n_steps), repeat(n_t), repeat(uncert_pos), repeat(n_h), repeat(size), repeat(gps_cost), repeat(cur_cost), repeat(uncert_cur), repeat(max_cur)))
    pool.close()
    pool.join()
