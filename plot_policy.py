import numpy as np
import matplotlib.pyplot as plt
import cmocean
import yaml
from os import getcwd

path = getcwd()

with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

seed = params['seed']
n_t = params['n_t']

data = np.load(path + '/data/charts/charts_' + str(seed) + '.npz', allow_pickle=True)
chart = data['chart']
dim = chart.shape[0]
water_p = np.meshgrid(np.arange(0, dim, 3), np.arange(0, dim, 3))
current = data['water_c']
water_c = current[0::3, 0::3]
data1 = np.load(path + '/data/policy/policy_gps_' + str(seed) + '.npz')
pos = data1['pos']
pos_est = data1['pos_est']
no_gps = data1['no_gps']
data2 = np.load(path + '/data/value/value_' + str(seed) + '.npz', allow_pickle=True)
value = data2['value']

if len(no_gps) > 0:
    no_gps_line = [no_gps[0]]
    for i in range(1, len(no_gps)):
        if no_gps[i-1] not in no_gps_line:
            no_gps_line.append(no_gps[i-1])

        no_gps_line.append(no_gps[i])

_plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(8, 8))
_plot_lines = []
mappable = _plot_axs.pcolormesh(chart.transpose(), vmin=-1.3, vmax=1.3, cmap=cmocean.cm.delta)
quiver = _plot_axs.quiver(water_p[0]+0.5, water_p[1]+0.5, water_c[:, :, 0].transpose(), water_c[:, :, 1].transpose(), scale=20, alpha=0.5, color='w')
_plot_axs.plot(pos[:, 0]*dim, pos[:, 1]*dim, c='k', zorder=1)
_plot_axs.scatter(pos[:, 0][0::n_t]*dim, pos[:, 1][0::n_t]*dim, c='k', zorder=1)
_plot_axs.scatter(pos[-1, 0]*dim, pos[-1, 1]*dim, c='k', zorder=1)
if len(no_gps) > 0:
    _plot_axs.plot(pos_est[no_gps_line[1:], 0]*dim, pos_est[no_gps_line[1:], 1]*dim, c='g', zorder=10)
    _plot_axs.scatter(pos_est[no_gps[1:], 0]*dim, pos_est[no_gps[1:], 1]*dim, c='g', s=15, zorder=10)
_plot_axs.scatter([pos[0, 0]*dim], [pos[0, 1]*dim], c='w', zorder=1)
_plot_axs.set_xticks([])
_plot_axs.set_yticks([])
_plot_axs.set_xlim([0, dim])
_plot_axs.set_ylim([0, dim])
plt.savefig(path + '/PDFs/policy/policy_gps_' + str(seed) + '.pdf')

_plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(8, 8))
_plot_lines = []
mappable = _plot_axs.pcolormesh(value.transpose(), vmin=-1.0, vmax=1.0, cmap='RdYlGn')
quiver = _plot_axs.quiver(water_p[0]+0.5, water_p[1]+0.5, water_c[:, :, 0].transpose(), water_c[:, :, 1].transpose(), scale=20, alpha=0.5, color='w')
_plot_axs.plot(pos[:, 0]*dim, pos[:, 1]*dim, c='k', zorder=1)
_plot_axs.scatter(pos[:, 0][0::n_t]*dim, pos[:, 1][0::n_t]*dim, c='k', zorder=1)
_plot_axs.scatter(pos[-1, 0]*dim, pos[-1, 1]*dim, c='k', zorder=1)
if len(no_gps) > 0:
    _plot_axs.plot(pos_est[no_gps_line[1:], 0]*dim, pos_est[no_gps_line[1:], 1]*dim, c='g', zorder=10)
    _plot_axs.scatter(pos_est[no_gps[1:], 0]*dim, pos_est[no_gps[1:], 1]*dim, c='g', zorder=10)
_plot_axs.scatter([pos[0, 0]*dim], [pos[0, 1]*dim], c='w', zorder=1)
_plot_axs.set_xticks([])
_plot_axs.set_yticks([])
_plot_axs.set_xlim([0, dim])
_plot_axs.set_ylim([0, dim])
plt.savefig(path + '/PDFs/policy/policy_value_gps_' + str(seed) + '.pdf')