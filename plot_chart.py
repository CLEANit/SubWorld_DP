import numpy as np
import matplotlib.pyplot as plt
import cmocean
import yaml
from os import getcwd

path = getcwd()

with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

seed = params['seed']
cur_scale = params['cur_scale']

data = np.load(path + '/data/charts/charts_' + str(seed) + '.npz', allow_pickle=True)
chart = data['chart']
dim = chart.shape[0]
water_p = np.meshgrid(np.arange(0, dim, 3), np.arange(0, dim, 3))
current = cur_scale*data['water_c']
water_c = current[0::3, 0::3]

_plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(8, 8))
_plot_lines = []
mappable = _plot_axs.pcolormesh(chart.transpose(), vmin=-1.3, vmax=1.3, cmap=cmocean.cm.delta)
quiver = _plot_axs.quiver(water_p[0]+0.5, water_p[1]+0.5, water_c[:, :, 0].transpose(), water_c[:, :, 1].transpose(), scale=20, alpha=0.5, color='w')
_plot_axs.set_xticks([])
_plot_axs.set_yticks([])
_plot_axs.set_xlim([0, dim])
_plot_axs.set_ylim([0, dim])
plt.savefig(path + '/PDFs/charts/chart_' + str(seed) + '.pdf')
