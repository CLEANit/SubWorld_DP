import numpy as np
import matplotlib.pyplot as plt
import yaml
from os import getcwd

path = getcwd()

with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

seed = params['seed']

data = np.load(path + '/data/value/value_' + str(seed) + '.npz', allow_pickle=True)
value = data['value']
rel_value = data['rel_value']
dim = value.shape[0]

_plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(8, 8))
_plot_lines = []
mappable = _plot_axs.pcolormesh(value.transpose(), vmin=-1, vmax=1.0, cmap='RdYlGn')
_plot_axs.set_xticks([])
_plot_axs.set_yticks([])
_plot_axs.set_xlim([0, dim])
_plot_axs.set_ylim([0, dim])
plt.savefig(path + '/PDFs/value/value_' + str(seed) + '.pdf')

_plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(8, 8))
_plot_lines = []
mappable = _plot_axs.pcolormesh(rel_value.transpose(), vmin=0.9, vmax=1.0, cmap='RdYlGn')
_plot_axs.set_xticks([])
_plot_axs.set_yticks([])
_plot_axs.set_xlim([0, dim])
_plot_axs.set_ylim([0, dim])
plt.savefig(path + '/PDFs/value/rel_value_' + str(seed) + '.pdf')
