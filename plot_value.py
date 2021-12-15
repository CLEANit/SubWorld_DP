import numpy as np
import matplotlib.pyplot as plt

seed = 2525

data = np.load('./SubWorld_DP/data/value/value_' + str(seed) + '.npz', allow_pickle=True)
value = data['value']
steps = data['steps']
dim = value.shape[0]

_plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(8, 8))
_plot_lines = []
mappable = _plot_axs.pcolormesh(value.transpose(), vmin=0.5, vmax=1.0, cmap='RdYlGn')
_plot_axs.set_xticks([])
_plot_axs.set_yticks([])
_plot_axs.set_xlim([0, dim])
_plot_axs.set_ylim([0, dim])
plt.savefig('./SubWorld_DP/PDFs/value/value_' + str(seed) + '.pdf')

_plot_fig, _plot_axs = plt.subplots(1, 1, figsize=(8, 8))
_plot_lines = []
mappable = _plot_axs.pcolormesh(steps.transpose(), vmin=0, vmax=steps.max(), cmap='RdYlGn')
_plot_axs.set_xticks([])
_plot_axs.set_yticks([])
_plot_axs.set_xlim([0, dim])
_plot_axs.set_ylim([0, dim])
plt.savefig('./SubWorld_DP/PDFs/value/steps_' + str(seed) + '.pdf')
