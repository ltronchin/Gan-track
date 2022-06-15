#!/usr/bin/

#  Libraries
print('Import the library')
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import json
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 16

data_dir = 'reports/claro_retrospettivo/stylegan2-ada/00000-stylegan2--gpus2-batch32-gamma0.4096/metric-fid50k_full_training.jsonl'
report_dir = 'reports/claro_retrospettivo/stylegan2-ada/00000-stylegan2--gpus2-batch32-gamma0.4096/'

metric_name = 'fid50k_full'
snap_interval = 5
total_kimg = 5000

data = [json.loads(line) for line in open(data_dir, 'r')]

metric = [data[snap_idx]['results'][metric_name] for snap_idx in range(len(data))]
# ticks = np.arange(start=0, stop=len(metric)*snap_interval, step=5)
kimg = np.arange(start=0, stop=len(metric) * snap_interval, step=5) *4

plt.plot(kimg, metric, linewidth=3, color='blue')
plt.xlabel('kimg')
plt.ylabel(metric_name)
plt.savefig(os.path.join(report_dir, f"{metric_name}_vs_kimg.png"), dpi=400, format='png')
plt.show()


