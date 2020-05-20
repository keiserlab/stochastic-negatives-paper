# coding: utf-8
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
        
nn_paths = get_ipython().getoutput(u'find . -name model_at_epoch_0.npz ')
nn_paths = [os.path.dirname(p) for p in nn_paths]
    
        
for nnp in nn_paths:
    lossf = sorted(glob.glob(os.path.join(nnp, '*loss.csv')))
    lossf = lossf[-1:] + lossf[:-1]
    losses = [np.loadtxt(lf) for lf in lossf]
    f, ax = plt.subplots()
    for i, l in enumerate(losses):
        ax.plot(l, label=os.path.split(lossf[i])[-1][:-4] )
    ax.set_ylim([0., 3.])
    ax.set_title(nnp[2:])
    plt.legend()
    f.savefig(os.path.join(nnp,'all_losses.png'), dpi=150, bbox_inches='tight')
    plt.close(f)
        
