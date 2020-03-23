# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:15:20 2020

@author: alexc
"""

import matplotlib.pyplot as plt
import numpy as np
from .confidence import map_likelihood_ratio


plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = (6.4, 4.8)
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.constrained_layout.use'] = True

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['HelveticaNeue']
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"


def plt_format(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(axis='both', labelsize=11)

    xt = ax.get_xticks()
    yt = ax.get_yticks()


    for y in yt[1:-1]:
        ax.plot([xt[0],xt[-1]], [y] * 2, "--", lw=0.5, color="black", alpha=0.3)

    for x in xt[1:-1]:
        ax.plot([x]*2, [yt[0],yt[-1]], "--", lw=0.5, color="black", alpha=0.3)


    ax.tick_params(axis="both", which="both", bottom=False, top=False,
                labelbottom=True, left=False, right=False, labelleft=True)
    
def plot_probability(model, include_data,
                     xlabel,
                     ylabel,
                     alpha,
                     save_dst,
                     show,
                     **kwargs
                     ):
    
    CI_level = kwargs.get('CI_level', [.5, .8, .9, .95])
    
    pred = model.predict_probability(**kwargs)
    
    fig, ax = plt.subplots()
    if include_data:
        ax.scatter(model.X[model.Y==0], model.Y[model.Y==0], 
                   zorder=10, alpha=alpha, color='#BF0A30')
        ax.scatter(model.X[model.Y==1], model.Y[model.Y==1], 
                   zorder=10, alpha=alpha, color='#3F704D')
    ax.plot(pred[0], pred[1], '-k', zorder=8, label='estimate', linewidth=2.5)
    if len(pred) > 2:
        cm = plt.get_cmap('magma')
        n_levels = pred[2].shape[1]
        for ix in range(n_levels):
            ax.plot(pred[0], pred[2][:, ix], color=cm((ix+1)/(n_levels+1)), 
                    label='{}%'.format(CI_level[ix]*100))
            ax.plot(pred[0], pred[3][:, ix], color=cm((ix+1)/(n_levels+1)))
            
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    ax.set_xlim(np.min(pred[0]), np.max(pred[0]))
    ax.legend(loc='best')
    ax.set_ylim(-.05, 1.05)
    plt_format(ax)
    
    if save_dst is not None:
        plt.savefig(save_dst)

    if show:
        plt.show()
        
def plot_confidence_region(model, limits, n, CI_levels, save_dst, show):
    # note, this is a little tricky, the keyword args levels, max_iter, 
    # and counter will not be used. counter should never be touched.
    # it is used for 'stateful' recursion
    
    X, Y, Z = map_likelihood_ratio(model, limits, n, levels=None, 
                                   max_iter=2, field=True)
    Z *= 100
    
    fig, ax = plt.subplots()
    
    if isinstance(CI_levels, int):
        CS = ax.contourf(X, Y, Z,
                         levels=CI_levels,
                         cmap='bone', vmin=0, vmax=100)
        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel('% Confidence Region')
    else:
        CS = ax.contour(X, Y, Z, 
                        levels=np.array(CI_levels)*100, 
                        cmap='magma', vmin=0, vmax=100)
        ax.grid(which='major')
        ax.clabel(CS, inline=1, fontsize=10)
    
    t1, t2 = model.est_names(True)
    ax.set_xlabel(t1)
    ax.set_ylabel(t2)

    if save_dst is not None:
        plt.savefig(save_dst)
        
    if show:
        plt.show()

