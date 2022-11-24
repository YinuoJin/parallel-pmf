import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch
from sklearn.manifold import TSNE
#from mayavi import mlab

# Reference:
# https://stackoverflow.com/questions/58903383/fancyarrowpatch-in-3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, **kwargs):
        FancyArrowPatch.__init__(self, posA=(0, 0), posB=(0, 0), **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def loss(df_loss, outdir, display=True):
    sns.set_theme()
    plt.figure(figsize=(10, 8))
    plt.suptitle('Log-likelihood')
    sns.lineplot(data=df_loss, x='Epoch', y='Loss', marker="o")
    if display:
        plt.show()


def _norm_vector(vec):
    return vec / np.linalg.norm(vec)
"""
#def _set_mlab_axis(scale, bg_color, axis_color):
    len_axis = 2*scale
    #mlab.figure(bgcolor=bg_color, size=(int(scale * 0.8), int(scale * 0.6)))
    #mlab.plot3d([-len_axis, len_axis], [0, 0], [0, 0], color=axis_color, tube_radius=10.)
    #mlab.plot3d([0, 0], [-len_axis, len_axis], [0, 0], color=axis_color, tube_radius=10.)
    #mlab.plot3d([0, 0], [0, 0], [-len_axis, len_axis], color=axis_color, tube_radius=10.)

    #mlab.text3d(len_axis + 50, -50, +50, 'Attr_1', color=axis_color, scale=100.)
    #mlab.text3d(0, len_axis + 50, +50, 'Attr_2', color=axis_color, scale=100.)
    #mlab.text3d(0, -50, len_axis + 50, 'Attr_3', color=axis_color, scale=100.)
"""

def tsne(df):
    sns.set_theme()
    fig = plt.figure(figsize=(80, 60))

    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(df['attr_1'], df['attr_2'], df['attr_3'],
               c=df['cluster'], s=10)
    ax.set_title('t-SNE 3D', fontsize=30)

    # legend
    n_clusters = len(np.unique(df['cluster']))
    cmap = sns.color_palette("rocket", n_colors=n_clusters)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in cmap]
    ax.legend(markers, np.arange(n_clusters), fontsize='xx-large')

    plt.show()


#@mlab.show
def tsne_interactive(df, scale=1000, ratio=100):
    labels = np.unique(df['cluster'])
    n_clusters = len(labels)

    cmap = sns.color_palette('viridis', n_colors=n_clusters)
    white = colors.to_rgb('white')
    black = colors.to_rgb('black')

    # set axis
    #_set_mlab_axis(scale*2, bg_color=white, axis_color=black)
    # scatter plot
    for label, c in zip(labels, cmap):
        df_label = df[df['cluster'] == label]
        #mlab.points3d(df_label['attr_1'] * ratio, df_label['attr_2'] * ratio, df_label['attr_3'] * ratio, color=c, scale_factor=30)


def arrow(vectors):
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    cmap = {0: 'r', 1: 'g', 2: 'b'}

    # plot axes
    x_axis = Arrow3D([-2, 2], [0, 0], [0, 0], lw=1, linestyle='dotted', arrowstyle="-", color='k')
    y_axis = Arrow3D([0, 0], [-2, 2], [0, 0], lw=1, linestyle='dotted', arrowstyle="-", color='k')
    z_axis = Arrow3D([0, 0], [0, 0], [-2, 2], lw=1, linestyle='dotted', arrowstyle="-", color='k')
    ax.add_artist(x_axis)
    ax.add_artist(y_axis)
    ax.add_artist(z_axis)

    # plot vectors
    for vec in vectors:
        vec = _norm_vector(vec)
        g = Arrow3D([0, vec[0]], [0, vec[1]], [0, vec[2]],
                    mutation_scale=10, lw=1, arrowstyle="-|>", color=cmap[np.argmax(np.abs(vec))])
        ax.add_artist(g)

    ax.set_xlabel('attr_1', fontsize=20, labelpad=20)
    ax.set_ylabel('attr_2', fontsize=20, labelpad=20)
    ax.set_zlabel('attr_3', fontsize=20, labelpad=20)

    ax.tick_params(labelsize=10)

    ax.axes.set_xlim3d(-1, 1)
    ax.axes.set_ylim3d(-1, 1)
    ax.axes.set_zlim3d(-1, 1)
    ax.set_title('Visualization of latent beta vectors (items)', fontsize=30)

    plt.show()

def arrow_joint(vec_users, vec_items):
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')

    # plot axes
    x_axis = Arrow3D([-2, 2], [0, 0], [0, 0], lw=1, linestyle='dotted', arrowstyle="-", color='k')
    y_axis = Arrow3D([0, 0], [-2, 2], [0, 0], lw=1, linestyle='dotted', arrowstyle="-", color='k')
    z_axis = Arrow3D([0, 0], [0, 0], [-2, 2], lw=1, linestyle='dotted', arrowstyle="-", color='k')
    ax.add_artist(x_axis)
    ax.add_artist(y_axis)
    ax.add_artist(z_axis)

    # plot vectors
    for vec in vec_users:
        vec = _norm_vector(vec)
        g = Arrow3D([0, vec[0]], [0, vec[1]], [0, vec[2]],
                    mutation_scale=10, lw=1, arrowstyle="-|>", color='r')
        ax.add_artist(g)

    for vec in vec_items:
        vec = _norm_vector(vec)
        g = Arrow3D([0, vec[0]], [0, vec[1]], [0, vec[2]],
                    mutation_scale=10, lw=1, arrowstyle="-|>", color='g')
        ax.add_artist(g)

    ax.axes.set_xlim3d(-1, 1)
    ax.axes.set_ylim3d(-1, 1)
    ax.axes.set_zlim3d(-1, 1)
    ax.set_title('Joint visualization of latent theta (user) & beta (item) vectors', fontsize=30)

    plt.show()

#@mlab.show
def arrow_interactive(vectors, names, show_title=False, is_similar=False, scale=1000):
    cmap = {0: 'r', 1: 'g', 2: 'b'}
    black = colors.to_rgb('black')
    white = colors.to_rgb('white')

    # set axis
    #_set_mlab_axis(scale, bg_color=white, axis_color=black)

    if is_similar:
        title_x, title_y, title_z = -500, scale*2+200, scale*2+200

    for vec, name in zip(vectors, names):
        vec = _norm_vector(vec)
        color = colors.to_rgb(cmap[np.argmax(np.abs(vec))])
        vec = scale * np.array(vec)
        #mlab.plot3d([0, vec[0]], [0, vec[1]], [0, vec[2]], color=color, tube_radius=5.)
        if show_title:
            if is_similar:
                #mlab.text3d(title_x, title_y, title_z, name, color=black, scale=30)
                title_z -= 100
            else:
                pass
                #mlab.text3d(1.05*vec[0] , 1.05*vec[1]*1.1, 1.05*vec[2], name, color=black, scale=30)


#@mlab.show
def arrow_joint_interactive(vec_users, vec_items, names, show_title=False, scale=1000):
    black = colors.to_rgb('black')
    white = colors.to_rgb('white')
    red = colors.to_rgb('red')
    green = colors.to_rgb('green')

    # set axis
    #_set_mlab_axis(scale, bg_color=white, axis_color=black)

    # plot user vectors
    for vec in vec_users:
        vec = _norm_vector(vec)
        vec = scale * np.array(vec)
        #mlab.plot3d([0, vec[0]], [0, vec[1]], [0, vec[2]], color=red, tube_radius=5.)

    title_x, title_y, title_z = -500, scale*2+200, scale*2+200

    # plot item vectors
    for vec, name in zip(vec_items, names):
        vec = _norm_vector(vec)
        vec = scale * np.array(vec)
        #mlab.plot3d([0, vec[0]], [0, vec[1]], [0, vec[2]], color=green, tube_radius=5.)

        if show_title:
            #mlab.text3d(title_x, title_y, title_z, name, color=black, scale=30)
            title_z -= 50
