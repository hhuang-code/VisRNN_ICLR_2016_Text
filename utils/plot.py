import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

import pdb

def plot_gate(input_gate, forget_gate, output_gate):
    # prepare data
    (input_left_s, input_right_s), (forget_left_s, forget_right_s), (output_left_s, output_right_s) = \
        input_gate, forget_gate, output_gate

    fig = plt.figure()

    # super title
    fig.suptitle('Saturation Plot', fontsize = 16)

    # plot input gate
    ax = fig.add_subplot(1, 3, 1, aspect = 'equal')
    x, y = input_right_s, input_left_s
    radius = 0.03 * np.ones(len(x))
    patches = []
    for xc, yc, r in zip(x, y, radius):
        circle = Circle((xc, yc), r)
        patches.append(circle)
    colors = 100 * np.ones(len(patches))    # color of circle
    p = PatchCollection(patches, cmap = matplotlib.cm.jet, alpha = 1)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    cb = fig.colorbar(p, ax = ax)
    cb.remove() # remove color bar
    plt.draw()  # update plot

    # plot forget gate
    ax = fig.add_subplot(1, 3, 2, aspect = 'equal')
    x, y = forget_right_s, forget_left_s
    radius = 0.03 * np.ones(len(x))
    patches = []
    for xc, yc, r in zip(x, y, radius):
        circle = Circle((xc, yc), r)
        patches.append(circle)
    colors = 100 * np.ones(len(patches))    # color of circle
    p = PatchCollection(patches, cmap = matplotlib.cm.jet, alpha = 1)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    cb = fig.colorbar(p, ax = ax)
    cb.remove() # remove color bar
    plt.draw()  # update plot

    # plot output gate
    ax = fig.add_subplot(1, 3, 3, aspect = 'equal')
    x, y = output_right_s, output_left_s
    radius = 0.03 * np.ones(len(x))
    patches = []
    for xc, yc, r in zip(x, y, radius):
        circle = Circle((xc, yc), r)
        patches.append(circle)
    colors = 100 * np.ones(len(patches))    # color of circle
    p = PatchCollection(patches, cmap = matplotlib.cm.jet, alpha = 1)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    cb = fig.colorbar(p, ax = ax)
    cb.remove() # remove color bar
    plt.draw()  # update plot

    # plt.subplots_adjust(left = 0.125, bottom = 0.15, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.3)
    plt.show()
