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
    # num of layers
    num_layers = len(input_gate)

    fig = plt.figure()

    # super title
    fig.suptitle('Saturation Plot', fontsize = 16)

    color_base = [100, 60, 30]

    # plot forget gate -------------------------------------------
    ax = fig.add_subplot(1, 3, 2, aspect = 'equal')
    ax.set_title('Forget Gate')
    ax.set_xlabel('fraction right saturated')
    ax.set_ylabel('fraction left saturated')

    for i in range(num_layers):
        x, y = forget_right_s[i], forget_left_s[i]
        radius = 0.03 * np.ones(len(x))
        patches = []
        for xc, yc, r in zip(x, y, radius):
            circle = Circle((xc, yc), r)
            patches.append(circle)

        colors = color_base[i] * np.ones(len(patches))    # color of circle
        p = PatchCollection(patches, cmap = matplotlib.cm.jet, alpha = 1)
        p.set_array(np.array(colors))
        ax.add_collection(p)

    # plot reverse diagnoal
    ax.plot(np.linspace(0, 1), np.linspace(1, 0))

    cb = fig.colorbar(p, ax = ax)
    cb.remove() # remove color bar
    plt.draw()  # update plot

    # plot input gate -------------------------------------------
    ax = fig.add_subplot(1, 3, 1, aspect = 'equal')
    ax.set_title('Input Gate')
    ax.set_xlabel('fraction right saturated')
    ax.set_ylabel('fraction left saturated')

    for i in range(num_layers):
        x, y = input_right_s[i], input_left_s[i]
        radius = 0.03 * np.ones(len(x))
        patches = []
        for xc, yc, r in zip(x, y, radius):
            circle = Circle((xc, yc), r)
            patches.append(circle)

        colors = 100 * np.ones(len(patches))  # color of circle
        p = PatchCollection(patches, cmap = matplotlib.cm.jet, alpha = 1)
        p.set_array(np.array(colors))
        ax.add_collection(p)

    # plot reverse diagnoal
    ax.plot(np.linspace(0, 1), np.linspace(1, 0))

    cb = fig.colorbar(p, ax=ax)
    cb.remove()  # remove color bar
    plt.draw()  # update plot

    # plot output gate -------------------------------------------
    ax = fig.add_subplot(1, 3, 3, aspect = 'equal')
    ax.set_title('Output Gate')
    ax.set_xlabel('fraction right saturated')
    ax.set_ylabel('fraction left saturated')

    for i in range(num_layers):
        x, y = output_right_s[i], output_left_s[i]
        radius = 0.03 * np.ones(len(x))
        patches = []
        for xc, yc, r in zip(x, y, radius):
            circle = Circle((xc, yc), r)
            patches.append(circle)

        colors = 100 * np.ones(len(patches))    # color of circle
        p = PatchCollection(patches, cmap = matplotlib.cm.jet, alpha = 1)
        p.set_array(np.array(colors))
        ax.add_collection(p)

    # plot reverse diagnoal
    ax.plot(np.linspace(0, 1), np.linspace(1, 0))

    cb = fig.colorbar(p, ax = ax)
    cb.remove() # remove color bar
    plt.draw()  # update plot

    plt.show()
