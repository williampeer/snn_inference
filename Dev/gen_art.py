import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

from ipywidgets import widgets
from ipywidgets import interact, interact_manual, interactive

x_bounds = [-10, 10]
y_bounds = [-10, 10]
x_buffer, y_buffer = 1, 1

x_plot = x_bounds + np.array([x_buffer, -x_buffer])
y_plot = y_bounds + np.array([y_buffer, -y_buffer])


def make_some_art(num_points=200, percent_to_fill=0.5, n_fill_lines=5, min_scalar=0.1, debug=False,
                  toggle_for_new=False):
    x = np.random.uniform(*x_bounds, size=num_points).reshape((num_points, 1))
    y = np.random.uniform(*y_bounds, size=num_points).reshape((num_points, 1))
    pts = np.hstack([x, y])

    vor = Voronoi(pts)
    verts = vor.vertices
    shapes_ind = vor.regions

    shapes_ind = [s + s[0:1] for s in shapes_ind if len(s) > 0 and -1 not in s]
    shapes = [verts[s] for s in shapes_ind]

    n_shapes_to_fill = int(percent_to_fill * len(shapes))
    shapes_to_fill = np.random.choice(shapes, size=n_shapes_to_fill, replace=False)

    fill = []

    for s in shapes_to_fill:
        center = np.mean(s, axis=0)
        for scaler in np.linspace(min_scalar, 1, num=n_fill_lines, endpoint=False):
            scaled = scaler * (s - center) + center
            fill.append(scaled)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_aspect('equal')

    if not debug:
        plt.grid(False)
        plt.axis('off')

    ax.set_xlim(*x_plot)
    ax.set_ylim(*y_plot)
    lc = LineCollection(shapes + fill)
    ax.add_collection(lc)

    return fig, ax


w = interactive(make_some_art,
                num_points=(10, 1000, 25),
                percent_to_fill=(0., 1., 0.05),
                n_fill_lines=(1, 20, 1),
                min_scalar=(0, 1, 0.01))

plt.show()
# """
# So now we have the fill figured out, and we have to put it back into place!
# """
#
# filled_polygon = shapes
#
# n_fill_lines = 5
# min_scalar = 0.1
#
# for scaler in np.linspace(min_scalar, 1, num=n_fill_lines, endpoint=False):
#     scaled = scaler * (polygon - center) + center
#     filled_polygon.append(scaled)
#
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_xlim(*x_plot)
# ax.set_ylim(*y_plot)
# lc = LineCollection(filled_polygon)
# ax.add_collection(lc)


# circles = []
# n_tries = 5000
#
# x_bounds = [-10, 10]
# y_bounds = [-10, 10]
# min_radius = 0.1
# max_radius = 2
#
# for i in range(n_tries):
#     cx, cy = np.random.uniform(*x_bounds), np.random.uniform(*y_bounds)
#     c = np.array([cx, cy])
#
#     r = max_radius
#
#     for ci, ri in circles:
#         dist = np.linalg.norm(c - ci)
#         largest_r = dist - ri
#         largest_r = np.clip(largest_r, 0, largest_r)
#         r = min(r, largest_r)
#
#     if r >= min_radius:
#         circles.append((c, r))
#
# blue_patches = []
# red_patches = []
#
# for c, r in circles:
#
#     if np.random.random() > 0.5:
#         blue_patches.append(mpatches.Circle(c, r, fill=None, edgecolor='blue'))
#     else:
#         red_patches.append(mpatches.Circle(c, r, fill=None, edgecolor='red'))
#
#
# fig, ax = plt.subplots(figsize=(10, 10))
#
# plt.grid(False)
# plt.axis('off')
# ax.set_aspect('equal')
#
# ax.set_xlim(x_bounds)
# ax.set_ylim(y_bounds)
#
# collection = PatchCollection(blue_patches+red_patches, match_original=True)
# ax.add_collection(collection)
#
# plt.show()
#
#
# fig, ax = plt.subplots(figsize=(10, 10))
#
# plt.grid(False)
# plt.axis('off')
# ax.set_aspect('equal')
#
# ax.set_xlim(x_bounds)
# ax.set_ylim(y_bounds)
#
# collection = PatchCollection(blue_patches, match_original=True)
# ax.add_collection(collection)
#
# plt.savefig('blue_layer.svg', bbox_inches = 'tight', pad_inches = 0)
# plt.show()