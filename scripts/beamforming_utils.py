import numpy as np
import matplotlib.pyplot as plt
import torch


def make_pixel_grid(xlims, zlims, dx, dz):
    x_pos = np.arange(xlims[0], xlims[1], dx)
    z_pos = np.arange(zlims[0], zlims[1], dz)
    zz, xx = np.meshgrid(z_pos, x_pos, indexing="ij") # 'ij' -> rows: z, columns: x
    yy = xx * 0
    grid = np.stack((xx, yy, zz), axis=-1)  # [nrows, ncols, 3]
    return grid


def make_pixel_grid_from_pos(x_pos, z_pos):
    zz, xx = np.meshgrid(z_pos, x_pos, indexing="ij") # 'ij' -> rows: z, columns: x
    yy = xx * 0
    grid = np.stack((xx, yy, zz), axis=-1)  # [nrows, ncols, 3]
    return grid


def plot_bimg(bimg, xlims, zlims, title):
    # plt.figure()
    # extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
    # plt.imshow(bimg, vmin=-60, cmap="gray", extent=extent, origin="upper")
    # plt.title(title)
    # plt.colorbar()
    # plt.show()

    fig, ax = plt.subplots(1, 1)
    extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
    im_plot = ax.imshow(bimg, vmin=-60, cmap="gray", extent=extent, origin="upper")
    ax.set_title(title)
    fig.colorbar(im_plot, ax=ax)
    plt.show()
    return fig, ax


if __name__ == '__main__':
    xlims = [-20*1e-3, 20*1e-3]
    zlims = [5*1e-3, 35*1e-3]
    dx, dz = 1e-3, 1e-3
    grid = make_pixel_grid(xlims, zlims, dx, dz)