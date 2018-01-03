#!/bin/python3

import coshrem.shearlet
from coshrem.shearletsystem import EdgeSystem, RidgeSystem
from coshrem.util.image import overlay, mask, thin_mask, curvature_rgb
from coshrem.util.curvature import curvature
import coshrem.util
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_systems(sys1, sys2, r=None):
    if r is None:
        r = range(sys1.shape[2])
    f, axes = plt.subplots(len(r), 3, sharex=True, sharey=True)
    for i, sh in enumerate(r):
        v_max = np.max(np.real(sys1[:, :, sh]))
        v_min = np.min(np.real(sys1[:, :, sh]))
        axes[i, 0].imshow(np.real(sys1[:, :, sh]), interpolation='none', vmax=v_max, vmin=v_min)
        axes[i, 0].set_title("Shearlet " + str(sh))
        axes[i, 1].imshow(np.real(sys2[:, :, sh]), interpolation='none', vmax=v_max, vmin=v_min)
        axes[i, 2].set_title("Difference " + str(sh))
        axes[i, 2].imshow(np.real(sys2[:, :, sh]) - np.real(sys1[:, :, sh]),
                          interpolation='none')
    plt.show()


def quickridge(filename, min_contrast=4, sys=None, max_curvature=10, cli=False):
    i = Image.open(filename).convert("L")
    img = np.asarray(i)
    if sys is None:
        sys = RidgeSystem(*img.shape)
    ri, ori = sys.detect(img, min_contrast)
    thinned_ri = mask(ri, thin_mask(ri))
    thinned_ori = mask(ori, thin_mask(ri))
    rgb_curv = curvature_rgb(curvature(thinned_ori), max_curvature=max_curvature)

    f, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(2, 4, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

    ax11.imshow(img, cmap='gray', interpolation='none')
    ax11.set_title("Original image")

    ax12.imshow(ri, cmap='jet', interpolation='none')
    ax12.set_title("Ridge\nmeasurement")

    ax13.imshow(thinned_ri, cmap='jet', interpolation='none')
    ax13.set_title("Thinned ridge\nmeasurement")

    ax21.imshow(overlay(img, ri), interpolation='none')
    ax21.set_title("Ridge overlay")

    ax22.imshow(ori, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax22.set_title("Orientation\nmeasurement")

    ax23.imshow(thinned_ori, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax23.set_title("Thinned orientation\nmeasurement")

    ax24.imshow(rgb_curv, interpolation='none')
    ax24.set_title("Local curvature\nmeasurement")

    if(cli):
        Image.fromarray(ri).save("ridges.jpg", "JPEG")
        f.set_size_inches(11.692, 8.267)
        f.set_dpi(300)
        f.savefig("ridge_example.jpg")
    else:
        plt.show()


def quickedge(filename, min_contrast=4, sys=None, cli=False, max_curvature=10):
    i = Image.open(filename).convert("L")
    img = np.asarray(i)
    if sys is None:
        sys = EdgeSystem(*img.shape)
    ed, ori = sys.detect(img, min_contrast)

    thinned_ed = mask(ed, thin_mask(ed))
    thinned_ori = mask(ed, thin_mask(ed))
    rgb_curv = curvature_rgb(curvature(thinned_ori), max_curvature=max_curvature)

    f, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(2, 4, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

    ax11.imshow(img, cmap='gray', interpolation='none')
    ax11.set_title("Original\nimage")

    ax12.imshow(ed, cmap='jet', interpolation='none')
    ax12.set_title("Edge\nmeasurement")

    ax13.imshow(thinned_ed, cmap='jet', interpolation='none')
    ax13.set_title("Thinned edge\nmeasurement")

    ax21.imshow(overlay(img, ed), interpolation='none')
    ax21.set_title("Edge overlay")

    ax22.imshow(ori, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax22.set_title("Orientation\nmeasurement")

    ax23.imshow(thinned_ori, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax23.set_title("Thinned orientation\nmeasurement")

    ax24.imshow(rgb_curv, cmap='jet', interpolation='none')
    ax24.set_title("Local curvature\nmeasurement")

    if(cli):
        Image.fromarray(ed, mode="L").save("edges.jpg", "JPEG")
        f.set_size_inches(11.692, 8.267)
        f.set_dpi(300)
        f.savefig("edge_example.jpg")
    else:
        plt.show()


if __name__ == "__main__":
    #  quickedge("tests/resources/img/lena.jpg", cli=True)
    pass
