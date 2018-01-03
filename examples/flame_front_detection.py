#!/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt

from coshrem.shearletsystem import EdgeSystem, RidgeSystem
from coshrem.util.image import overlay, mask, thin_mask, curvature_rgb
from coshrem.util.curvature import curvature
import coshrem.util
from PIL import Image

def output_filename(str, fn):
    basename, extension = os.path.splitext(os.path.basename(fn))
    dirname = basename.split("_")[0]
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname + "/" + basename + "." + str + '.jpg'

def main():
    filename_edge = "../tests/resources/data/K05_OH_full.txt"
    filename_ridge = "../tests/resources/data/K05_CH_full.txt"


    # Open an image data from txtfile and read it in as numpy array
    img_edges = np.loadtxt(filename_edge)
    img_ridges = np.loadtxt(filename_ridge)

    # Create a shearlet system for edge detection with tweaked parameters
    edge_sys = EdgeSystem(*img_edges.shape,
                          wavelet_eff_supp=100,
                          gaussian_eff_supp=25,
                          scales_per_octave=2,
                          shear_level=3,
                          alpha=0.8,
                          octaves=4
                          )

    # Perform edge and edge-orientation measurement
    edges, edge_orientations = edge_sys.detect(img_edges, min_contrast=70, pivoting_scales='lowest')

    # Create a shearlet system for ridge detection with tweaked parameters
    ridge_sys = RidgeSystem(*img_ridges.shape,
                            wavelet_eff_supp=60,
                            gaussian_eff_supp=20,
                            scales_per_octave=4,
                            shear_level=3,
                            alpha=0.2,
                            octaves=4
                            )


    # Perform ridge and ridge-orientation measurement
    ridges, ridge_orientations = ridge_sys.detect(img_ridges, min_contrast=20)

    # Use a mask on the array to thin ridges and ridges to single pixel width
    thinned_ridges = mask(ridges, thin_mask(ridges))
    thinned_ridge_orientations = mask(ridge_orientations, thin_mask(ridges))
    thinned_edges = mask(edges, thin_mask(edges))
    thinned_edge_orientations = mask(edge_orientations, thin_mask(edges))

    # Perform curvature measurement on the thinned orientations and convert into rgb image
    rgb_ridge_curv = curvature_rgb(curvature(thinned_ridge_orientations), max_curvature=10)
    rgb_edge_curv = curvature_rgb(curvature(thinned_edge_orientations), max_curvature=10)

    # Write images to disk
    Image.fromarray(np.uint8(ridges* 255), mode="L").save(output_filename("ridges", filename_ridge), quality=100)
    Image.fromarray(overlay(img_ridges, ridges), mode="RGB").save(output_filename("overlay_ridges", filename_ridge), quality=100)
    Image.fromarray(np.uint8(thinned_ridges * 255), mode="L").save(output_filename("thin_ridges", filename_ridge), quality=100)
    Image.fromarray(np.uint8(edges* 255), mode="L").save(output_filename("edges", filename_edge), quality=100)
    Image.fromarray(overlay(img_edges, edges), mode="RGB").save(output_filename("overlay_edges", filename_edge), quality=100)
    Image.fromarray(np.uint8(thinned_edges* 255), mode="L").save(output_filename("thin_edges", filename_edge), quality=100)

    # Create overview diagram with matplotlib
    f, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24),
        (ax31, ax32, ax33, ax34), (ax41, ax42, ax43, ax44)) = plt.subplots(4, 4, sharex=True, sharey=True,
                                                                           subplot_kw={'adjustable':'box-forced'})

    ax11.imshow(img_edges, cmap='gray', interpolation='none')
    ax11.set_title("Original edge image")

    ax12.imshow(edges, cmap='jet', interpolation='none')
    ax12.set_title("Edge\nmeasurement")

    ax13.imshow(thinned_edges, cmap='jet', interpolation='none')
    ax13.set_title("Thinned edge\nmeasurement")

    ax14.axis('off')

    ax21.imshow(overlay(img_edges, edges), interpolation='none')
    ax21.set_title("Edge overlay")

    ax22.imshow(edge_orientations, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax22.set_title("Edge orientation\nmeasurement")

    ax23.imshow(thinned_edge_orientations, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax23.set_title("Thinned orientation\nmeasurement of edges")

    ax24.imshow(rgb_edge_curv, interpolation='none')
    ax24.set_title("Local edge\ncurvature measurement")

    ax31.imshow(img_ridges, cmap='gray', interpolation='none')
    ax31.set_title("Original ridge image")

    ax32.imshow(ridges, cmap='jet', interpolation='none')
    ax32.set_title("Ridge\nmeasurement")

    ax33.imshow(thinned_ridges, cmap='jet', interpolation='none')
    ax33.set_title("Thinned ridge\nmeasurement")

    ax34.axis('off')

    ax41.imshow(overlay(img_ridges, ridges), interpolation='none')
    ax41.set_title("Ridge overlay")

    ax42.imshow(ridge_orientations, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax42.set_title("Ridge orientation\nmeasurement")

    ax43.imshow(thinned_ridge_orientations, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax43.set_title("Thinned ridge\norientation measurement")

    ax44.imshow(rgb_ridge_curv, interpolation='none')
    ax44.set_title("Local ridge curvature\nmeasurement")
    f.set_size_inches(16, 16)
    f.set_dpi(1600)
    f.savefig(output_filename("overview", filename_edge))


if __name__ == "__main__":
    main()
