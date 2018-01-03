#!/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt

from coshrem.shearletsystem import EdgeSystem
from coshrem.util.image import overlay, mask, thin_mask, curvature_rgb
from coshrem.util.curvature import curvature
import coshrem.util
from PIL import Image


def main():
    filename = "../tests/resources/img/lena.jpg"
    def output_filename(str):
        basename, extension = os.path.splitext(os.path.basename(filename))
        if not os.path.exists(basename):
            os.makedirs(basename)
        return basename + "/" + basename + "." + str + extension

    # Open an image and read it in as numpy array
    i = Image.open(filename).convert("L")
    img = np.asarray(i)

    # Create a shearlet system for edge detection using default parameters
    #
    # An identical system would be created by
    #
    # sys = EdgeSystem(*img.shape,
    #                   wavelet_eff_supp=70,
    #                   gaussian_eff_supp=25,
    #                   scales_per_octave=2,
    #                   shear_level=3,
    #                   alpha=0.5,
    #                   octaves=3.5
    #                   )
    sys = EdgeSystem(*img.shape)

    # Perform edge and edge-orientation measurement
    edges, orientations = sys.detect(img, min_contrast=4)

    # Use a mask on the array to thin edges to single pixel width
    thinned_edges = mask(edges, thin_mask(edges))
    thinned_orientations = mask(edges, thin_mask(edges))

    # Perform curvature measurement on the thinned orientations and convert into rgb image
    rgb_curvature = curvature_rgb(curvature(thinned_orientations), max_curvature=10)

    # Write images to disk
    Image.fromarray(np.uint8(edges * 255), mode="L").save(output_filename("edges"))
    Image.fromarray(overlay(img, edges), mode="RGB").save(output_filename("overlay_edges"))
    Image.fromarray(np.uint8(thinned_edges * 255), mode="L").save(output_filename("thin_edges"))

    # Create overview diagram with matplotlib
    f, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(2, 4, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

    ax11.imshow(img, cmap='gray', interpolation='none')
    ax11.set_title("Original\nimage")

    ax12.set_title("Edge\nmeasurement")
    ax12.imshow(edges, cmap='jet', interpolation='none')

    ax13.set_title("Thinned edge\nmeasurement")
    ax13.imshow(thinned_edges, cmap='jet', interpolation='none')

    ax14.axis('off')

    ax21.set_title("Edge overlay")
    ax21.imshow(overlay(img, edges), interpolation='none')

    # To visualize the orientation measurement, use a cyclic colormap where 0 and 180 are similar or equal
    ax22.set_title("Orientation\nmeasurement")
    ax22.imshow(orientations, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')

    ax23.set_title("Thinned orientation\nmeasurement")
    ax23.imshow(thinned_orientations, coshrem.util.image.cyclic_cmap(), interpolation='none')

    ax24.set_title("Local curvature\nmeasurement")
    ax24.imshow(rgb_curvature, cmap='jet', interpolation='none')

    f.set_size_inches(16, 9)
    f.set_tight_layout(True)
    f.set_dpi(1600)
    f.savefig(output_filename("diagram"))


if __name__ == "__main__":
    main()

