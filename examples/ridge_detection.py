#!/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt

from coshrem.shearletsystem import RidgeSystem
from coshrem.util.image import overlay, mask, thin_mask, curvature_rgb
from coshrem.util.curvature import curvature
import coshrem.util
from PIL import Image


def main():
    filename = "../tests/resources/img/B0.png"
    def output_filename(str):
        basename, extension = os.path.splitext(os.path.basename(filename))
        if not os.path.exists(basename):
            os.makedirs(basename)
        return basename + "/" + basename + "." + str + extension

    # Open an image and read it in as numpy array
    i = Image.open(filename).convert("L")
    img = np.asarray(i)

    # Create a shearlet system for ridge detection using default parameters
    #
    # The same system could be created via
    #
    # sys = RidgeSystem(*img.shape,
    #                    wavelet_eff_supp=60,
    #                    gaussian_eff_supp=20,
    #                    scales_per_octave=4,
    #                    shear_level=3,
    #                    alpha=0.2,
    #                    octaves=3.5
    #                    )
    sys = RidgeSystem(*img.shape)

    # Perform ridge and ridge-orientation measurement
    ridges, orientations = sys.detect(img, min_contrast=10)

    # Use a mask on the array to thin ridges to single pixel width
    thinned_ridges = mask(ridges, thin_mask(ridges))
    thinned_orientations = mask(orientations, thin_mask(ridges))

    # Perform curvature measurement on the thinned orientations and convert into rgb image
    rgb_curv = curvature_rgb(curvature(thinned_orientations), max_curvature=10)

    # Write images to disk
    Image.fromarray(np.uint8(ridges* 255), mode="L").save(output_filename("ridges"))
    Image.fromarray(overlay(img, ridges), mode="RGB").save(output_filename("overlay_ridges"))
    Image.fromarray(np.uint8(thinned_ridges * 255), mode="L").save(output_filename("thin_ridges"))

    # Create overview diagram with matplotlib
    f, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(2, 4, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

    ax11.imshow(img, cmap='gray', interpolation='none')
    ax11.set_title("Original image")

    ax12.imshow(ridges, cmap='jet', interpolation='none')
    ax12.set_title("Ridge\nmeasurement")

    ax13.imshow(thinned_ridges, cmap='jet', interpolation='none')
    ax13.set_title("Thinned ridge\nmeasurement")

    ax14.axis('off')

    ax21.imshow(overlay(img, ridges), interpolation='none')
    ax21.set_title("Ridge overlay")

    ax22.imshow(orientations, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax22.set_title("Orientation\nmeasurement")

    ax23.imshow(thinned_orientations, cmap=coshrem.util.image.cyclic_cmap(), interpolation='none')
    ax23.set_title("Thinned orientation\nmeasurement")

    ax24.imshow(rgb_curv, interpolation='none')
    ax24.set_title("Local curvature\nmeasurement")

    f.set_tight_layout(True)
    f.set_size_inches(16, 9)
    f.set_dpi(1600)
    f.savefig(output_filename("diagram"))


if __name__ == "__main__":
    main()

