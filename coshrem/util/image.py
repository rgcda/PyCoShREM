import numpy as np
import numpy.ma as ma
from skimage.morphology import medial_axis
import matplotlib.colors as colors
import matplotlib.cm as cm


def overlay(img, img_overlay):
    img_overlay = img_overlay / img_overlay.max()
    img = (img / img.max()) * (1 - img_overlay)
    red = np.empty((img.shape[0], img.shape[1], 3))
    img_rgb = np.empty((img.shape[0], img.shape[1], 3))
    red[:, :, 0] = img_overlay
    for i in range(3):
        img_rgb[:, :, i] = img
    return np.uint8((img_rgb + red) * 255)


def thin_mask(img):
    return np.invert(medial_axis(img))


def mask(img, mask):
    ret = ma.masked_array(img)
    ret.fill_value = -1
    ret.mask = mask
    return ret


def cyclic_cmap():
    white = '#ffffff'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    return colors.LinearSegmentedColormap.from_list('anglemap',
                                                    [black, red, white, blue, black],
                                                    N=180,
                                                    gamma=1)


def curvature_rgb(curvature, max_curvature=10):
    rgb = curvature
    rgb[curvature > max_curvature] = max_curvature
    rgb = rgb * 255 / max_curvature
    rgb[curvature < 0] = -1
    return cm.jet(rgb)
