"""
Base functions and helpers to construct 2D real-valued even-symmetric shearlet in the frequency domain.
"""
import numpy as np
from coshrem.util.cone import cone_orientation
from functools import lru_cache


def construct_shearlet(rows, cols, wavelet_eff_supp,
                       gaussian_eff_supp, scales_per_octave,
                       shear_level, alpha, sample_wavelet_off_origin,
                       scale, ori, coneh, ks):
    """Construct a 2D real-valued even-symmetric shearlet in the frequency domain.

    Args:
        rows (int): Height of the constructed shearlet.
        cols (int): Width of the constructed shearlet.
        wavelet_eff_supp (int): Effective support for wavelet function used in construction.
        gaussian_eff_supp (int): Effective support for Gauss function used in construction.
        scales_per_octave (float): Number of scales per octave.
        shear_level (int): Amount of shearing applied.
        sample_wavelet_off_origin (bool): Sample the wavelet off-origin.
        scale: Scaling parameter
        ori: Orientation parameter

    Returns:
        A even-symmetric real-valued shearlet.

    """
    if ori not in coneh:
        rows, cols = cols, rows

    omega_wav = (63 * float(wavelet_eff_supp) / 512) * yapuls(rows)
    omega_gau = (74 * float(gaussian_eff_supp) / 512) * yapuls(cols * (2 ** (shear_level - 2)))
    # Scaling omega
    omega_gau = omega_gau / ((2 ** ((scale - 1) / scales_per_octave)) ** alpha)
    omega_wav = omega_wav / (2 ** ((scale - 1) / scales_per_octave))
    # Mexican hat wavelet in the frequency domain
    wav_freq = np.atleast_2d(2 * np.pi * np.multiply(np.power(omega_wav, 2), np.exp(np.power(omega_wav, 2) / -2)))
    # Off-Origin sampling
    if sample_wavelet_off_origin:
        wav_freq = padarray(np.fft.fftshift(wav_freq), (1, wav_freq.shape[1] * 2))
        wav_time = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(wav_freq))))

        wav_slice = slice(1, None, 2) if (len(wav_freq) % 2 != 0) else slice(0, -1, 2)
        if ori in coneh:
            wav_freq = np.fft.fft(np.fft.ifftshift(wav_time[::-1, wav_slice]))
        else:
            wav_freq = np.fft.fft(np.fft.ifftshift(wav_time[:, wav_slice]))

    gau_freq = np.atleast_2d(np.exp(-1 * np.power(omega_gau, 2) / 2))

    if ori in coneh:
        shearlet = np.fft.fftshift(wav_freq.T * gau_freq)
        shearlet = shear(shearlet, -1 * ks[ori - 1], 2)
        shearlet = shearlet[:, ::(2 ** (shear_level - 2))]

    else:
        shearlet = np.fft.fftshift(gau_freq.T * wav_freq)
        shearlet = shear(shearlet, -1 * ks[ori - 1], 1)
        shearlet = shearlet[::(2 ** (shear_level - 2)), :]

    return shearlet


def padarray(array, newsize):
    """
    Pad array
    Args:
        array: input array
        newsize: shape for padded array

    Returns:
        padded array

    """
    sizediff = [b-a for a, b in zip(array.shape, newsize)]
    pad = [diff // 2 if (diff % 2 == 0)
           else (diff // 2) + 1
           for diff in sizediff]
    lshift = [1 if (size % 2 == 0) and not (diff % 2 == 0)
              else 0
              for diff, size in zip(sizediff, array.shape)]
    return np.pad(array, [(a - s, new - (a - s + old))
                          for a, s, new, old in zip(pad, lshift, newsize, array.shape)],
                  mode='constant')


def shear(data, k, axis):
    """
    Discretely shear the input data on given axis by k.

    Originally from 'ShearLab 3D'. See http://www.shearlab.org/ .
    Args:
        data: The input array (e.g. base shearlet)
        k: The amount of shearing
        axis: Axis to shear on

    Returns:
        Sheared input data

    """
    if k == 0:
        return data
    rows, cols = data.shape

    ret = np.zeros(data.shape, dtype=data.dtype)
    if axis == 1:
        for col in range(cols):
            ret[:, col] = np.roll(data[:, col], __shift(k, cols, col))
    else:
        for row in range(rows):
            ret[row, :] = np.roll(data[row, :], __shift(k, rows, row))

    return ret


def __shift(k, total, x):
    """Compute (circular) shift for one column during shearing."""
    return k * ((total // 2) - x)


def yapuls(npuls):
    """
    Originally from 'Yet Another Wavelet Toolbox (YAWTb)'. See http://sites.uclouvain.be/ispgroup/yawtb/ :
    Original documentation:

    Returns a pulsation vector puls of size npuls which is the concatenation of two
    subvectors whose elements are respectively in [0, \pi) and [-\pi, 0).

    Args:
        npuls: length of the pulsation vector

    Returns:
        Pulsation vector

    """
    npuls_2 = (npuls - 1) // 2
    return (2 * np.pi / npuls) * np.concatenate((np.arange(npuls_2 + 1),
                                                 np.arange(npuls_2 - npuls + 1, 0)))