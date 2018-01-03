from abc import ABCMeta, abstractmethod
import numpy as np
from coshrem.shearlet import construct_shearlet
from coshrem.util.cone import cone_orientation


class ShearletSystem(metaclass=ABCMeta):
    """Abstract Base Class for Shearlet-Systems.

    This class can not be instantiated directly and provides some common methods for creation of
    Edge or Ridge measurement systems.

    """

    def __init__(self, rows, cols, wavelet_eff_supp,
                 gaussian_eff_supp, scales_per_octave,
                 shear_level, alpha, scales, normalize,
                 octaves):
        self.rows = rows
        self.cols = cols
        self.wavelet_eff_supp = (wavelet_eff_supp
                                 if wavelet_eff_supp else np.min((rows, cols)) / 7)
        self.gaussian_eff_supp = (gaussian_eff_supp
                                  if gaussian_eff_supp else np.min((rows, cols)) / 20)
        self.scales_per_octave = scales_per_octave
        self.shear_level = shear_level
        self.alpha = alpha
        self.scales = (scales if scales else np.arange(1, (scales_per_octave * octaves) + 1))

        self.n_oris = 2 ** self.shear_level + 2

        self.normalize = normalize

        _, self._coneh, self._ks = cone_orientation(shear_level)

        self.hilbert_matrix = np.ones((rows, cols, 2))
        self.hilbert_matrix[:(rows//2), :, 0] = -1
        self.hilbert_matrix[:, (cols//2):, 1] = -1

    @property
    def shape(self):
        """Shape of shearlets in this system"""
        return (self.rows, self.cols)

    @abstractmethod
    def detect(self):
        """Abstract method for Edge/Ridge Detection.

        Has to be implemented in Subclasses.
        Depending on the type of th system will perform edge or ridge detection on a given image.

        """
        pass

    def _transform(self, x):
        """Compute shearlet transformation of given image.

        Args:
            x: Image (e.g. 2-dimensional array)

        Returns:
            Coefficients for every pixel of the image for every shearlet in this system.
        """
        x = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
        coeffs = np.zeros_like(self.shearlets)

        for i in range(self.n_shearlets):
            self.__coeff_generation(x, coeffs, i)
        return coeffs

    def __coeff_generation(self, x, coeffs, i):
        coeffs[..., i] = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(x * (self.shearlets[:, :, i].conj())))).conj()

    def _orientations_to_angles(self, orientations):
        """Map an orientation measurement of an image to angles.

        Args:
            orientations: orientation measurement (obtained by `detect` method)

        Returns:
            Orientations of edges or ridges in angles.

        """
        mapping = np.vectorize(self.__single_ori_to_angle)
        return mapping(orientations)

    def __single_ori_to_angle(self, ori):
        """Map the orientation measurement of a single pixel to angles."""
        if (np.isnan(ori) or ori < 0):
            return -1

        cone = self._coneh.min() <= ori <= self._coneh.max()
        delta = ori - np.floor(ori)

        if np.floor(ori) == len(self._ks) - 1:
            k = (1 - delta) * self._ks[int(ori)]
        else:
            k = (1 - delta) * self._ks[int(ori)] + delta * self._ks[int(ori) + 1]

        if cone == 1:
            ret = np.pi / 2 + np.arctan((k / (2 ** (self.shear_level - 2))))
        else:
            if ori < self._coneh.min():
                ret = np.pi - np.arctan((k / (2 ** (self.shear_level - 2))))
            else:
                ret = -1 * np.arctan((k / (2 ** (self.shear_level - 2))))
        return 180 * ret / np.pi


class RidgeSystem(ShearletSystem):
    """A Shearlet-System of real-valued shearlets for ridge measurement.

    During class creation a system of shearlets is constructed.

    Args:
        rows (int): Height of the constructed shearlet.
        cols (int): Width of the constructed shearlet.
        wavelet_eff_supp (Optional[int]): Effective support for wavelet function used in construction.

            Defaults to ``min(rows,cols) / 7``
        gaussian_eff_supp (Optional[int]): Effective support for Gauss function used in construction.

            Defaults to ``min(rows,cols) / 20``
        scales_per_octave (Optional[float]): Number of scales per octave.

            Defaults to ``2``.
        shear_level (Optional[int]): Amount of shearing applied.

            Defaults to ``3``.
        normalize (Optional[bool]): Normalize shearlets during construction.

            Defaults to ``True``.

    Attributes:
        n_shearlets (int): Number of shearlets.
        rows (int): Height of shearlets.
        cols (int): Widthof shearlets.
        n_oris (int): Number of orientations.

        shearlets: Multidimensional array with shearlets in this system.

    """

    def __init__(self, rows, cols, wavelet_eff_supp=None,
                 gaussian_eff_supp=None, scales_per_octave=2,
                 shear_level=3, alpha=0.5, scales=None, normalize=True,
                 octaves=3.5):
        ShearletSystem.__init__(self, rows, cols, wavelet_eff_supp, gaussian_eff_supp,
                                scales_per_octave, shear_level, alpha, scales, normalize,
                                octaves)

        self.n_shearlets = len(self.scales) * self.n_oris * 2
        self.shearlets = np.zeros((self.rows, self.cols, self.n_shearlets), dtype=np.complex_)

        self.positive_widths = np.zeros(self.n_shearlets)
        self.expected_coeffs = np.zeros((self.rows, len(self.scales)))

        for j, scale in enumerate(self.scales):
            for ori in range(self.n_oris):
                for off_origin in [1, 0]:
                    shearlet = self._single_shearlet(scale, ori + 1)

                    if self.normalize and ori == 0:
                        # TODO: normalize and positive_widths in frequency domain
                        shearlet_t = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(shearlet)))
                        center_ridge = np.real(shearlet_t[shearlet_t.shape[0] // 2, (shearlet_t.shape[1] // 2):])
                        positive_width = (np.argmax(center_ridge < 0) - 1 + (1 - off_origin)) * 2 - 1 + off_origin
                        self.positive_widths[2 * j + (1 - off_origin)] = positive_width

                        help_width = (positive_width + off_origin)//2
                        # add 1 for numpy/matlab slice difference
                        normalization = 2 * np.sum(np.real(
                            shearlet_t[:, (shearlet_t.shape[1] // 2) - help_width:(shearlet_t.shape[1] // 2) + 1 + off_origin + help_width]))

                        shearlet = shearlet / normalization
                        for k in range(off_origin,
                                       int(self.positive_widths[
                                           off_origin:(self.positive_widths.shape[0] - (1 - off_origin))].max() - 1 + off_origin) // 2):
                            self.expected_coeffs[2 * k + (1 - off_origin) - 1, j] = np.real(np.sum(shearlet_t[:, (
                                (shearlet_t.shape[1] // 2) - k):((shearlet_t.shape[1] // 2) + (1 - off_origin) + k)]))

                    else:
                        shearlet = shearlet / normalization

                    self.shearlets[:, :, 2 * self.n_oris * j + 2 * ori + (1 - off_origin)] = shearlet

    def _single_shearlet(self, scale, ori, off_origin=None):
        """Construct a single base shearlet for use in this ridge detection system.

        For further information on parameters see the documentation of the function `construct_shearlet`.

        Args:
            scale: the scaling factor for this shearlet.
            ori: the orientation for this shearlet
            off_origin: if `True` this shearlet will be sampled off-origin.

        Returns:
            A shearlet to be used for transformations in an ridge system.

        """
        shearlet_f = construct_shearlet(self.rows, self.cols, self.wavelet_eff_supp,
                                        self.gaussian_eff_supp, self.scales_per_octave,
                                        self.shear_level, self.alpha,
                                        off_origin, scale, ori, self._coneh, self._ks)
        if ori not in self._coneh:
            shearlet_f = shearlet_f + shearlet_f * self.hilbert_matrix[:, :, 1]
        else:
            shearlet_f = shearlet_f + shearlet_f * self.hilbert_matrix[:, :, 0]
        return shearlet_f

    def detect(self, x, min_contrast=None, offset=1, positive_only=True, negative_only=False, pivoting_scales='all'):
        """Compute ridges in image based on this shearlet system.

        The image to perform ridge detection on must have the same shape that was used
        during creation of this RidgeSystem.

        >>> img = np.array([[255, 120], [120, 255]])
        >>> sys = RidgeSystem(*img.shape)
        >>> x.shape == sys.shape
        True
        >>> ridges, orientations = sys.detect(img)

        By default only "positive" ridges are detected. "Positive ridges" are brighter ridges on darker background,
        whereas "negative" ridges are darker than the background. If both should be detected, artifacts will show up,
        but it is still  possible by setting both ``positive_only`` and ``negative_only`` to ``False``. By default only
        "positive", brighter Ridges are detected.

        The ``pivoting_scales`` attribute may be used to only detect ridges via low ('lowest') or high ('highest')
        frequency shearlets of this system, effectively filtering out higher frequecies (e.g. noise) of the source
        image.

        Args:
            x: Image (2-dimensional array) to compute ridges on.
            min_contrast (Optional[int]): Minimal contrast.
                This is a hard cutoff value during edge measurement.
            pivoting_scales (Optional[string]): Scales used during pivot search.
                One of ``all``, ``highest``, ``lowest``.

                Defaults to ``all``.
            positive_only (Optional[boolean]): Detect "positive" (brighter) ridges.

                Defaults to ``True``.
            negative_only (Optional[boolean]): Detect "negative" (darker) ridges.

                Defaults to ``False``.

        Returns:
            An ridge measurement of ``x`` with this system. The returned shape is equal to the image (and system) shape.
            An orientation measurement giving the angle of every edge detected by the system.

        """
        if min_contrast is None:
            min_contrast = (x.max() - x.min()) / 30
        transformed = self._transform(x)
        transformed = np.reshape(transformed,
                                 (transformed.shape[0], transformed.shape[1], (self.n_oris * 2),
                                  transformed.shape[2] // (self.n_oris * 2)),
                                 order="F")

        offset = int(offset * self.scales_per_octave)

        ridgeness = np.zeros_like(x, dtype=np.float)
        widths = np.zeros_like(x, dtype=np.float)
        heights = np.zeros_like(x, dtype=np.float)

        orientations = np.zeros_like(x, dtype=np.float)

        ci = np.abs(np.imag(transformed[:, :, :, :(-offset)]))
        cr = np.real(transformed[:, :, :, offset:])

        max_pivot = np.argmax(np.reshape(np.abs(cr),
                                         (cr.shape[0], cr.shape[1],
                                          cr.shape[2] * cr.shape[3]),
                                         order="F"
                                         ), axis=2)
        pivot_oris = np.mod(max_pivot, self.n_oris * 2)
        pivot_scales = np.fix(((max_pivot) / (self.n_oris * 2))).astype(int)

        positive_widths = self.positive_widths[offset * 2:].astype(int)
        expected_coeffs = self.expected_coeffs[:positive_widths.max() + 1, offset:]

        scales = ci.shape[3]  # all scales

        linc, rinc, cone_border1, cone_border2 = self._orientation_incs()
        cone_border1 -= 1
        cone_border2 -= 1

        right_hit = [cone_border1, cone_border2]
        left_hit = [r + 1 for r in right_hit]

        for row in range(self.rows):
            for col in range(self.cols):
                # Ridge detection
                po = pivot_oris[row, col] // 2
                even = pivot_oris[row, col] % 2
                ps = pivot_scales[row, col]

                widths[row, col] = positive_widths[2 * ps + even]
                heights[row, col] = cr[row, col, 2 * po + even, ps] / expected_coeffs[ positive_widths[2 * ps + even], ps]

                normalization = 0

                for scale in range(scales):
                    ridgeness[row, col] += cr[row, col, 2 * po + even, scale]
                    normalization += max(abs(cr[row, col, 2 * po + even, scale]),
                                         abs(heights[row, col] * expected_coeffs[
                                             positive_widths[2 * ps + even], scale]))

                ridgeness[row, col] = abs(ridgeness[row, col])

                for scale in range(scales):
                    ridgeness[row, col] -= ci[row, col, 2 * po + even, scale]
                    ridgeness[row, col] -= min_contrast * abs(expected_coeffs[positive_widths[2 * ps + even], scale])

                ridgeness[row, col] = max(ridgeness[row, col], 0) / normalization

                # Tangent orientation
                if ridgeness[row, col] > 0:
                    pcr = np.abs(cr[row, col, 2 * po + even, ps])
                    lpcr = pcr
                    rpcr = pcr

                    lcr = abs(cr[row, col, 2 * (po + linc[po]) + even, ps])
                    rcr = abs(cr[row, col, 2 * (po + rinc[po]) + even, ps])

                    rpangle = lpangle = pangle = po

                    if po in right_hit:
                        rpcr = max(cr[row, col, 2 * (po + 1) + even, ps], 0)
                        rcr = min(rcr, rpcr)
                        rpangle = pangle + 1
                    if po in left_hit:
                        lpcr = max(cr[row, col, 2 * (po - 1) + even, ps], 0)
                        lcr = min(lcr, lpcr)
                        lpangle = pangle - 1

                    with np.errstate(invalid='ignore'):
                        if np.divide(rcr, rpcr) > np.divide(lcr, lpcr):
                            lcr = rpcr * lcr / lpcr
                            pangle = rpangle
                            pcr = rpcr
                        else:
                            rcr = lpcr * rcr / rpcr
                            pangle = lpangle
                            pcr = lpcr

                    with np.errstate(invalid='ignore', divide='ignore'):
                        orientations[row, col] = pangle + (rcr - lcr) / (2 * (pcr - min(rcr, lcr)))
                    if orientations[row, col] < 0:
                        orientations[row, col] += self.n_oris
                    if orientations[row, col] >= self.n_oris:
                        orientations[row, col] -= self.n_oris
                else:
                    orientations[row, col] = -1

        if positive_only:
            ridgeness[heights < 0] = 0
            orientations[heights < 0] = -1
        elif negative_only:
            ridgeness[heights > 0] = 0
            orientations[heights > 0] = -1

        return ridgeness, self._orientations_to_angles(orientations)

    def _orientation_incs(self):
        """Compute offsets for neighbouring orientations.

        For each shearlet in the system, the offset to reach a "neighbouring orientation" are returned as two lists.
        Also the borders between cones are returned.

        Used during orientation measurement.

        """
        cone_border1 = (1 << (self.shear_level - 2)) + 1
        cone_border2 = (self.n_oris) - (1 << (self.shear_level - 2))
        linc = [-1] * self.n_oris
        rinc = [1] * self.n_oris

        linc[0] = self.n_oris - 1
        linc[cone_border1] = -2
        linc[cone_border2] = -2

        rinc[self.n_oris - 1] = -1 * self.n_oris + 1
        rinc[cone_border1 - 1] = 2
        rinc[cone_border2 - 1] = 2
        rinc[cone_border2 - 1] = -4 if self.shear_level == 2 else 2

        return linc, rinc, cone_border1, cone_border2


class EdgeSystem(ShearletSystem):
    """A Shearlet-System of real-valued shearlets for edge measurement.

    During class creation a system of shearlets is constructed.

    Args:
        rows (int): Height of the constructed shearlet.
        cols (int): Width of the constructed shearlet.
        wavelet_eff_supp (Optional[int]): Effective support for wavelet function used in construction.

            Defaults to ``min(rows,cols) / 7``
        gaussion_eff_supp (Optional[int]): Effective support for Gauss function used in construction.

            Defaults to ``min(rows,cols) / 20``
        scales_per_octave (Optional[float]): Number of scales per octave.

            Defaults to ``2``.
        shear_level (Optional[int]): Amount of shearing applied.

            Defaults to ``3``.
        sampleWaveletOffOrigin (Optional[bool]): Sample the wavelet off-origin. Defaults to ``True``.
        normalize (Optional[bool]): Normalize shearlets during construction.

            Defaults to ``True``.
        scales: Number of scales of shearlets.

    Attributes:
        n_shearlets (int): Number of shearlets.
        rows (int): Height of shearlets.
        cols (int): Widthof shearlets.
        n_oris (int): Number of orientations.

        shearlets: Multidimensional array with shearlets in this system.

    """

    def __init__(self, rows, cols, wavelet_eff_supp=None,
                 gaussian_eff_supp=None, scales_per_octave=2,
                 shear_level=3, alpha=0.8, scales=None,
                 normalize=True, octaves=3.5, sampleWaveletOffOrigin=True):
        ShearletSystem.__init__(self, rows, cols, wavelet_eff_supp, gaussian_eff_supp,
                                scales_per_octave, shear_level, alpha, scales, normalize,
                                octaves)
        self.sampleWaveletOffOrigin = sampleWaveletOffOrigin

        self.n_shearlets = len(self.scales) * self.n_oris
        self.shearlets = np.zeros((self.rows, self.cols, self.n_shearlets), dtype=np.complex_)

        for j, scale in enumerate(self.scales):
            for ori in range(self.n_oris):
                shearlet = self._single_shearlet(scale, ori + 1)
                # Calculate normalization factor (on 'lower half' of shearlet)
                # for this scale
                # TODO: Normalization in frequency domain
                if self.normalize:
                    if ori == 0:
                        shearlet_t = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(shearlet)))
                        normalization = np.abs(np.sum(np.imag(shearlet_t[:, :(shearlet_t.shape[1] // 2)])))
                    with np.errstate(invalid='ignore'):
                        shearlet = shearlet / normalization

                self.shearlets[:, :, self.n_oris * j + ori] = shearlet

    def _single_shearlet(self, scale, ori):
        """Construct a single base shearlet for use in this edge detection system.

        For further information on parameters see the documentation of the function `construct_shearlet`.

        Args:
            scale: the scaling factor for this shearlet.
            ori: the orientation for this shearlet
            off_origin: if `True` this shearlet will be sampled off-origin.

        Returns:
            A shearlet to be used for transformations in an edge system.

        """
        shearlet_f = construct_shearlet(self.rows, self.cols, self.wavelet_eff_supp,
                                      self.gaussian_eff_supp, self.scales_per_octave,
                                      self.shear_level, self.alpha,
                                      self.sampleWaveletOffOrigin, scale, ori, self._coneh, self._ks)

        if ori in self._coneh:
            shearlet_f = shearlet_f + (self.hilbert_matrix[:, :, 0] * shearlet_f)
            # shearlet_f = np.fliplr(np.flipud(_hilbert_f(shearlet_f * -1)))
            # if not self.sampleWaveletOffOrigin:
            #     shearlet_f = np.roll(shearlet_f, -1, axis=0)
        else:
            if ori > np.max(self._coneh):
                shearlet_f = -1 * (shearlet_f + self.hilbert_matrix[:, :, 1] * shearlet_f)
                # shearlet_f = _hilbert_f(shearlet_f.T * -1).T
                # shearlet_f = np.roll(shearlet_f, 1, axis=1)
            else:
                shearlet_f = shearlet_f + self.hilbert_matrix[:, :, 1] * shearlet_f
                # shearlet_f = _hilbert_f(shearlet_f.T).T
        return shearlet_f

    def _full_circle_coeffs(self, coeffs):
        """Compute second semi-circle coefficients"""
        coeffs = np.reshape(coeffs, (self.rows, self.cols, self.n_oris, (self.n_shearlets // self.n_oris)),
                            order="F")
        coeffs = np.concatenate((coeffs, np.zeros_like(coeffs)), axis=2)
        for ori in range(self.n_oris):
            shift_first_circle = True
            if (ori + 1) in self._coneh:
                shift = -1
                axis = 0
            else:
                axis = 1
                if (ori + 1) > max(self._coneh):
                    shift = 1
                else:
                    shift_first_circle = False
                    shift = -1
            if shift_first_circle:
                coeffs[:, :, ori, :] = np.roll(coeffs[:, :, ori, :], -shift, axis=axis)
            coeffs[:, :, self.n_oris + ori, :] = np.roll(-1 * coeffs[:, :, ori, :], shift, axis=axis)
        return coeffs

    def detect(self, x, min_contrast=4, offset=1, pivoting_scales='all'):
        """Compute edges in image based on this shearlet system.

        The image to perform edge detection on has have the same shape that was used
        during creation of this EdgeSystem, e.g.

        >>> img = np.array([[255, 120], [120, 255]])
        >>> sys = EdgeSystem(*img.shape)
        >>> img.shape == sys.shape
        True
        >>> edges, orientations = sys.detect(img)

        Args:
            x: Image (2-dimensional array) to compute edges on.
            min_contrast (Optional[int]): Minimal contrast.
                This is a hard cutoff value during edge measurement.
            pivoting_scales (Optional[string]): Scales used during pivot search.
                One of ``all``, ``highest``, ``lowest``.

                Defaults to ``all``.

        Returns:
            An edge measurement of ``x`` with this system. The returned shape is equal to the image (and system) shape.
            An orientation measurement giving the angle of every edge detected by the system.

        """
        transformed = self._full_circle_coeffs(self._transform(x))
        offset = int(offset * self.scales_per_octave);

        edgeness = np.zeros_like(x, dtype=np.float)
        orientations = np.zeros_like(x, dtype=np.float)

        ci = np.imag(transformed[:, :, :, offset:])
        cr = np.abs(np.real(transformed[:, :, :, :(transformed.shape[3] - offset)]))

        if pivoting_scales == 'all':
            pivot_scales = np.array(range(ci.shape[3]))
        elif pivoting_scales == 'highest':
            pivot_scales = np.array([ci.shape[3] - 1])
        elif pivoting_scales == 'lowest':
            pivot_scales = np.array([1])
        else:
            raise AttributeError("Parameter pivoting_scales has to be one of 'all', 'lowest', 'highest'.")

        ci_pivot = ci[:, :, :, pivot_scales]
        max_pivot = np.argmax(np.reshape(ci_pivot,
                                         (ci_pivot.shape[0], ci_pivot.shape[1],
                                          ci_pivot.shape[2] * ci_pivot.shape[3]),
                                         order="F"
                                         ), axis=2)
        pivot_oris = np.mod(max_pivot, self.n_oris * 2)
        pivot_scales = pivot_scales[np.fix(((max_pivot) / (self.n_oris * 2))).astype(int)]

        scales = ci.shape[3]  # all scales
        linc, rinc, cone_border1, cone_border2 = self._orientation_incs()
        cone_border1 -= 1
        cone_border2 -= 1

        right_hit = [cone_border1, cone_border2, cone_border1 + self.n_oris, cone_border2 + self.n_oris]
        left_hit = [r + 1 for r in right_hit]

        for row in range(self.rows):
            for col in range(self.cols):
                # Edge detection
                po = pivot_oris[row, col]
                ps = pivot_scales[row, col]
                pci = ci[row, col, po, ps]
                for scale in range(scales):
                    edgeness[row, col] = edgeness[row, col] + min(ci[row, col, po, scale], pci) - cr[
                        row, col, po, scale] - min_contrast
                edgeness[row, col] = max(0, edgeness[row, col]) / (scales * pci)

                # Tangent orientation
                if edgeness[row, col] > 0:
                    rpangle = lpangle = pangle = po
                    lpci = pci
                    rpci = pci
                    lci = max(ci[row, col, po + linc[po], ps], 0)
                    rci = max(ci[row, col, po + rinc[po], ps], 0)

                    if po in right_hit:
                        rpci = max(ci[row, col, po + 1, ps], 0)
                        rci = min(rci, rpci)
                        rpangle = pangle + 1
                    if po in left_hit:
                        lpci = max(ci[row, col, po - 1, ps], 0)
                        lci = min(lci, lpci)
                        lpangle = pangle - 1

                    with np.errstate(invalid='ignore'):
                        if np.divide(rci, rpci) > np.divide(lci, lpci):
                            lci = (rpci * lci) / lpci
                            pangle = rpangle
                            pci = rpci
                        else:
                            rci = (lpci * rci) / rpci
                            pangle = lpangle
                            pci = lpci

                    with np.errstate(invalid='ignore', divide='ignore'):
                        orientations[row, col] = pangle + ((rci - lci) / (4 * pci - 2 * (rci + lci)))
                    if orientations[row, col] < 0:
                        orientations[row, col] += self.n_oris * 2
                    if orientations[row, col] >= (self.n_oris * 2):
                        orientations[row, col] -= self.n_oris * 2
                    if orientations[row, col] >= self.n_oris:
                        orientations[row, col] -= self.n_oris
                else:
                    orientations[row, col] = -1

        return edgeness, self._orientations_to_angles(orientations)

    def _orientation_incs(self):
        """Compute offsets for neighbouring orientations.

        For each shearlet in the system, the offset to reach a "neighbouring orientation" are returned as two lists.
        Also the borders between cones are returned.

        Used during orientation measurement.

        """
        cone_border1 = (1 << (self.shear_level - 2)) + 1
        cone_border2 = (self.n_oris) - (1 << (self.shear_level - 2))
        linc = [-1] * (self.n_oris * 2)
        rinc = [1] * (self.n_oris * 2)

        linc[0] = (self.n_oris * 2) - 1
        linc[cone_border1] = -2
        linc[cone_border2] = -2
        linc[cone_border1 + self.n_oris] = -2
        linc[cone_border2 + self.n_oris] = -2

        rinc[(self.n_oris * 2) - 1] = -2 * self.n_oris + 1
        rinc[cone_border1 - 1] = 2
        rinc[cone_border2 - 1] = 2
        rinc[cone_border1 + (self.n_oris) - 1] = 2
        rinc[cone_border2 + (self.n_oris) - 1] = -10 if self.shear_level == 2 else 2

        return linc, rinc, cone_border1, cone_border2



def _hilbert(x):
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    Xf = np.fft.fft2(x)
    x = np.fft.ifft2(_hilbert_f(Xf))
    return x


def _hilbert_f(Xf):
    N = Xf.shape[0]
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if len(Xf.shape) > 1:
        ind = [np.newaxis] * Xf.ndim
        ind[0] = slice(None)
        h = h[ind]
    return Xf * h
