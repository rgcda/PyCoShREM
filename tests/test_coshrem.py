import unittest
import coshrem.shearletsystem
import scipy.io
import scipy.misc
import tests.util


class TestRidgeSystemConstruction(unittest.TestCase):
    # @unittest.skip("System File is too big to commit")
    def test_example(self):
        print("Example Ridge System")
        matdata = scipy.io.loadmat(
            "./tests/resources/RidgeSystem-Example.mat")
        system = coshrem.shearletsystem.RidgeSystem(512, 512, wavelet_eff_supp=60,
                                                    gaussian_eff_supp=20, scales_per_octave=4,
                                                    shear_level=3, alpha=0.2, octaves=3.5)
        error = tests.util.mse(matdata['sh'], system.shearlets)
        print("Mean squared error between Matlab and Python System: ", error)
        self.assertTrue(error.imag < 1)
        self.assertTrue(error.real < 1)


class TestRidgeDetection(unittest.TestCase):
    def test_b0(self):
        matdata = scipy.io.loadmat(
            "./tests/resources/B0-ridges.mat")
        system = coshrem.shearletsystem.RidgeSystem(512, 512, wavelet_eff_supp=60,
                                                    gaussian_eff_supp=20, scales_per_octave=4,
                                                    shear_level=3, alpha=0.2, octaves=3.5)
        img = scipy.misc.imread("./tests/resources/img/B0.png")
        ridges, oris = system.detect(img)
        error = tests.util.mse(matdata['ridges'], ridges)
        print("Mean squared error between Matlab and Python Ridges: ", error)
        self.assertTrue(error < 1)


@unittest.skip
class TestEdgeSystemConstruction(unittest.TestCase):
    def test_noNorm_noOffOrigin1(self):
        print("No normalization, no Off-Origin sampling")
        matdata = scipy.io.loadmat(
            "./tests/resources/EdgeSystem-200x200-noOffOrigin-noNormalization.mat")
        system = coshrem.shearletsystem.EdgeSystem(200, 200, normalize=False, sampleWaveletOffOrigin=False)
        error = tests.util.mse(matdata['shearlets'], system.shearlets)
        print("Mean squared error between Matlab and Python System: ", error)
        self.assertTrue(error.imag < float("1e-25"))
        self.assertTrue(error.real < float("1e-25"))

    def test_noNorm_noOffOrigin2(self):
        print("No normalization, no Off-Origin sampling")
        matdata = scipy.io.loadmat(
            "./tests/resources/EdgeSystem-128x128-noOffOrigin-noNormalization.mat")
        system = coshrem.shearletsystem.EdgeSystem(128, 128, normalize=False, sampleWaveletOffOrigin=False)
        error = tests.util.mse(matdata['shearlets'], system.shearlets)
        print("Mean squared error between Matlab and Python System: ", error)
        self.assertTrue(abs(error.imag) < float("1e-25"))
        self.assertTrue(abs(error.real) < float("1e-25"))

    def test_noOffOrigin(self):
        print("With normalization, no Off-Origin sampling")
        matdata = scipy.io.loadmat(
            "./tests/resources/EdgeSystem-128x128-noOffOrigin.mat")
        system = coshrem.shearletsystem.EdgeSystem(128, 128, sampleWaveletOffOrigin=False)
        error = tests.util.mse(matdata['shearlets'], system.shearlets)
        print("Mean squared error between Matlab and Python System: ", error)
        self.assertTrue(abs(error.imag) < float("1e-25"))
        self.assertTrue(abs(error.real) < float("1e-25"))

    def test_Norm_offOrigin(self):
        print("With normalization, with Off-Origin sampling")
        matdata = scipy.io.loadmat(
            "./tests/resources/EdgeSystem-128x128.mat")
        system = coshrem.shearletsystem.EdgeSystem(128, 128)
        error = tests.util.mse(matdata['shearlets'], system.shearlets)
        print("Mean squared error between Matlab and Python System: ", error)
        self.assertTrue(abs(error.imag) < float("1e-25"))
        self.assertTrue(abs(error.real) < float("1e-25"))

    def test_Norm_offOrigin_oddDims(self):
        print("Odd dimensional system. With normalization, with Off-Origin sampling")
        matdata = scipy.io.loadmat(
            "./tests/resources/EdgeSystem-501x400.mat")
        system = coshrem.shearletsystem.EdgeSystem(501, 400)
        error = tests.util.mse(matdata['shearlets'], system.shearlets)
        print("Mean squared error between Matlab and Python System: ", error)
        self.assertTrue(abs(error.imag) < float("1e-25"))
        self.assertTrue(abs(error.real) < float("1e-25"))


@unittest.skip
class TestEdgeDetection(unittest.TestCase):
    def test_lena(self):
        matdata = scipy.io.loadmat(
            "./tests/resources/lena-edges.mat")
        img = scipy.misc.imread("./tests/resources/img/lena.jpg")
        system = coshrem.shearletsystem.EdgeSystem(*img.shape)
        edges, oris = system.detect(img)
        edge_error = tests.util.mse(matdata['edges'], edges)
        print("MSE between Matlab and Python Edges: ", edge_error)
        self.assertTrue(abs(edge_error) < 1)


if __name__ == '__main__':
    unittest.main()
