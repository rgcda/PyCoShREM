import unittest

import numpy as np
import coshrem.shearlet


class TestUtil(unittest.TestCase):

    def test_yapuls(self):
        npuls = 20
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10,
                             -9, -8, -7, -6, -5, -4, -3, -2, -1])
        self.assertTrue(np.array_equal(coshrem.shearlet.yapuls(npuls),
                                       expected * (2*np.pi/npuls)))
