import unittest

from pymatgen.io.espresso.pwscf import myfunc


class FuncTest(unittest.TestCase):
    def test_myfunc(self):
        self.assertEqual(myfunc(1, 1), 2)
