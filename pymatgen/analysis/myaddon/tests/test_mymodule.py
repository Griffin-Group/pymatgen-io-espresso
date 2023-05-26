import unittest

from pymatgen.analysis.myaddon.mymodule import myfunc


class FuncTest(unittest.TestCase):
    def test_myfunc(self):
        self.assertEqual(myfunc(1, 1), 2)
