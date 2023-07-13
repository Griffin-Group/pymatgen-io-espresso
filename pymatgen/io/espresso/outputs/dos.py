"""
Classes for reading/manipulating projwfc.x/dos.x DOS files
"""

from __future__ import annotations

import itertools
import re
import xml.etree.ElementTree as ET

import numpy as np
from monty.json import MSONable
import pandas as pd
from tabulate import tabulate

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure, Site
from pymatgen.core.units import (
    Ry_to_eV,
    bohr_to_ang,
)

from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.io.espresso.outputs.projwfc import AtomicState
from pymatgen.io.espresso.utils import parse_pwvals, ibrav_to_lattice, projwfc_orbital_to_vasp

class Dos(MSONable):
    """
    Class for representing DOS data from a PWscf dos.x or projwfc.x calculation
    """
    def __init__(self, fildos):
        """
        Args:
            fildos (str): filepath to the dos file. Note that this is
                the same as in dos.x/projwfc.x input, so it shouldn't include the rest
                of the filename. For example, fildos="path/to/fildos" will look for
                "path/to/fildos.pdos_atm#_wfc#..." type of files
        """
        pass