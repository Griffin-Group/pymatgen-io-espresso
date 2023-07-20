"""
Classes for reading/manipulating projwfc.x/dos.x DOS files
"""

from __future__ import annotations

import itertools
import re
import os
import xml.etree.ElementTree as ET

import numpy as np
import glob
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

    def __init__(self, fildos, noncolinear=False):
        """
        Args:
            fildos (str): filepath to the dos file. Note that this is
                the same as in dos.x/projwfc.x input, so it shouldn't include the rest
                of the filename. For example, fildos="path/to/fildos" will look for
                "path/to/fildos.pdos_atm#_wfc#..." type of files
            noncolinear (bool): Whether the calculation is noncolinear WITHOUT SOC.
                                projwfc.x's output does not distinguish between
                                noncolinear without SOC and spin polarized calculations.
                                If False, the calculation is assumed to be spin polarized
                                (if the DOS file format leaves any ambiguity)
        """
        self.fildos = fildos
        self.atomic_states = []
        all_energies = []

        filenames = glob.glob(f"{fildos}.pdos_atm*")
        for f in filenames:
            states, E = self._get_atomic_states(f, noncolinear)
            self.atomic_states.extend(states)
            all_energies.append(E)

        if not np.allclose(all_energies, all_energies[0], rtol=0, atol=1e-4):
            raise ValueError("Energies from all files do not match")
        self.energies = all_energies[0]
        self.order_states()

    def order_states(self):
        """
        Sets the index (state #) of the states in the same order as the projwfc.x output
        """
        noncolinear = self.atomic_states[0].s_z is not None
        lspinorb = self.atomic_states[0].j is not None

        if lspinorb:
            sort_order = lambda x: (x.site.atom_i, x.wfc_i, x.l, x.j, x.mj)
        elif noncolinear:
            sort_order = lambda x: (x.site.atom_i, x.wfc_i, x.l, x.m, x.s_z)
        else:
            sort_order = lambda x: (x.site.atom_i, x.wfc_i, x.l, x.m)

        self.atomic_states = sorted(self.atomic_states, key=sort_order)
        for i, state in enumerate(self.atomic_states):
            state.state_i = i + 1

    @staticmethod
    def _get_atomic_states(filename, noncolinear):
        """
        Gets an AtomicState object for each atomic state in the file

        The filename format is pdos_atm#<atom_i>(<symbol>)_wfc#<wfc_i>(<spdf>[_<j>])

        The number/order of columns is as follows:
            First column is always energy in eV, the rest depend on the calculation:
                Colinear: 1 for LDOS, 1 for each PDOS_m
                Colinear spin pol or noncolinear no SOC: 2 for LDOS (u/d), 2 for ea PDOS_m (u/d)
                Noncolinear with SOC: 1 for LDOS, 1 for each PDOS_mj
            LDOS = \sum_m PDOS_m or \sum_mj PDOS_mj

        params:
            filename (str): filename of the pdos file
        returns:
            list of AtomicState objects
        """

        match = re.match(
            r".*?.pdos_atm#(\d+)\((\w+)\)_wfc#(\d+)\(([spdf])(?:_j(\d.5))*\)", filename
        )
        site = Site(match[2], [np.nan] * 3, properties={"atom_i": int(match[1])})
        wfc_i = int(match[3])
        l = OrbitalType[match[4]].value
        j = float(match[5]) if match[5] is not None else None

        data = np.loadtxt(filename, skiprows=1)
        energies = data[:, 0]
        if (ncols := data.shape[1]) == (2 + 2 * l + 1) and j is None:
            # Colinear case
            pdos = [{Spin.up: p} for p in data[:, 2:].T]
            params = [
                {"site": site, "wfc_i": wfc_i, "l": l, "m": m} for m in np.arange(1, 2 * l + 2)
            ]
        elif ncols == (1 + 2 * (1 + 2 * l + 1)) and j is None:
            # Colinear spin polarized or noncolinear without SOC
            if noncolinear:
                pdos = [{Spin.up: p} for p in data[:, 3:].T]
                params = [
                    {"site": site, "wfc_i": wfc_i, "l": l, "m": m, "s_z": s_z}
                    for m, s_z in itertools.product(np.arange(1, 2 * l + 2), [0.5, -0.5])
                ]
            else:
                pdos = [
                    {Spin.up: pu, Spin.down: pd} for pu in data[:, 3::2].T for pd in data[:, 4::2].T
                ]
                params = [
                    {"site": site, "wfc_i": wfc_i, "l": l, "m": m} for m in np.arange(1, 2 * l + 2)
                ]
        elif ncols == (2 + 2 * j + 1) and j is not None:
            # Noncolinear with SOC
            pdos = [{Spin.up: p} for p in data[:, 2:].T]
            params = [
                {"site": site, "wfc_i": wfc_i, "l": l, "j": j, "mj": mj}
                for mj in np.arange(-j, j + 1)
            ]
        else:
            raise ValueError(f"Unexpected number of columns in {filename}")

        atomic_states = [AtomicState(pm, pdos=pd) for pm, pd in zip(params, pdos)]

        return atomic_states, energies
