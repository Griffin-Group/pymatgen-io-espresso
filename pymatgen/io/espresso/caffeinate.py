"""
Classes for converting VASP inputs to PWSCF inputs.

Supported VASP inputs:
    KPOINTS*
    POSCAR
Not (yet) supported:
    INCAR
    KPOINTS: generalized regular grids or
             fully automatic (KSPACING) grids
"""

# TODO: imports need linting!

from __future__ import annotations

import contextlib
import warnings
from copy import copy

import numpy as np

from pymatgen.core.structure import Structure

from pymatgen.io.vasp.inputs import Kpoints, Poscar

from pymatgen.io.espresso.inputs import pwin


def caffeinate(vasp_in):
    if isinstance(vasp_in, Kpoints):
        return _caffeinate_kpoints(vasp_in)
    elif isinstance(vasp_in, Poscar):
        return _caffeinate_poscar(vasp_in)
    else:
        # TODO: Define a warning
        return vasp_in

def _caffeinate_kpoints(kpoints):
    """
    Convert a Kpoints object to a KPointsCard object.

    NOTE: Cartesian coordinates are preserved in their
    original form, i.e. in units of 2*pi/a where a is
    defined in an accompanying Poscar object.
    """
    k_style = kpoints.style
    k_num = kpoints.num_kpts
    k_pts = kpoints.kpts
    k_shift = kpoints.kpts_shift
    k_wts = kpoints.kpts_weights
    k_coord = kpoints.coord_type
    k_lab = kpoints.labels

    if k_style.name.lower()[0] in "gm":
        if ( 
            all(int(x) == 1 for x in k_pts[0]) and 
            all(x == 0.0 for x in k_shift) 
            ):
            option = "gamma"
        else:
            option = "automatic"


        # QE generates Monkhorst-Pack grids.
        # Need to convert a gamma-centered VASP grid into
        # an MP grid with a shift (TODO)


    elif k_style.name.lower()[0] == "l":
        if k_coord.lower()[0] == "r":
            option = "crystal_b"
        else:
            option = "tpiba_b"
    elif k_style.name.lower()[0] == "r" and k_num > 0:
        option = "crystal"
        # TODO: check for gamma
    elif k_style.name.lower()[0] == "c" and k_num > 0:
        option = "tpiba"
        # TODO: check for gamma
    else:
        # In this case the style is "a" (automatic),
        # corresponding to either a fully-automatic grid
        # with a defined spacing (officially deprecated)
        # or to a generalized regular grid.
        # Neither option has been implemented yet.
        # TODO: Define a warning 
        return kpoints

    if "tpiba" in option:
        # TODO: Define a warning.
        # Without an accompanying POSCAR, VASP's cartesian coordinates
        # cannot properly be converted to PWSCF's tpiba coordinates.
        # This warning can be ignored in pwin if a Poscar object is
        # provided.
        pass

    # DEBUGGING (TODO)
    return option

# TODO:
#def _caffeinate_poscar(poscar):
    """
    Convert a Poscar object to the following three objects:
        - AtomicPositionsCard
        - AtomicSpeciesCard
        - CellParametersCard
    """
