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
# TODO: Gamma vs. M-P conversion should be tested with an actual VASP/QE comp.

from __future__ import annotations

import contextlib
import warnings

import numpy as np

from pymatgen.core.structure import Structure

from pymatgen.io.vasp.inputs import Kpoints, Poscar
from pymatgen.io.espresso.inputs.pwin import KPointsCard

# TODO
# introduce class Caffeinator():
    # TODO: define get_pwin method 
    # which reads in a full set of VASP inputs 
    # and returns a full pwin object

# Functions for converting a specified VASP input file
# to PWin format

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

    NOTE: Cartesian coordinates are preserved in their original form, 
    i.e. in units of 2*pi/a where a is defined in an accompanying Poscar object.
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
            opt_str = "gamma"
        else:
            opt_str = "automatic"

        # TODO: option assignment from string should be fixed in next version
        option = KPointsCard.opts.from_string(opt_str)
        grid = [ int(x) for x in k_pts[0] ]
        shift = [ bool(x) for x in k_shift ]

        # TODO: needs checking
        # convert gamma-centered grids to Monkhorst-Pack grids, if necessary
        # (i.e. for even subdivisions)
        if k_style.name.lower()[0] == "g":
            for i, x in enumerate(grid):
                if not x % 2:
                    shift[i] = not shift[i]

        pw_k = []
        pw_wts = []
        pw_lbls = []

    elif k_style.name.lower()[0] == "l":
        if k_coord.lower()[0] == "r":
            opt_str = "crystal_b"
        else:
            opt_str = "tpiba_b"

        pw_k = [list(k_pts[0])]
        pw_lbls = [k_lab[0]]
        pw_wts = [k_num]
        for i in range(1,len(k_lab)):
            if k_lab[i] == k_lab[i - 1]:
                pass
            elif not i % 2:
                pw_lbls.append(k_lab[i])
                pw_wts[-1] = 1
                pw_wts.append(k_num)
                pw_k.append(list(k_pts[i]))
            else:
                pw_lbls.append(k_lab[i])
                pw_wts.append(k_num)
                pw_k.append(list(k_pts[i]))

        pw_wts[-1] = 1
        option = KPointsCard.opts.from_string(opt_str)
        grid = []
        shift = []

    elif k_style.name.lower()[0] in "rc" and k_num > 0:
        if k_num == 1 and (
                all(int(x) == 1 for x in k_pts[0]) and
                all(x == 0.0 for x in k_shift)
                ):
            opt_str = "gamma"
        elif k_style.name.lower()[0] == "c":
            opt_str = "tpiba"
        else:
            opt_str = "crystal"

        option = KPointsCard.opts.from_string(opt_str)
        grid = []
        shift = []
        pw_k = []
        pw_wts = []
        pw_lbls = []
        # TODO: finish parsing k-points 
    

    else:
        # In this case the style we have either a fully-automatic grid with a 
        # defined spacing (officially deprecated by VASP) or a generalized 
        # regular grid.
        # Neither option has a direct PWSCF parallel and conversion has not
        # been implemented yet.
        # TODO: Define a warning 
        return kpoints

    if "tpiba" in opt_str:
        # TODO: Define a warning.
        # Without an accompanying POSCAR, VASP's cartesian coordinates
        # cannot properly be converted to PWSCF's tpiba coordinates.
        # This warning can be ignored in pwin if a Poscar object is
        # provided.
        pass

    # DEBUGGING (TODO)
    return KPointsCard(
            option = option,
            grid = grid,
            shift = shift,
            k = pw_k, 
            weights = pw_wts,
            labels = pw_lbls)

# TODO:
def _caffeinate_poscar(poscar):
    """
    Convert a Poscar object to the following three objects:
        - AtomicPositionsCard
        - AtomicSpeciesCard
        - CellParametersCard
    """
    pass

