"""
Convert VASP inputs to PWSCF inputs.

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
# TODO: Stylistic updates
    # - Useful to leave module-level functions?

from __future__ import annotations

import contextlib
import warnings

import numpy as np

from pymatgen.core.structure import Structure

from pymatgen.io.vasp.inputs import Kpoints, Poscar
from pymatgen.io.espresso.inputs.pwin import (
        KPointsCard,
        AtomicSpeciesCard,
        AtomicPositionsCard,
        CellParametersCard,
        PWin
        )


"""
Module-level functions for converting pmg's VASP input file objects 
to PWin-compatible cards
"""
def caffeinate(vasp_in):
    if isinstance(vasp_in, Kpoints):                                           
        return _caffeinate_kpoints(vasp_in)
    elif isinstance(vasp_in, Poscar):
        return _caffeinate_poscar(vasp_in)
    else:
        raise CaffeinationError(
                "Input file type not recognized (or not yet supported)"
                )

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

    if k_style.name in ["Gamma","Monkhorst"]:
        if ( 
            all(int(x) == 1 for x in k_pts[0]) and 
            all(x == 0.0 for x in k_shift) 
            ):
            opt_str = "gamma"
        else:
            opt_str = "automatic"

        # TODO: option assignment from string should be fixed in next version
        option = KPointsCard.opts.from_string(opt_str)
        shift = [bool(x) for x in k_shift]
        grid = []
        for i, x in enumerate(list(k_pts[0])):
            grid.append(int(x))
            if k_style.name == "Gamma" and not x % 2:
                shift[i] = not shift[i]
        # TODO: Gamma-to-MP conversion needs testing!
        pw_k = []
        pw_wts = []
        pw_lbls = []

    elif k_style.name == "Line_mode":
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

    elif k_style.name in ["Reciprocal","Cartesian"] and k_num > 0:
        if k_num == 1 and (
                all(int(x) == 1 for x in k_pts[0]) and
                all(x == 0.0 for x in k_shift)
                ):
            opt_str = "gamma"
        elif k_style.name == "Cartesian":
            opt_str = "tpiba"
        else:
            opt_str = "crystal"

        option = KPointsCard.opts.from_string(opt_str)
        pw_k = [list(x) for x in k_pts]
        pw_wts = k_wts
        pw_lbls = []
        grid = []
        shift = []
        #TODO: Debug this!
        #kpoints card does not include the explicit k-points

        #TODO:
        # Need to define a warning for explicit tetrahedra.
        # Caffeinator can preserve the warning and swap out the occupations
        # tag for something else reasonable.

    else:
        raise CaffeinationError(
                ("\nConversion of generalized regular grids or fully-automatic "
                "grids is not currently implemented. "
                "Please use one of the following KPOINTS file types:\n"
                "  - Gamma-centered\n"
                "  - Monkhorst-Pack\n"
                "  - Explicit mesh\n"
                "  - Line-mode")
                )

    if "tpiba" in opt_str:
        # TODO: This warning can be ignored if a Poscar object is provided.
        # Need to add filtering in the Caffeinator methods.
        warnings.warn(
                (
                "\nWarning: VASP's cartesian coordinates cannot be fully "
                "converted to tpiba coordinates without an accompanying "
                "POSCAR file! Use the following k-points at your own risk."),
                CaffeinationWarning,
                stacklevel=10)
        #TODO: Would prefer a cleaner way of suppressing the stacktrace
        #than messing with the stacklevel this way

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


class Caffeinator():
    """
    Class for converting VASP input sets to pwin objects.
    Caffeinator 
    """
    # TODO: All of this

    def __init__(self):
        pass
    # Need to initialize with a complete VASP input set
    # and optional settings for the QE namelists

    @classmethod
    def get_pwin(cls, vasp_dir: str):
        """
        Parse a directory containing a VASP input set
        and produce a PWin object
        """
        return cls()


class CaffeinationError(Exception):
    """
    Exception class for caffeination
    """

class CaffeinationWarning(Warning):
    """
    Warning class for caffeination
    """
