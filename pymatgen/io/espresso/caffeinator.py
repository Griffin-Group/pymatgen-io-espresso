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

# TODO: Commented imports reserved for future updates.
#from __future__ import annotations

import warnings

import numpy as np

from pymatgen.core.structure import Structure
#from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import (
        Kpoints, 
        Poscar,
        )

#from pymatgen.io.espresso import utils
from pymatgen.io.espresso.inputs.pwin import (
        KPointsCard,
        AtomicSpeciesCard,
        AtomicPositionsCard,
        CellParametersCard,
        SystemNamelist,
        PWin,
        )


"""
Module-level functions for converting pmg's VASP input file objects 
to PWin-compatible cards and namelists.

caffeinate(vasp_in) returns: [relevant namelists], [relevant cards]
"""
def caffeinate(vasp_in, **kwargs):
    if isinstance(vasp_in, Kpoints):                                           
        return _caffeinate_kpoints(vasp_in)
    elif isinstance(vasp_in, Poscar):
        return _caffeinate_poscar(vasp_in, **kwargs)
    else:
        raise CaffeinationError(
                "Input file type not recognized (or not yet supported)"
                )

def _caffeinate_kpoints(kpoints):
    """
    Convert a Kpoints object to a KPointsCard object.

    NOTE: Cartesian coordinates are preserved in their original form, i.e. 
    in units of 2*pi/a where a is defined in an accompanying Poscar object.
    """
    if kpoints.style.name in ["Gamma","Monkhorst"]:
        option, grid, shift, k, weights, labels = _caffeinate_grid(kpoints)

    elif kpoints.style.name == "Line_mode":
        option, grid, shift, k, weights, labels = _caffeinate_linemode(kpoints)

    elif (
            kpoints.style.name in ["Reciprocal","Cartesian"] and 
            kpoints.num_kpts > 0
        ):
        option, grid, shift, k, weights, labels = _caffeinate_explicit(kpoints)

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

    if "tpiba" in str(option):
        # TODO: This warning can be ignored if a Poscar object is provided.
        # Need to add filtering in the Caffeinator methods.
        warnings.warn(
                (
                "\nWarning: VASP's cartesian coordinates cannot be fully "
                "converted to tpiba coordinates without an accompanying "
                "POSCAR file! Use the following k-points at your own risk."),
                CaffeinationWarning)
        #TODO: Make warning pretty

    #TODO: Return logic
    #come back to this post-Caffeinator
    return KPointsCard(
            option = option,
            grid = grid,
            shift = shift,
            k = k, 
            weights = weights,
            labels = labels)

def _caffeinate_grid(kpoints):
    if ( 
        all(int(x) == 1 for x in kpoints.kpts[0]) and 
        all(x == 0.0 for x in kpoints.kpts_shift) 
        ):
        opt_str = "gamma"
    else:
        opt_str = "automatic"
    option = KPointsCard.opts.from_string(opt_str)
    shift = [bool(x) for x in kpoints.kpts_shift]
    grid = []
    for i, x in enumerate(list(kpoints.kpts[0])):
        grid.append(int(x))
        if kpoints.style.name == "Gamma" and not x % 2:
            shift[i] = not shift[i]
    # TODO: Gamma-to-MP conversion needs testing!
    k = []
    weights = []
    labels = []
    return option, grid, shift, k, weights, labels

def _caffeinate_linemode(kpoints):
    if kpoints.coord_type.lower()[0] == "r":
        opt_str = "crystal_b"
    else:
        opt_str = "tpiba_b"
    k = [list(kpoints.kpts[0])]
    labels = [kpoints.labels[0]]
    weights = [kpoints.num_kpts]
    for i in range(1,len(kpoints.labels)):
        if kpoints.labels[i] == kpoints.labels[i - 1]:
            pass
        elif not i % 2:
            labels.append(kpoints.labels[i])
            weights[-1] = 1
            weights.append(kpoints.num_kpts)
            k.append(list(kpoints.kpts[i]))
        else:
            labels.append(kpoints.labels[i])
            weights.append(kpoints.num_kpts)
            k.append(list(kpoints.kpts[i]))
    weights[-1] = 1
    option = KPointsCard.opts.from_string(opt_str)
    grid = []
    shift = []
    return option, grid, shift, k, weights, labels

def _caffeinate_explicit(kpoints):
    if kpoints.num_kpts == 1 and all(int(x) == 0 for x in kpoints.kpts[0]):
        opt_str = "gamma"
    elif kpoints.style.name == "Cartesian":
        opt_str = "tpiba"
    else:
        opt_str = "crystal"
    option = KPointsCard.opts.from_string(opt_str)
    k = []
    labels = []
    for x in kpoints.kpts:
        k.append(list(x))
        labels.append("")
    weights = kpoints.kpts_weights
    grid = []
    shift = []
    if kpoints.tet_number != 0:
        warnings.warn(
                ("\nWarning: explicit tetrahedra are not compatible "
                "with PWscf and will not be preserved in the kpoints "
                "card."),
                CaffeinationWarning)
        #TODO: Make warning pretty
    #TODO:
    # Caffeinator can swap out the occupations tag for something else 
    # reasonable.
    # Define a unique warning category so that the two k-point warnings
    # defined in this module can be easily filtered?
    return option, grid, shift, k, weights, labels

def _caffeinate_poscar(poscar, **kwargs):
    """
    Convert a Poscar object to the following objects:
        - AtomicPositionsCard
        - AtomicSpeciesCard
        - CellParametersCard
        - Partially-initialized System namelist

    Keyword arguments:
        - ibrav: bool | False
          If set to True, choose the appropriate ibrav != 0
    """

    #TODO: clean this up
    ibrav = kwargs.get("ibrav", False)
    if ibrav not in [True, "True", "true", "T", "t"]:
        ibrav = False
    else:
        ibrav = True

    struct = poscar.structure

    system = SystemNamelist(
            {"nat":len(struct.species),
             "ntyp":len(species)})
    species = set(struct.species)

    lattice = struct.lattice
    #TODO: Check that lattices are always in Angstrom! (They probably are)

    if not ibrav:
        system["ibrav"] = 0
    else:
        raise CaffeinationError(
                "ibrav != 0 is not yet supported"
                )
        #TODO: Add lattice_to_ibrav to utils.py!
        #NOT YET IMPLEMENTED

    atomic_species = AtomicSpeciesCard(
            None,
            [str(s) for s in species],
            [s.atomic_mass for s in species],
            [f"{s}.upf" for s in species],
            )

    atomic_positions = AtomicPositionsCard(
            AtomicPositionsCard.opts.crystal,
            [str(s) for s in struct.species],
            struct.frac_coords,
            None,
            )

    cell_params = CellParametersCard(
            CellParametersCard.opts.angstrom,
            lattice.matrix[0],
            lattice.matrix[1],
            lattice.matrix[2],
            )

    #TODO: Return logic
    #come back to this post-Caffeinator
    return system, atomic_species, atomic_positions, cell_params


#class Caffeinator:
#    """
#    Class for converting VASP input sets to pwin objects. 
#    """
    # TODO: All of this

class CaffeinationError(Exception):
    """
    Exception class for caffeination
    """

class CaffeinationWarning(Warning):
    """
    Warning class for caffeination
    """
