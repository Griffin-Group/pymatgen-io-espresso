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
# from __future__ import annotations

import warnings

import numpy as np

# from pymatgen.io.espresso import utils
from pymatgen.io.espresso.inputs.pwin import (
    AtomicPositionsCard,
    AtomicSpeciesCard,
    CellParametersCard,
    KPointsCard,
    SystemNamelist,
)

# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import (
    Kpoints,
    Poscar,
)

"""
Module-level functions for converting pmg's VASP input file objects 
to PWin-compatible cards and namelists.

caffeinate(vasp_in) currently returns a tuple of relevant namelists and cards.
This may be updated in the future. 
"""


def caffeinate(vasp_in, **kwargs):
    if isinstance(vasp_in, Kpoints):
        return _caffeinate_kpoints(vasp_in)
    elif isinstance(vasp_in, Poscar):
        try:
            return _caffeinate_poscar(vasp_in, ibrav=kwargs.get("ibrav", False))
        except TypeError as e:
            raise CaffeinationError(
                "Could not parse boolean keyword argument 'ibrav'"
            ) from e
    else:
        raise CaffeinationError("Input file type not recognized (or not yet supported)")


def _caffeinate_kpoints(kpoints):
    """
    Convert a Kpoints object to a KPointsCard object.

    NOTE: Cartesian coordinates are preserved in their original form, i.e.
    in units of 2*pi/a where a is defined in an accompanying Poscar object.
    """
    grid, shift, k, weights, labels = [], [], [], [], []
    if kpoints.style.name in ["Gamma", "Monkhorst"]:
        option, grid, shift = _convert_grid_k(kpoints)

    elif kpoints.style.name == "Line_mode":
        option, k, weights, labels = _convert_linemode_k(kpoints)

    elif kpoints.style.name in ["Reciprocal", "Cartesian"] and kpoints.num_kpts > 0:
        option, k, weights, labels = _convert_explicit_k(kpoints)
    else:
        raise CaffeinationError(
            (
                "\nConversion of generalized regular grids or fully-automatic "
                "grids is not currently implemented. "
                "Please use one of the following KPOINTS file types:\n"
                "  - Gamma-centered\n"
                "  - Monkhorst-Pack\n"
                "  - Explicit mesh\n"
                "  - Line-mode"
            )
        )
    if "tpiba" in str(option):
        warnings.warn(
            (
                "\nWarning: VASP's cartesian coordinates cannot be fully "
                "converted to tpiba coordinates without an accompanying "
                "POSCAR file! Use the following k-points at your own risk."
            ),
            CartesianWarning,
        )
        # TODO: Make warning pretty

    # TODO: Return logic
    # come back to this post-Caffeinator
    return KPointsCard(option, grid, shift, k, weights, labels)


def _convert_grid_k(kpoints):
    if all(int(x) == 1 for x in kpoints.kpts[0]) and all(
        x == 0.0 for x in kpoints.kpts_shift
    ):
        return KPointsCard.opts.from_string("gamma"), [], []
    option = KPointsCard.opts.from_string("automatic")
    shift = [bool(x) for x in kpoints.kpts_shift]
    grid = []
    for i, x in enumerate(list(kpoints.kpts[0])):
        grid.append(int(x))
        if kpoints.style.name == "Gamma" and not x % 2:
            shift[i] = not shift[i]
    # TODO: Gamma-to-MP conversion needs testing!
    return option, grid, shift


def _convert_linemode_k(kpoints):
    opt_str = "crystal_b" if kpoints.coord_type.lower()[0] == "r" else "tpiba_b"
    k = [list(kpoints.kpts[0])]
    labels = [kpoints.labels[0]]
    weights = [kpoints.num_kpts]
    for i in range(1, len(kpoints.labels)):
        if kpoints.labels[i] == kpoints.labels[i - 1]:
            continue
        if not i % 2:
            weights[-1] = 1
        labels.append(kpoints.labels[i])
        weights.append(kpoints.num_kpts)
        k.append(list(kpoints.kpts[i]))
    weights[-1] = 1
    option = KPointsCard.opts.from_string(opt_str)
    return option, k, weights, labels


def _convert_explicit_k(kpoints):
    if kpoints.num_kpts == 1 and all(int(x) == 0 for x in kpoints.kpts[0]):
        return KPointsCard.opts.from_string("gamma"), [], [], []
    elif kpoints.style.name == "Cartesian":
        opt_str = "tpiba"
    else:
        opt_str = "crystal"
    option = KPointsCard.opts.from_string(opt_str)
    k = np.array(kpoints.kpts)
    labels = [""] * kpoints.num_kpts
    weights = kpoints.kpts_weights
    if kpoints.tet_number != 0:
        warnings.warn(
            (
                "\nWarning: explicit tetrahedra are not compatible "
                "with PWscf and will not be preserved in the kpoints "
                "card."
            ),
            CaffeinationWarning,
        )
        # TODO: Make warning pretty
    return option, k, weights, labels


def _caffeinate_poscar(poscar, ibrav: bool = False):
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
    struct = poscar.structure
    species = set(struct.species)
    system = SystemNamelist({"nat": len(struct.species), "ntyp": len(species)})
    lattice = struct.lattice
    if not ibrav:
        system["ibrav"] = 0
    else:
        raise CaffeinationError("ibrav != 0 is not yet supported")
        # TODO: Add lattice_to_ibrav to utils.py!
        # NOT YET IMPLEMENTED
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
    # TODO: Return logic
    # come back to this post-Caffeinator
    return system, atomic_species, atomic_positions, cell_params


# class Caffeinator:
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


class CartesianWarning(CaffeinationWarning):
    """
    Warning class for tpiba conversion
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return str(self.message)
