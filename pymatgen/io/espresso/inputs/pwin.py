"""
Classes for reading/manipulating PWscf xml files.
"""


from __future__ import annotations

import contextlib
import warnings
from copy import copy

import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.units import bohr_to_ang

from pymatgen.io.espresso.utils import ibrav_to_lattice
from pymatgen.io.espresso.inputs.base import (
    BaseInputFile,
    InputCard,
    InputNamelist,
    CardOptions,
    SupportedInputs,
)


class ControlNamelist(InputNamelist):
    """&CONTROL namelist"""

    name = "control"
    required = True


class SystemNamelist(InputNamelist):
    """&SYSTEM namelist"""

    name = "system"
    required = True


class ElectronsNamelist(InputNamelist):
    """&ELECTRONS namelist"""

    name = "electrons"
    required = True


class IonsNamelist(InputNamelist):
    """&IONS namelist"""

    name = "ions"
    required = False


class CellNamelist(InputNamelist):
    """&CELL namelist"""

    name = "cell"
    required = False


class FcpNamelist(InputNamelist):
    """&FCP namelist"""

    name = "fcp"
    required = False


class RismNamelist(InputNamelist):
    """&RISM namelist"""

    name = "rism"
    required = False


class AtomicSpeciesCard(InputCard):
    """ATOMIC_SPECIES card"""

    name = "atomic_species"
    required = True
    opts = None
    default_option = None
    default_deprecated = False

    def __init__(self, option, symbols, masses, files):
        self.option = option
        self.symbols = symbols
        self.masses = masses
        self.files = files

    def get_body(self, indent):
        return "".join(
            f"\n{indent}{symbol:>3} {self.masses[i]:>10.6f} {self.files[i]}"
            for i, symbol in enumerate(self.symbols)
        )

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        symbols = [item[0] for item in body]
        masses = [item[1] for item in body]
        files = [item[2] for item in body]
        return cls(option, symbols, masses, files)


class AtomicPositionsCard(InputCard):
    """ATOMIC_POSITIONS card"""

    class AtomicPositionsOptions(CardOptions):
        alat = "alat"
        bohr = "bohr"
        angstrom = "angstrom"
        crystal = "crystal"
        crystal_sg = "crystal_sg"

    name = "atomic_positions"
    required = True
    opts = AtomicPositionsOptions
    default_option = opts.alat
    default_deprecated = True

    def __init__(self, option, symbols, positions):
        self.option = option
        self.symbols = symbols
        self.positions = positions

    def get_body(self, indent):
        return "".join(
            f"\n{indent}{symbol:>3} {self.positions[i][0]:>13.10f} {self.positions[i][1]:>13.10f} {self.positions[i][2]:>13.10f}"
            for i, symbol in enumerate(self.symbols)
        )

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        symbols = [line[0] for line in body]
        positions = [np.array(line[1:]) for line in body]
        return cls(option, symbols, positions)


class KPointsCard(InputCard):
    """K_POINTS card"""

    class KPointsOptions(CardOptions):
        automatic = "automatic"
        gamma = "gamma"
        tpiba = "tpiba"
        crystal = "crystal"
        tpiba_b = "tpiba_b"
        crystal_b = "crystal_b"
        tpiba_c = "tpiba_c"
        crystal_c = "crystal_c"

    name = "k_points"
    required = True
    opts = KPointsOptions
    default_option = opts.tpiba
    default_deprecated = False

    def __init__(self, option, grid, shift, k, weights, labels):
        self.option = option
        self.grid = grid
        self.shift = shift
        self.k = k
        self.weights = weights
        self.labels = labels

    def get_body(self, indent):
        if self.option == self.opts.automatic:
            body = (
                f"\n{indent}{self.grid[0]:>3}"
                f" {self.grid[1]:>3} {self.grid[2]:>3}"
                f" {int(self.shift[0]):>3}"
                f" {int(self.shift[1]):>3}"
                f" {int(self.shift[2]):>3}"
            )
        elif self.option != self.opts.gamma:
            body = f"\n{len(self.k)}"
            for k, w, l in zip(self.k, self.weights, self.labels):
                body += f"\n{indent}{k[0]:>13.10f} {k[1]:>13.10f} {k[2]:>13.10f}"
                body += f" {w:>4}" if w == int(w) else f" {w:>10.6f}"
                body += f" ! {l}" if l else ""
        return body

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        grid, shift, k, weights, labels = [], [], [], [], []
        if option == cls.opts.automatic:
            grid, shift = body[0][:3], [bool(s) for s in body[0][3:]]
        elif option != cls.opts.gamma:
            for line in body[1:]:
                k.append(line[:3])
                weights.append(line[3])
                labels.append(" ".join(line[4:]).strip("!").lstrip() if len(line) > 4 else "")

        return cls(option, grid, shift, k, weights, labels)

    @property
    def line_mode(self):
        """Whether the k-points are in line mode"""
        return self.option in [
            self.opts.crystal,
            self.opts.crystal_b,
            self.opts.tpiba,
            self.opts.tpiba_b,
        ]

    @property
    def band_mode(self):
        """Whether the k-points are in band mode"""
        return self.option in [
            self.opts.crystal_b,
            self.opts.tpiba_b,
        ]

    @property
    def coords_are_cartesian(self):
        """Whether the k-points are in cartesian coordinates"""
        return self.option in [
            self.opts.tpiba,
            self.opts.tpiba_b,
            self.opts.tpiba_c,
        ]


class AdditionalKPointsCard(InputCard):
    """ADDITIONAL_K_POINTS card"""

    class AdditionalKPointsOptions(CardOptions):
        tpiba = "tpiba"
        crystal = "crystal"
        tpiba_b = "tpiba_b"
        crystal_b = "crystal_b"
        tpiba_c = "tpiba_c"
        crystal_c = "crystal_c"

    name = "additional_k_points"
    required = False
    opts = AdditionalKPointsOptions
    default_option = opts.tpiba
    default_deprecated = False

    def __init__(self, option, k, weights, labels):
        self.option = option
        self.k = k
        self.weights = weights
        self.labels = labels

    def get_body(self, indent):
        body = f"\n{len(self.k)}"
        for k, w, l in zip(self.k, self.weights, self.labels):
            body += f"\n{indent}{k[0]:>13.10f} {k[1]:>13.10f} {k[2]:>13.10f}"
            body += f" {w:>4}" if w == int(w) else f" {w:>10.6f}"
            body += f" ! {l}" if l else ""
        return body

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        k, weights, labels = [], [], []
        for line in body[1:]:
            k.append(line[:3])
            weights.append(line[3])
            labels.append(" ".join(line[4:]).strip("!").lstrip() if len(line) > 4 else "")

        return cls(option, k, weights, labels)


class CellParametersCard(InputCard):
    """CELL_PARAMETERS card"""

    class CellParametersOptions(CardOptions):
        alat = "alat"
        bohr = "bohr"
        angstrom = "angstrom"

    name = "cell_parameters"
    required = False
    opts = CellParametersOptions
    default_option = opts.alat
    default_deprecated = True

    def __init__(self, option, a1, a2, a3):
        self.option = option
        self.a1, self.a2, self.a3 = a1, a2, a3

    def get_body(self, indent):
        body = f"\n{indent}{self.a1[0]:>13.10f}" f" {self.a1[1]:>13.10f}" f" {self.a1[2]:>13.10f}"
        body += f"\n{indent}{self.a2[0]:>13.10f}" f" {self.a2[1]:>13.10f}" f" {self.a2[2]:>13.10f}"
        body += f"\n{indent}{self.a3[0]:>13.10f}" f" {self.a3[1]:>13.10f}" f" {self.a3[2]:>13.10f}"
        return body

    @classmethod
    def from_string(cls, s: str):
        """Parse a string containing an ATOMIC_SPECIES card"""
        option, body = cls.split_card_string(s)
        a1, a2, a3 = map(np.array, body)
        return cls(option, a1, a2, a3)


class ConstraintsCard(InputCard):
    """CONSTRAINTS card (not fully implemented)"""

    name = "constraints"
    required = False
    opts = None
    default_option = None
    default_deprecated = False


class OccupationsCard(InputCard):
    """OCCUPATIONS card (not fully implemented)"""

    name = "occupations"
    required = False
    opts = None
    default_option = None
    default_deprecated = False


class AtomicVelocitiesCard(InputCard):
    """ATOMIC_VELOCITIES card (not fully implemented)"""

    class AtomicVelocitiesOptions(CardOptions):
        au = "a.u."

    name = "atomic_velocities"
    required = False
    opts = AtomicVelocitiesOptions
    # TODO: this card *requires* an option, it has no default
    default_option = opts.au
    default_deprecated = True


class AtomicForcesCard(InputCard):
    """ATOMIC_FORCES card (not fully implemented)"""

    name = "atomic_forces"
    required = False
    opts = None
    default_option = None
    default_deprecated = False


class SolventsCard(InputCard):
    """SOLVENTS card (not fully implemented)"""

    class SolventsOptions(CardOptions):
        cell = "1/cell"
        molL = "mol/L"
        gcm3 = "g/cm^3"

    name = "solvents"
    required = False
    opts = SolventsOptions
    # TODO: this card *requires* an option, it has no default
    default_option = None
    default_deprecated = False


class HubbardCard(InputCard):
    """HUBBARD card (not fully implemented)"""

    class HubbardOptions(CardOptions):
        atomic = "atomic"
        othoatomic = "ortho-atomic"
        normatomic = "norm-atomic"
        wf = "wf"
        pseudo = "pseudo"

    name = "hubbard"
    required = False
    opts = HubbardOptions
    # TODO: this card *requires* an option, it has no default
    default_option = opts.atomic
    default_deprecated = True


class PWin(BaseInputFile):
    """
    Class for PWscf input files
    """

    class PWinCards(SupportedInputs):
        atomic_species = AtomicSpeciesCard
        atomic_positions = AtomicPositionsCard
        k_points = KPointsCard
        additional_k_points = AdditionalKPointsCard
        cell_parameters = CellParametersCard
        constraints = ConstraintsCard
        occupations = OccupationsCard
        atomic_velocities = AtomicVelocitiesCard
        atomic_forces = AtomicForcesCard
        solvents = SolventsCard
        hubbard = HubbardCard

    class PWinNamelists(SupportedInputs):
        control = ControlNamelist
        system = SystemNamelist
        electrons = ElectronsNamelist
        ions = IonsNamelist
        cell = CellNamelist
        fcp = FcpNamelist
        rism = RismNamelist

    card_classes = PWinCards
    namelist_classes = PWinNamelists

    @property
    def site_symbols(self):
        """
        The list of site symbols in the input file
        (i.e., the atomic_species card)
        """
        return list({site.species_string for site in self.structure})

    @property
    def structure(self):
        """
        Returns:
            Structure object
        """
        # TODO: move to validate
        if self.atomic_positions is None:
            raise ValueError("ATOMIC_POSITIONS card is missing.")
        option = self.atomic_positions.option
        species = atomic_positions.symbols
        coords = np.array(atomic_positions.positions)
        if option == AtomicPositionsCard.opts.alat:
            coords *= self.alat
            coords_are_cartesian = True
        elif option == AtomicPositionsCard.opts.bohr:
            coords *= bohr_to_ang
            coords_are_cartesian = True
        elif option == AtomicPositionsCard.opts.angstrom:
            coords_are_cartesian = True
        elif option == AtomicPositionsCard.opts.crystal:
            coords_are_cartesian = False
        elif option == AtomicPositionsCard.opts.crystal_sg:
            raise ValueError("Atomic positions with crystal_sg option are not supported.")

        return Structure(self.lattice, species, coords, coords_are_cartesian=coords_are_cartesian)

    @structure.setter
    def structure(self, structure):
        """
        Args:
            structure (Structure): Structure object to replace the current structure with
        """
        # self._validate()
        self.lattice = structure.lattice
        if self.system is None:
            self.system = SystemNamelist()
        self.system["nat"] = len(structure.species)
        species = set(structure.species)
        self.system["ntyp"] = len(species)

        if self.atomic_species is not None and set(self.atomic_species.symbols) != {
            str(s) for s in species
        }:
            warnings.warn(
                "The atomic species in the input file does not "
                "match the species in the structure object. "
                "The atomic species in the input file will be overwritten."
            )
            self.atomic_species = AtomicSpeciesCard(
                None,
                [str(s) for s in species],
                [s.atomic_mass for s in species],
                [f"{s}.UPF" for s in species],
            )
        self.atomic_positions = AtomicPositionsCard(
            AtomicPositionsCard.opts.crystal,
            [str(s) for s in structure.species],
            structure.frac_coords,
        )

    @property
    def lattice(self):
        """
        Returns:
            Lattice object (in ANGSTROM no matter what's in the input file)
        """
        # TODO: move to validate
        try:
            ibrav = self.system["ibrav"]
        except KeyError as e:
            raise ValueError("ibrav must be set in system namelist") from e
        if ibrav != 0:
            return ibrav_to_lattice(ibrav, self.celldm)
        if self.cell_parameters is None:
            raise ValueError("cell_parameters must be set if ibrav=0")
        lattice_matrix = np.stack(
            (
                self.cell_parameters.a1,
                self.cell_parameters.a2,
                self.cell_parameters.a3,
            )
        )
        if self.cell_parameters.option == CellParametersCard.opts.alat:
            lattice_matrix *= self.alat
        elif self.cell_parameters.option == CellParametersCard.opts.bohr:
            lattice_matrix *= bohr_to_ang
        elif self.cell_parameters.option != CellParametersCard.opts.angstrom:
            raise ValueError(
                f"cell_parameters option must be one of 'alat', 'bohr', or 'angstrom'. {self.cell_parameters.option} is not supported."
            )
        return Lattice(lattice_matrix)

    @lattice.setter
    def lattice(self, lattice):
        """
        Args:
            lattice (Lattice): Lattice object to replace the current lattice with
        """
        # Adjust the lattice related tags in the system namelist
        if self.system is None:
            self.system = SystemNamelist()
        self.system["ibrav"] = 0
        keys = ["celldm", "A", "B", "C", "cosAB", "cosAC", "cosBC"]
        for key in keys:
            with contextlib.suppress(KeyError):
                del self.system[key]

        self.cell_parameters = CellParametersCard(
            CellParametersCard.opts.angstrom,
            lattice.matrix[0],
            lattice.matrix[1],
            lattice.matrix[2],
        )

    @property
    def alat(self):
        """
        Returns alat (either celldm(1) or A) in ANGSTROM with proper error handling
        """
        celldm = copy(self.system.get("celldm", None))
        A = self.system.get("A", None)
        # TODO: move to validate
        if celldm is None and A is None:
            raise ValueError("either celldm(1) or A must be set if any cards options are alat.")
        if celldm is not None and A is not None:
            raise ValueError("celldm(1) and A cannot both be set.")
        return celldm[0] * bohr_to_ang if celldm is not None else A

    @property
    def celldm(self):
        """
        Gets celldm from the input file.
        If celldm is in the input file, returns it with the first element converted to angstrom and padded with zeros to length 6.
        If A is in the input instead, then it returns:
            celldm = [A, B/A, C/A, cosBC, cosAC, cosAB] (with A in angstrom)
        with missing values padded to zeros

        Returns:
            celldm (list): list of celldm parameters, with shape (6,)
        """

        def get_celldm_from_ABC():
            # A is already in angstrom
            B = self.system.get("B", 0)
            C = self.system.get("C", 0)
            cosAB = self.system.get("cosAB", 0)
            cosAC = self.system.get("cosAC", 0)
            cosBC = self.system.get("cosBC", 0)
            return [A, B / A, C / A, cosBC, cosAC, cosAB]

        celldm = copy(self.system.get("celldm", None))
        A = self.system.get("A", None)
        # TODO: move to validate
        if celldm is None and A is None:
            raise ValueError("either celldm(1) or A must be set if ibrav != 0")
        if celldm is not None and A is not None:
            raise ValueError("celldm(1) and A cannot both be set.")
        if celldm is not None:
            celldm[0] *= bohr_to_ang  # celldm(1) is in bohr
            # Get it to the right length since not all 6 are required in input
            celldm = np.pad(celldm, (0, 6 - len(celldm)))
        elif A is not None:
            celldm = get_celldm_from_ABC()
        return celldm


class PWinParserError(Exception):
    """
    Exception class for PWin parsing.
    """
