"""
Classes for reading/manipulating PWscf xml files.
"""


from __future__ import annotations

import contextlib
import datetime
import itertools
import logging
import math
import os
import pathlib
import re
import warnings
from io import StringIO
from copy import copy, deepcopy
from collections import OrderedDict
from enum import Enum

import numpy as np
from monty.json import MSONable
from monty.os.path import zpath
import f90nml
import pandas as pd
from tabulate import tabulate

from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure, Species, Site
from pymatgen.core.units import (
    unitized,
    Ha_to_eV,
    Ry_to_eV,
    eV_to_Ha,
    bohr_to_ang,
    ang_to_bohr,
)
from pymatgen.io.espresso.pwin_cards import PWinCards

# from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.espresso.utils import parse_pwvals, ibrav_to_lattice


class PWin(MSONable):
    """
    Class for PWscf input files
    """

    all_card_names = [c.name for c in PWinCards]

    # First three are required, rest are optional
    _all_namelists = [
        "control",
        "system",
        "electrons",
        "ions",
        "cell",
        "fcp",
        "rism",
    ]

    # TODO: doc string
    def __init__(self, namelists, cards, filename=None, bad_PWin_warning=True):
        """
        Args:
            namelists: dict of dicts of namelists
            cards: dict of dicts of cards
            filename (str): filename
            bad_PWin_warning (bool): Whether to warn if the PW input file is not
                valid (only a few checks currently implemented).
                Defaults to True.
        """
        self.bad_PWin_warning = bad_PWin_warning
        self.filename = filename

        # TODO: do this in a more elegant way
        # Also assigning empty dictionaries to namelists
        # and cards if they are None, which allows easier
        # assignment
        self.control = namelists.get("control", None)
        self.system = namelists.get("system", None)
        self.electrons = namelists.get("electrons", None)
        self.ions = namelists.get("ions", None)
        self.cell = namelists.get("cell", None)
        self.fcp = namelists.get("fcp", None)
        self.rism = namelists.get("rism", None)

        self.cards = OrderedDict({c.name: cards.get(c.name, None) for c in PWinCards})

        self._structure = None
        self._lattice = None

        self._validate()

    @property
    def atomic_species(self):
        return self.cards["atomic_species"]

    @atomic_species.setter
    def atomic_species(self, value):
        if not isinstance(value, PWinCards.atomic_species.value):
            raise TypeError(f"atomic_species must be of type {PWinCards.atomic_species.value}")
        self.cards["atomic_species"] = value

    @atomic_species.deleter
    def atomic_species(self):
        self.cards["atomic_species"] = None

    @property
    def atomic_positions(self):
        return self.cards["atomic_positions"]

    @atomic_positions.setter
    def atomic_positions(self, value):
        if not isinstance(value, PWinCards.atomic_positions.value):
            raise TypeError(f"atomic_positions must be of type {PWinCards.atomic_positions.value}")
        self.cards["atomic_positions"] = value

    @atomic_positions.deleter
    def atomic_positions(self):
        self.cards["atomic_positions"] = None

    @property
    def k_points(self):
        return self.cards["k_points"]

    @k_points.setter
    def k_points(self, value):
        if not isinstance(value, PWinCards.k_points.value):
            raise TypeError(f"k_points must be of type {PWinCards.k_points.value}")
        self.cards["k_points"] = value

    @k_points.deleter
    def k_points(self):
        self.cards["k_points"] = None

    @property
    def additional_k_points(self):
        return self.cards["additional_k_points"]

    @additional_k_points.setter
    def additional_k_points(self, value):
        if not isinstance(value, PWinCards.additional_k_points.value):
            raise TypeError(
                f"additional_k_points must be of type {PWinCards.additional_k_points.value}"
            )
        self.cards["additional_k_points"] = value

    @additional_k_points.deleter
    def additional_k_points(self):
        self.cards["additional_k_points"] = None

    @property
    def cell_parameters(self):
        return self.cards["cell_parameters"]

    @cell_parameters.setter
    def cell_parameters(self, value):
        if not isinstance(value, PWinCards.cell_parameters.value):
            raise TypeError(f"cell_parameters must be of type {PWinCards.cell_parameters.value}")
        self.cards["cell_parameters"] = value

    @cell_parameters.deleter
    def cell_parameters(self):
        self.cards["cell_parameters"] = None

    @property
    def constraints(self):
        return self.cards["constraints"]

    @constraints.setter
    def constraints(self, value):
        if not isinstance(value, PWinCards.constraints.value):
            raise TypeError(f"constraints must be of type {PWinCards.constraints.value}")
        self.cards["constraints"] = value

    @constraints.deleter
    def constraints(self):
        self.cards["constraints"] = None

    @property
    def occupations(self):
        return self.cards["occupations"]

    @occupations.setter
    def occupations(self, value):
        if not isinstance(value, PWinCards.occupations.value):
            raise TypeError(f"occupations must be of type {PWinCards.occupations.value}")
        self.cards["occupations"] = value

    @occupations.deleter
    def occupations(self):
        self.cards["occupations"] = None

    @property
    def atomic_velocities(self):
        return self.cards["atomic_velocities"]

    @atomic_velocities.setter
    def atomic_velocities(self, value):
        if not isinstance(value, PWinCards.atomic_velocities.value):
            raise TypeError(
                f"atomic_velocities must be of type {PWinCards.atomic_velocities.value}"
            )
        self.cards["atomic_velocities"] = value

    @atomic_velocities.deleter
    def atomic_velocities(self):
        self.cards["atomic_velocities"] = None

    @property
    def atomic_forces(self):
        return self.cards["atomic_forces"]

    @atomic_forces.setter
    def atomic_forces(self, value):
        if not isinstance(value, PWinCards.atomic_forces.value):
            raise TypeError(f"atomic_forces must be of type {PWinCards.atomic_forces.value}")
        self.cards["atomic_forces"] = value

    @atomic_forces.deleter
    def atomic_forces(self):
        self.cards["atomic_forces"] = None

    @property
    def solvents(self):
        return self.cards["solvents"]

    @solvents.setter
    def solvents(self, value):
        if not isinstance(value, PWinCards.solvents.value):
            raise TypeError(f"solvents must be of type {PWinCards.solvents.value}")
        self.cards["solvents"] = value

    @solvents.deleter
    def solvents(self):
        self.cards["solvents"] = None

    @property
    def hubbard(self):
        return self.cards["hubbard"]

    @hubbard.setter
    def hubbard(self, value):
        if not isinstance(value, PWinCards.hubbard.value):
            raise TypeError(f"hubbard must be of type {PWinCards.hubbard.value}")
        self.cards["hubbard"] = value

    @hubbard.deleter
    def hubbard(self):
        self.cards["hubbard"] = None

    @classmethod
    def from_file(cls, filename, suppress_bad_PWin_warn=False):
        """
        Reads a PWin from file

        Args:
            string (str): String to parse.
            suppress_bad_PWin_warn (bool): Whether to suppress warnings for bad PWin files.

        Returns:
            PWin object
        """
        parser = f90nml.Parser()
        parser.comment_tokens += "#"

        pwi_str = pathlib.Path(filename).read_text()
        namelists = parser.reads(pwi_str)
        namelists = namelists.todict()
        cards = cls._parse_cards(pwi_str)

        return cls(namelists, cards, filename, suppress_bad_PWin_warn)

    def to_str(self, indent=2):
        """
        Return the PWscf input file as a string
        """
        self._validate()
        namelists = {}
        # Some of the namelists can be {}, so we test against None instead of truthiness
        if self.control is not None:
            namelists["control"] = self.control
        if self.system is not None:
            namelists["system"] = self.system
        # Creating the namelist now helps preserve order for some reason
        namelists = f90nml.namelist.Namelist(namelists)  # type: ignore
        if self.electrons is not None:
            namelists.update({"electrons": self.electrons})
        if self.ions is not None:
            namelists.update({"ions": self.ions})
        if self.cell is not None:
            namelists.update({"cell": self.cell})
        if self.fcp is not None:
            namelists.update({"fcp": self.fcp})
        if self.rism is not None:
            namelists.update({"rism": self.rism})

        stream = StringIO()
        namelists.indent = indent * " "
        namelists.write(stream)
        string = stream.getvalue()
        # Strip empty lines between namelists
        string = re.sub(r"\n\s*\n", "\n", string)

        for c in self.cards.values():
            string += str(c) if c is not None else ""

        return string

    def to_file(self, filename="pw.in", indent=2, overwrite=False):
        """
        Save the PWscf input file to a file
        """
        string = self.to_str(indent=indent)
        ascii_str = string.encode("ascii")
        if not overwrite and os.path.exists(filename):
            raise IOError("File exists! Use overwrite=True to force overwriting.")
        with open(filename, "wb") as f:
            f.write(ascii_str)

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
        atomic_positions = self.atomic_positions
        species = atomic_positions.symbols
        coords = atomic_positions.positions
        if atomic_positions.option == atomic_positions.opts.alat:
            coords *= self.alat
            coords_are_cartesian = True
        elif atomic_positions.option == atomic_positions.opts.bohr:
            coords *= bohr_to_ang
            coords_are_cartesian = True
        elif atomic_positions.option == atomic_positions.opts.angstrom:
            coords_are_cartesian = True
        elif atomic_positions.option == atomic_positions.opts.crystal:
            coords_are_cartesian = False
        elif atomic_positions.option == atomic_positions.opts.crystal_sg:
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
            self.system = OrderedDict()
        self.system["nat"] = len(structure.species)
        species = set(structure.species)
        self.system["ntyp"] = len(species)

        if self.atomic_species is not None:
            new_symbols = {str(s) for s in species}
            if self.atomic_species.symbols == new_symbols:
                return
            else:
                warnings.warn(
                    "The atomic species in the input file does not "
                    "match the species in the structure object. "
                    "The atomic species in the input file will be overwritten."
                )
            Card = PWinCards.atomic_species.value
            self.atomic_species = Card(
                None,
                [str(s) for s in species],
                [s.atomic_mass for s in species],
                [f"{s}.UPF" for s in species],
            )
        Card = PWinCards.atomic_positions.value
        self.atomic_positions = Card(
            Card.opts.crystal, [str(s) for s in structure.species], structure.frac_coords
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
        if ibrav == 0:
            if self.cell_parameters is None:
                raise ValueError("cell_parameters must be set if ibrav=0")
            lattice_matrix = np.stack(
                (
                    self.cell_parameters.a1,
                    self.cell_parameters.a2,
                    self.cell_parameters.a3,
                )
            )
            if self.cell_parameters.option == self.cell_parameters.opts.alat:
                lattice_matrix *= self.alat
            elif self.cell_parameters.option == self.cell_parameters.opts.bohr:
                lattice_matrix *= bohr_to_ang
            elif self.cell_parameters.option != self.cell_parameters.opts.angstrom:
                raise ValueError(
                    f"cell_parameters option must be one of 'alat', 'bohr', or 'angstrom'. {self.cell_parameters.option} is not supported."
                )
            return Lattice(lattice_matrix)
        else:
            celldm = self.celldm()
            return ibrav_to_lattice(ibrav, celldm)

    @lattice.setter
    def lattice(self, lattice):
        """
        Args:
            lattice (Lattice): Lattice object to replace the current lattice with
        """
        # Adjust the lattice related tags in the system namelist
        if self.system is None:
            self.system = OrderedDict()
        self.system["ibrav"] = 0
        keys = ["celldm", "A", "B", "C", "cosAB", "cosAC", "cosBC"]
        for key in keys:
            with contextlib.suppress(KeyError):
                del self.system[key]

        # Adjust the cell_parameters card
        Card = PWinCards.cell_parameters.value
        self.cell_parameters = Card(
            Card.opts.angstrom, lattice.matrix[0], lattice.matrix[1], lattice.matrix[2]
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

        # TODO: move to validate
        celldm = copy(self.system.get("celldm", None))
        A = self.system.get("A", None)
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

    @classmethod
    def _parse_cards(cls, pwi_str):
        cards_strs = pwi_str.rsplit("/", 1)[1].split("\n")
        cards_strs = [c for c in cards_strs if c]
        card_idx = [
            i
            for i, string in enumerate(cards_strs)
            if string.split()[0].lower() in cls.all_card_names
        ]
        cards = {}
        for i, j in zip(card_idx, card_idx[1:] + [None]):  # type: ignore
            card_name = cards_strs[i].split()[0].lower()
            card_string = "\n".join(cards_strs[i:j])
            Card = PWinCards.from_string(card_name)
            cards[card_name] = Card.from_string(card_string)

        return cards

    def _validate(self):
        required_namelists = [self.control, self.system, self.electrons]
        if all(required_namelists):
            valid_namelists = True
        else:
            valid_namelists = False
            msg = "PWscf input file is missing required namelists:"
            for i, nml in enumerate(required_namelists):
                if not nml:
                    msg += f" &{self._all_namelists[i].upper()}"
            msg += ". Partial data available."
            if self.bad_PWin_warning:
                warnings.warn(msg, UserWarning)

        required_cards = [self.atomic_species, self.atomic_positions, self.k_points]
        if all(required_cards):
            valid_cards = True
        else:
            valid_cards = False
            msg = "PWscf input file is missing required cards:"
            for i, nml in enumerate(required_cards):
                if not nml:
                    msg += f" {self.all_card_names[i].upper()}"
            msg += ". Partial data available."
            if self.bad_PWin_warning:
                warnings.warn(msg, UserWarning)

        return valid_namelists and valid_cards


class PWinParserError(Exception):
    """
    Exception class for PWin parsing.
    """
