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

    _indent = 2
    all_card_names = [c.name for c in PWinCards]
    all_namelist_names = [
        "control",
        "system",
        "electrons",
        "ions",
        "cell",
        "fcp",
        "rism",
    ]
    namelists_required = [True, True, True, False, False, False, False]

    # TODO: doc string
    def __init__(self, namelists, cards, bad_PWin_warning=True):
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

        self.cards = OrderedDict({c.name: cards.get(c.name, None) for c in PWinCards})
        self.namelists = OrderedDict({n: namelists.get(n, None) for n in self.all_namelist_names})

        for prop_name in self.all_card_names + self.all_namelist_names:
            setattr(
                self.__class__,
                prop_name,
                property(
                    self._make_getter(prop_name),
                    self._make_setter(prop_name),
                    self._make_deleter(prop_name),
                ),
            )

    def _make_getter(self, name):
        if name in self.all_card_names:
            return lambda self: self.cards[name]
        elif name in self.all_namelist_names:
            return lambda self: self.namelists[name]

    def _make_setter(self, name):
        if name in self.all_card_names:

            def setter(self, value):
                if not isinstance(value, c := PWinCards.from_string(name)):
                    raise TypeError(f"{name} must be of type {c}")
                self.cards[name] = value

            return setter
        elif name in self.all_namelist_names:

            def setter(self, value):
                if not isinstance(value, OrderedDict):
                    raise TypeError(f"{name} must be of type OrderedDict")
                self.namelists[name] = value

            return setter

    def _make_deleter(self, name):
        if name in self.all_card_names:
            return lambda self: self.cards.__setitem__(name, None)
        elif name in self.all_namelist_names:
            return lambda self: self.namelists.__setitem__(name, None)

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

        return cls(namelists, cards, suppress_bad_PWin_warn)

    def __str__(self):
        """
        Return the PWscf input file as a string
        """
        string = ""
        for n, v in self.namelists.items():
            if v is not None:
                nl = f90nml.namelist.Namelist({n: v})
                nl.indent = self._indent * " "
                string += str(nl) + "\n"
        # Upper case namelists (e.g., &CONTROL)
        string = re.sub(r"^&(\w+)", lambda m: m.group(0).upper(), string, flags=re.MULTILINE)

        for c in self.cards.values():
            if c is not None:
                c.indent = self._indent
                string += str(c)

        return string

    def to_file(self, filename, indent=2):
        """
        Save the PWscf input file to a file
        """
        self._indent = indent
        with open(filename, "wb") as f:
            f.write(self.__str__().encode("ascii"))

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

    def validate(self):
        """
        Very basic validation for the input file.
        Currently only checks that required namelists and cards are present.
        """
        required_namelist_names = [
            nml for nml in self.all_namelist_names if nml in self.namelists_required
        ]
        if any(self.namelists[nml] is None for nml in required_namelist_names):
            msg = "PWscf input file is missing required namelists:"
            for nml in required_namelist_names:
                if self.namelists[nml] is None:
                    msg += f" &{nml.upper()}"
            warnings.warn(msg, PWinParserWarning)

        required_card_names = [c.name for c in PWinCards if c.value.required]
        if any(self.cards[card] is None for card in required_card_names):
            msg = "PWscf input file is missing required cards:"
            for card in required_card_names:
                if self.cards[card] is None:
                    msg += f" {card.upper()}"
            warnings.warn(msg, PWinParserWarning)


class PWinParserError(Exception):
    """
    Exception class for PWin parsing.
    """


# Custom warning for invalid input
class PWinParserWarning(UserWarning):
    """
    Warning class for PWin parsing.
    """
